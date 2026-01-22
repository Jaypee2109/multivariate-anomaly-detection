import os

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from transformer import TransformerTimeSeriesWithLearnableTime2Vec

# -------------------------------------------------
# Dataset loading
# -------------------------------------------------


def normalize_series(values):
    """
    Normalize each series (column) independently using z-score.

    Args:
        values: np.ndarray of shape (T, num_series)

    Returns:
        normalized_values: np.ndarray same shape as values
        mean: np.ndarray shape (num_series,)
        std: np.ndarray shape (num_series,)
    """
    mean = values.mean(axis=0, keepdims=True)
    std = values.std(axis=0, keepdims=True)
    normalized_values = (values - mean) / (std + 1e-8)
    return normalized_values, mean.squeeze(), std.squeeze()


def denormalize_series(values, mean, std):
    """
    Undo z-score normalization.

    Args:
        values: np.ndarray of shape (T, num_series)
        mean: np.ndarray shape (num_series,)
        std: np.ndarray shape (num_series,)

    Returns:
        denormalized_values: np.ndarray same shape as values
    """
    return values * std + mean


def load_raw_electricity(file_path):
    df = pd.read_csv(
        file_path,
        sep=";",
        decimal=",",
        parse_dates=["timestamp"],
    ).sort_values("timestamp")

    timestamps = df["timestamp"].values
    values = df.drop(columns=["timestamp"]).values.astype(np.float32)

    return timestamps, values


def setup_distributed():
    rank = int(os.environ["SLURM_PROCID"])
    local_rank = int(os.environ["SLURM_LOCALID"])
    world_size = int(os.environ["SLURM_NTASKS"])

    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    torch.cuda.set_device(local_rank)

    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        rank=rank,
        world_size=world_size,
    )

    return rank, local_rank, world_size


class ElectricityWindowDataset(Dataset):
    def __init__(self, values, timestamps, lag, split, ratios=(0.75, 0.15)):
        self.values = values
        self.timestamps = pd.to_datetime(timestamps)
        self.lag = lag

        n = len(timestamps)
        tr, va = int(ratios[0] * n), int((ratios[0] + ratios[1]) * n)

        if split == "train":
            self.start, self.end = 0, tr
        elif split == "val":
            self.start, self.end = tr, va
        elif split == "test":
            self.start, self.end = va, n
        else:
            raise ValueError(f"Unknown split {split}")

        self.num_series = values.shape[1]
        self.windows_per_series = (self.end - self.start) - lag
        assert self.windows_per_series > 0

    def __len__(self):
        return self.num_series * self.windows_per_series

    def __getitem__(self, idx):
        sid = idx // self.windows_per_series
        t = idx % self.windows_per_series + self.start

        x = self.values[t : t + self.lag, sid]
        y = self.values[t + self.lag, sid]

        ts = self.timestamps[t : t + self.lag]
        tf_x = self._time_features(ts, sid)
        tf_y = self._time_features([self.timestamps[t + self.lag]], sid, target=True)

        return (
            torch.from_numpy(x[:, None]),
            torch.tensor(y)[None],
            torch.from_numpy(tf_x),
            torch.from_numpy(tf_y),
        )

    def _time_features(self, ts, sid, target=False):
        ts = pd.to_datetime(ts)
        feats = np.stack(
            [
                [
                    t.hour / 23,
                    (t.minute // 15) / 3,
                    t.weekday() / 6,
                    t.dayofyear / 365,
                    sid / self.num_series,
                ]
                for t in ts
            ],
            axis=0,
        ).astype(np.float32)

        if target:
            feats = np.concatenate([feats[0], [1.0 / self.lag]])
        return feats


def build_ar_test_windows(values, timestamps, lag, horizons, max_windows=100):
    n, num_series = values.shape
    max_h = max(horizons)
    windows = []
    rng = np.random.default_rng(42)

    for _ in range(max_windows):
        sid = rng.integers(0, num_series)
        t = rng.integers(lag, max(lag + 1, n - max_h))

        init_vals = values[t - lag : t, sid]
        init_ts = timestamps[t - lag : t]
        true_vals = values[t : t + max_h, sid]

        windows.append((init_vals, init_ts, true_vals, sid))  # include sid
    return windows


def build_dataloaders(
    values,
    timestamps,
    lag,
    batch_size,
    world_size,
    rank,
    ratios=(0.75, 0.15, 0.1),
):
    """
    Builds DDP-safe DataLoaders for electricity dataset using lazy windowing.

    Args:
        values: np.ndarray of shape (T, num_series)
        timestamps: np.ndarray of datetime64
        lag: int, window size
        batch_size: int
        world_size: int, total DDP ranks
        rank: int, current rank
        ratios: tuple, (train_ratio, val_ratio, test_ratio)

    Returns:
        train_loader, val_loader, train_sampler
    """
    # Datasets
    train_ds = ElectricityWindowDataset(
        values, timestamps, lag, split="train", ratios=ratios
    )
    val_ds = ElectricityWindowDataset(
        values, timestamps, lag, split="val", ratios=ratios
    )

    test_ds = ElectricityWindowDataset(
        values, timestamps, lag, split="test", ratios=ratios
    )

    train_sampler = DistributedSampler(
        train_ds,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    if rank == 0:
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            persistent_workers=True,
        )

        test_loader = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            persistent_workers=True,
        )
    else:
        val_loader = None
        test_loader = None

    return train_loader, val_loader, test_loader, train_sampler


def teacher_forcing_ratio(epoch, max_epochs, min_ratio=0.5):
    return max(min_ratio, 1.0 - epoch / max_epochs)


def apply_scheduled_sampling(model, xb, tfxb, tfyb, ratio):
    """
    Applies proper scheduled sampling to a batch of sequences.

    xb: (B, lag, 1)       - input sequences
    tfxb: (B, lag, TFx)   - time features for input
    tfyb: (B, TFy)        - time features for target step
    ratio: float          - teacher forcing ratio (0 = fully model, 1 = fully teacher)

    Returns:
        xb_new: (B, lag, 1) - sequence with some inputs replaced by predictions
    """
    if ratio >= 1.0:
        return xb  # full teacher forcing

    B, L, _ = xb.shape
    device = xb.device
    xb_new = xb.clone()  # start with the true inputs

    # Iterate over the lag dimension
    for t in range(L):
        # Only replace with model prediction according to mask
        mask = (torch.rand(B, device=device) > ratio).float()  # 1 = use prediction

        if mask.sum() == 0:
            continue  # skip if all teacher forcing

        # Take the sequence up to current timestep t for model input
        x_input = xb_new[:, : t + 1, :]
        tfx_input = tfxb[:, : t + 1, :]

        with torch.no_grad():
            # Model predicts next step
            pred = model(torch.cat([x_input, tfx_input], dim=-1), tfy=tfyb)  # (B, 1)

        # Replace positions in xb_new where mask == 1
        pred = pred.unsqueeze(1)  # (B,1,1)
        xb_new[:, t : t + 1, :] = (
            xb_new[:, t : t + 1, :] * (1 - mask[:, None, None])
            + pred * mask[:, None, None]
        )

    return xb_new


def forecast_autoregressive(
    model,
    init_vals,
    init_ts,
    steps,
    lag,
    device,
    series_mean,
    series_std,
    sid,
):
    model.eval()
    vals = list(init_vals)
    ts = list(pd.to_datetime(init_ts))
    preds = []

    for step in range(steps):
        hours = np.array([t.hour / 23 for t in ts[-lag:]], dtype=np.float32)
        minutes = np.array([(t.minute // 15) / 3 for t in ts[-lag:]], dtype=np.float32)
        wdays = np.array([t.weekday() / 6 for t in ts[-lag:]], dtype=np.float32)
        doy = np.array([t.dayofyear / 365 for t in ts[-lag:]], dtype=np.float32)

        tfx = np.stack([hours, minutes, wdays, doy], axis=-1)
        x = torch.tensor(vals[-lag:], device=device)[None, :, None]
        tfx = torch.tensor(tfx, device=device)[None]

        tfy = torch.tensor(
            [
                [
                    ts[-1].hour / 23,
                    (ts[-1].minute // 15) / 3,
                    ts[-1].weekday() / 6,
                    ts[-1].dayofyear / 365,
                    (step + 1) / steps,
                ]
            ],
            device=device,
        )

        with torch.no_grad():
            pred = model(torch.cat([x, tfx], dim=-1), tfy=tfy).item()

        pred_denorm = pred * series_std[sid] + series_mean[sid]

        preds.append(pred_denorm)
        vals.append(pred)
        ts.append(ts[-1] + (ts[-1] - ts[-2]))

    return preds


def evaluate_ar_quick(
    model,
    ar_windows,
    horizons,
    lag,
    device,
    series_mean,
    series_std,
):
    metrics = {h: {"mae": [], "rmse": []} for h in horizons}
    model.eval()

    for init_vals, init_ts, true_vals, sid in ar_windows:  # unpack sid
        preds = forecast_autoregressive(
            model,
            init_vals,
            init_ts,
            max(horizons),
            lag,
            device,
            series_mean,
            series_std,
            sid=sid,
        )

        preds = np.array(preds)
        true_vals = np.array(true_vals)

        # Denormalize true values too
        true_vals = true_vals * series_std[sid] + series_mean[sid]

        for h in horizons:
            err = preds[:h] - true_vals[:h]
            mae = np.mean(np.abs(err))
            rmse = np.sqrt(np.mean(err**2))

            metrics[h]["mae"].append(mae)
            metrics[h]["rmse"].append(rmse)

    return {
        h: {
            "MAE": float(np.mean(metrics[h]["mae"])),
            "RMSE": float(np.mean(metrics[h]["rmse"])),
        }
        for h in horizons
    }


def train_ddp(
    model,
    train_loader,
    val_loader,
    train_sampler,
    optimizer,
    scheduler,
    criterion,
    epochs,
    device,
    rank,
    ar_windows=None,
    ar_eval_every=1,
    lag=96,
    save_path=None,
    series_mean=None,
    series_std=None,
):
    best_val = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        train_sampler.set_epoch(epoch)

        tf_ratio = teacher_forcing_ratio(epoch, epochs)
        total_loss = 0.0

        for xb, yb, tfxb, tfyb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            tfxb = tfxb.to(device, non_blocking=True)
            tfyb = tfyb.to(device, non_blocking=True)

            xb = apply_scheduled_sampling(
                model,
                xb,
                tfxb,
                tfyb,
                tf_ratio,
            )

            inp = torch.cat([xb, tfxb], dim=-1)

            optimizer.zero_grad()
            out = model(inp, tfy=tfyb)
            loss = criterion(out, yb)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        # -------- Validation + AR logging (rank 0)
        if rank == 0:
            model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for xb, yb, tfxb, tfyb in val_loader:
                    xb = xb.to(device)
                    yb = yb.to(device)
                    tfxb = tfxb.to(device)
                    tfyb = tfyb.to(device)

                    out = model(torch.cat([xb, tfxb], dim=-1), tfy=tfyb)
                    val_loss += criterion(out, yb).item()

            val_loss /= max(1, len(val_loader))

            ar_str = ""
            if ar_windows and epoch % ar_eval_every == 0:
                ar_metrics = evaluate_ar_quick(
                    model.module,
                    ar_windows,
                    horizons=(1, 5, 10, 20),
                    lag=lag,
                    device=device,
                    series_mean=series_mean,
                    series_std=series_std,
                )

                ar_str = " | AR "
                for h, m in ar_metrics.items():
                    ar_str += f"h{h}(MAE:{m['MAE']:.3f},RMSE:{m['RMSE']:.3f}) "

            print(
                f"Epoch {epoch:03d} | "
                f"TF {tf_ratio:.2f} | "
                f"Train {total_loss / len(train_loader):.4f} | "
                f"Val {val_loss:.4f}"
                f"{ar_str}",
                flush=True,
            )

            if val_loss < best_val:
                best_val = val_loss
                torch.save(model.module.state_dict(), save_path)

        dist.barrier()


def evaluate_one_step(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for xb, yb, tfxb, tfyb in test_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            tfxb = tfxb.to(device)
            tfyb = tfyb.to(device)

            out = model(torch.cat([xb, tfxb], dim=-1), tfy=tfyb)
            total_loss += criterion(out, yb).item()

    return total_loss / max(1, len(test_loader))


def main():
    torch.set_default_dtype(torch.float32)
    # -------------------------------
    # Dynamically determine MASTER_ADDR
    # -------------------------------
    rank, local_rank, world_size = setup_distributed()

    device = torch.device("cuda", local_rank)
    torch.set_float32_matmul_precision("medium")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device = torch.device("cuda", local_rank)
    torch.set_float32_matmul_precision("medium")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    MODEL_DIM = 384
    NUM_HEADS = 12
    NUM_LAYERS = 6
    EPOCHS = 10
    DIM_FEEDFORWARD = 1536
    TFx_DIM = 5
    TFyDIM = 6
    AR_EVAL_EVERY = 2

    model = TransformerTimeSeriesWithLearnableTime2Vec(
        t2v_dim=16,
        model_dim=MODEL_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        dropout=0.1,
        dim_feedforward=DIM_FEEDFORWARD,
        tfx_dim=TFx_DIM,
        tfy_dim=TFyDIM,
    ).to(device)

    # ---- Load data (every rank loads)
    lag = 96

    timestamps, values = load_raw_electricity(
        "/home/sc.uni-leipzig.de/uj74reda/Transformer/electricity/LD2011_2014.txt"
    )

    values, series_mean, series_std = normalize_series(values)

    train_loader, val_loader, test_loader, train_sampler = build_dataloaders(
        values=values,
        timestamps=timestamps,
        lag=lag,
        batch_size=64,
        world_size=world_size,
        rank=rank,
    )

    if rank == 0:
        print("Param count before DDP:", sum(p.numel() for p in model.parameters()))
    dist.barrier()
    print(
        f"Rank {rank}/{world_size} | "
        f"Local rank {local_rank} | "
        f"GPU {torch.cuda.current_device()}",
        flush=True,
    )

    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False,
    )
    ddp_param_count = sum(p.numel() for p in model.parameters())
    print(f"[Rank {rank}] param count AFTER DDP: {ddp_param_count}")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=1e-3,
        epochs=EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
    )

    criterion = torch.nn.MSELoss()

    SAVE_LOC = f"/home/sc.uni-leipzig.de/uj74reda/Transformer/models/electricity_model_learnable_t2v_{EPOCHS}_epochs_{NUM_HEADS}_heads_{NUM_LAYERS}_layers_{MODEL_DIM}_dim_{DIM_FEEDFORWARD}_dim_feedforward_{TFx_DIM}_tfx_dim_{TFyDIM}_tfy_dim_ddp_fixed"

    train_ddp(
        model,
        train_loader,
        val_loader,
        train_sampler,
        optimizer,
        scheduler,
        criterion,
        epochs=EPOCHS,
        device=device,
        rank=rank,
        save_path=SAVE_LOC,
        ar_eval_every=AR_EVAL_EVERY,
    )

    if rank == 0:
        test_loss = evaluate_one_step(
            model.module,
            test_loader,
            criterion,
            device,
        )
        print(f"\n==== TEST ONE-STEP MSE ====\n{test_loss:.4f}")

        print("\n==== FINAL AUTOREGRESSIVE TEST ====\n")
        ar_windows = build_ar_test_windows(
            values,
            timestamps,
            lag=lag,
            horizons=(1, 5, 10, 20),
            max_windows=200,
        )

        ar_metrics = evaluate_ar_quick(
            model.module,
            ar_windows,
            horizons=(1, 5, 10, 20),
            lag=lag,
            device=device,
            series_mean=series_mean,
            series_std=series_std,
        )

        for h, m in ar_metrics.items():
            print(f"H{h:02d} | MAE {m['MAE']:.4f} | RMSE {m['RMSE']:.4f}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
