import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

from transformer import TransformerTimeSeriesWithLearnableTime2Vec
from torch.amp import autocast, GradScaler

# -------------------------------------------------
# Setup
# -------------------------------------------------
torch.set_float32_matmul_precision("medium")

device = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available() else "cpu"
)

if device.type in ("cpu", "mps"):
    el_path = "data/raw/small.txt"
else:
    el_path = "/home/sc.uni-leipzig.de/uj74reda/Transformer/electricity/LD2011_2014.txt"


# -------------------------------------------------
# Dataset loading
# -------------------------------------------------

from torch.utils.data import Dataset, DataLoader


class ElectricityWindowDataset(Dataset):
    def __init__(self, df, lag=96, split="train", train_ratio=0.75, val_ratio=0.15):
        """
        df: pandas DataFrame with columns [timestamp, series1, series2, ...]
        split: "train", "val", "test"
        """
        self.lag = lag
        self.series_cols = [c for c in df.columns if c != "timestamp"]
        self.timestamps = pd.to_datetime(df["timestamp"].values)
        self.split = split

        n = len(df)
        train_end = int(train_ratio * n)
        val_end = int((train_ratio + val_ratio) * n)

        if split == "train":
            self.indices = slice(0, train_end)
        elif split == "val":
            self.indices = slice(train_end, val_end)
        else:
            self.indices = slice(val_end, n)

        # Keep series values normalized individually
        self.series_data = []
        for col in self.series_cols:
            values = df[col].values.astype(np.float32)
            mean = values[:train_end].mean()
            std = values[:train_end].std() + 1e-6
            values = (values - mean) / std
            self.series_data.append(values[self.indices])

        self.series_data = np.stack(self.series_data, axis=1)  # shape: (T, num_series)
        self.timestamps = self.timestamps[self.indices]

        self.num_series = len(self.series_cols)
        self.length = len(self.series_data) - lag

    def __len__(self):
        return self.length * self.num_series  # all series windows

    def __getitem__(self, idx):
        series_id = idx % self.num_series
        t_idx = idx // self.num_series

        x = self.series_data[t_idx : t_idx + self.lag, series_id][:, None]
        y = np.array([self.series_data[t_idx + self.lag, series_id]], dtype=np.float32)

        ts_window = self.timestamps[t_idx : t_idx + self.lag + 1]  # for TFx & TFy
        TFx = self.build_time_features(ts_window[: self.lag])
        TFy = self.build_time_features(ts_window[self.lag :], for_y=True)
        TFy = np.squeeze(TFy, axis=0)  # remove extra dimension
        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
            torch.tensor(TFx, dtype=torch.float32),
            torch.tensor(TFy, dtype=torch.float32),
        )

    def build_time_features(self, ts, for_y=False):
        hour = np.array([t.hour / 23.0 for t in ts], dtype=np.float32)
        quarter = np.array([(t.minute // 15) / 3.0 for t in ts], dtype=np.float32)
        weekday = np.array([t.weekday() / 6.0 for t in ts], dtype=np.float32)
        dayofyear = np.array([t.dayofyear / 365.0 for t in ts], dtype=np.float32)
        tf = np.stack([hour, quarter, weekday, dayofyear], axis=-1)
        if for_y:
            # Append lag scaling
            tf = np.concatenate([tf, np.array([[1.0 / self.lag]] * len(tf))], axis=-1)
        return tf


# -------------------------------------------------
# Window builder
# -------------------------------------------------
def build_windows_electricity(values, timestamps, lag):
    values = np.asarray(values, dtype=np.float32)
    timestamps = pd.to_datetime(timestamps)

    T = len(values)
    if T <= lag:
        return torch.empty(0), torch.empty(0), torch.empty(0), torch.empty(0)

    hour = np.array([t.hour / 23.0 for t in timestamps], dtype=np.float32)
    quarter = np.array([(t.minute // 15) / 3.0 for t in timestamps], dtype=np.float32)
    weekday = np.array([t.weekday() / 6.0 for t in timestamps], dtype=np.float32)
    dayofyear = np.array([t.dayofyear / 365.0 for t in timestamps], dtype=np.float32)

    time_feats = np.stack([hour, quarter, weekday, dayofyear], axis=-1)

    X, y, TFx, TFy = [], [], [], []

    for i in range(T - lag):
        X.append(values[i : i + lag])
        TFx.append(time_feats[i : i + lag])
        y.append(values[i + lag])
        TFy.append(np.concatenate([time_feats[i + lag], [1.0 / lag]], axis=0))

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    TFx = np.asarray(TFx, dtype=np.float32)
    TFy = np.asarray(TFy, dtype=np.float32)

    return (
        torch.from_numpy(X)[:, :, None],
        torch.from_numpy(y)[:, None],
        torch.from_numpy(TFx),
        torch.from_numpy(TFy),
    )


# -------------------------------------------------
# Autoregressive forecast
# -------------------------------------------------
def forecast_autoregressive_electricity(
    model,
    init_vals,
    init_ts,
    steps,
    lag,
):

    model.eval()
    device = next(model.parameters()).device

    vals = list(init_vals)
    ts = list(pd.to_datetime(init_ts))
    preds = []

    for k in range(steps):
        w_vals = np.array(vals[-lag:], dtype=np.float32)

        w_ts = ts[-lag:]
        hour = np.array([t.hour / 23.0 for t in w_ts], dtype=np.float32)
        quarter = np.array([(t.minute // 15) / 3.0 for t in w_ts], dtype=np.float32)
        weekday = np.array([t.weekday() / 6.0 for t in w_ts], dtype=np.float32)
        dayofyear = np.array([t.dayofyear / 365.0 for t in w_ts], dtype=np.float32)

        TFx = np.stack([hour, quarter, weekday, dayofyear], axis=-1)

        x = torch.tensor(w_vals)[None, :, None].to(device)
        tfx = torch.tensor(TFx)[None, :, :].to(device)

        inp = torch.cat([x, tfx], dim=-1)

        next_ts = ts[-1] + (ts[-1] - ts[-2])

        tfy = torch.tensor(
            [
                [
                    next_ts.hour / 23.0,
                    (next_ts.minute // 15) / 3.0,
                    next_ts.weekday() / 6.0,
                    next_ts.dayofyear / 365.0,
                    (k + 1) / steps,
                ]
            ],
            dtype=torch.float32,
        ).to(device)

        with torch.no_grad():
            pred = model(inp, tfy=tfy).item()

        preds.append(pred)
        vals.append(pred)
        ts.append(next_ts)

    return preds


# -------------------------------------------------
# AR evaluation helpers
# -------------------------------------------------
def select_ar_validation_windows(val_sets, max_windows=100):
    ar = []
    for X, y, TFx, TFy, sid, ts in val_sets:
        ar.append(
            (
                X[0].squeeze(-1).cpu().numpy(),
                pd.to_datetime(ts[: X.shape[1]]),
                y[:50].cpu().numpy().flatten(),
                sid,
            )
        )
        if len(ar) >= max_windows:
            break
    return ar


def evaluate_ar_quick(model, ar_windows, horizons, lag):
    metrics = {h: [] for h in horizons}
    with torch.no_grad():
        for vals, ts, true, sid in ar_windows:
            preds = forecast_autoregressive_electricity(
                model, vals, ts, max(horizons), lag
            )
            for h in horizons:
                metrics[h].append(np.mean(np.abs(np.array(preds[:h]) - true[:h])))
    return {h: float(np.mean(v)) for h, v in metrics.items()}


# -------------------------------------------------
# Training
# -------------------------------------------------


def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    criterion,
    epochs,
    ar_windows,
    ar_eval_every,
    lag,
    num_series,
    save_path,
    device,
):
    best_val = float("inf")
    scaler = GradScaler()  # mixed precision

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for xb, yb, tfxb, tfyb in train_loader:
            xb = xb.to(device)
            tfxb = tfxb.to(device)
            tfyb = tfyb.to(device)
            yb = yb.to(device)

            inp = torch.cat([xb, tfxb], dim=-1)

            optimizer.zero_grad()
            with autocast(device_type=device.type):  # mixed precision
                out = model(inp, tfy=tfyb)
                loss = criterion(out, yb)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += loss.item() * xb.size(0)

        # -----------------------
        # Validation
        # -----------------------
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb, tfxb, tfyb in val_loader:
                xb = xb.to(device)
                tfxb = tfxb.to(device)
                tfyb = tfyb.to(device)
                yb = yb.to(device)
                inp = torch.cat([xb, tfxb], dim=-1)
                out = model(inp, tfy=tfyb)
                val_loss += criterion(out, yb).item() * xb.size(0)

        total_train = len(train_loader.dataset)
        total_val = len(val_loader.dataset)
        avg_train_loss = total_loss / total_train
        avg_val_loss = val_loss / total_val

        # -----------------------
        # Autoregressive eval
        # -----------------------
        ar_str = ""
        if epoch % ar_eval_every == 0 and ar_windows is not None:
            ar_metrics = evaluate_ar_quick(model, ar_windows, (1, 5, 10, 20), lag)
            ar_str = " | " + " ".join(f"h{h}:{v:.3f}" for h, v in ar_metrics.items())

        print(
            f"Epoch {epoch:02d} | Train {avg_train_loss:.6f} | Val {avg_val_loss:.6f}{ar_str}"
        )

        # Save best model
        if avg_val_loss < best_val:
            best_val = avg_val_loss
            torch.save(model.state_dict(), save_path + ".pth")


# -------------------------------------------------
# Evaluation
# -------------------------------------------------
def evaluate_autoregressive(
    model,
    test_dataset,
    horizons=(1, 5, 10, 20),
    max_windows_per_series=50,
    lag=96,
    device="cuda",
):
    """
    Evaluate autoregressive forecasts using a lazy dataset (no precomputed windows in memory).

    Args:
        model: trained PyTorch model
        test_dataset: ElectricityWindowDataset with split="test"
        horizons: prediction horizons to evaluate
        max_windows_per_series: max windows to use per series (to limit runtime)
        lag: context length
        device: torch device
    Returns:
        summary: dict[horizon] = {"MAE": float, "RMSE": float}
    """
    model.eval()
    num_series = test_dataset.num_series
    results = {h: {"mae": [], "rmse": []} for h in horizons}

    # We'll iterate series by series
    for series_id in range(num_series):
        # collect timestamps and values for this series
        series_values = test_dataset.series_data[:, series_id]
        series_timestamps = test_dataset.timestamps

        n_windows = min(
            max_windows_per_series, len(series_values) - lag - max(horizons)
        )
        if n_windows <= 0:
            continue

        for start in range(n_windows):
            init_vals = series_values[start : start + lag]
            init_ts = series_timestamps[start : start + lag]

            # AR prediction
            preds = forecast_autoregressive_electricity(
                model=model,
                init_vals=init_vals,
                init_ts=init_ts,
                steps=max(horizons),
                lag=lag,
            )

            true = series_values[start + lag : start + lag + max(horizons)]

            for h in horizons:
                p = np.array(preds[:h])
                t = true[:h]

                mae = np.mean(np.abs(p - t))
                rmse = np.sqrt(np.mean((p - t) ** 2))

                results[h]["mae"].append(mae)
                results[h]["rmse"].append(rmse)

    # Aggregate results
    summary = {}
    for h in horizons:
        summary[h] = {
            "MAE": float(np.mean(results[h]["mae"])),
            "RMSE": float(np.mean(results[h]["rmse"])),
        }

    return summary


# -------------------------------------------------
# Main
# -------------------------------------------------
def main():
    LAG = 96
    MODEL_DIM = 128
    NUM_HEADS = 8
    NUM_LAYERS = 8
    EPOCHS = 5
    BATCH_SIZE = 64
    DIM_FEEDFORWARD = 1536
    TFx_DIM = 4
    TFyDIM = 5

    df = pd.read_csv(
        el_path, sep=";", decimal=",", parse_dates=["timestamp"]
    ).sort_values("timestamp")

    train_dataset = ElectricityWindowDataset(df, lag=LAG, split="train")
    val_dataset = ElectricityWindowDataset(df, lag=LAG, split="val")
    test_dataset = ElectricityWindowDataset(df, lag=LAG, split="test")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=1,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
    )

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

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=1e-3,
        steps_per_epoch=len(train_loader),  # number of batches
        epochs=EPOCHS,
        pct_start=0.1,
    )

    ar_windows = []
    for i in range(min(100, len(val_dataset))):
        xb, yb, tfxb, tfyb = val_dataset[i]
        ar_windows.append(
            (
                xb.squeeze(-1).numpy(),
                val_dataset.timestamps[: val_dataset.lag],
                np.array([yb]),
                i % val_dataset.num_series,
            )
        )
        if len(ar_windows) >= 50:
            break

    if device.type in ("cpu", "mps"):
        SAVE_LOC = f"electricity_model_learnable_t2v_{EPOCHS}_epochs_{NUM_HEADS}_heads_{NUM_LAYERS}_layers_{MODEL_DIM}_dim_{DIM_FEEDFORWARD}_dim_feedforward_{TFx_DIM}_tfx_dim_{TFyDIM}_tfy_dim"
        AR_EVAL_EVERY = 1
    else:
        SAVE_LOC = f"/home/sc.uni-leipzig.de/uj74reda/Transformer/models/electricity_model_learnable_t2v_{EPOCHS}_epochs_{NUM_HEADS}_heads_{NUM_LAYERS}_layers_{MODEL_DIM}_dim_{DIM_FEEDFORWARD}_dim_feedforward_{TFx_DIM}_tfx_dim_{TFyDIM}_tfy_dim"
        AR_EVAL_EVERY = 1

    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        epochs=EPOCHS,
        ar_windows=ar_windows,
        ar_eval_every=AR_EVAL_EVERY,
        lag=LAG,
        num_series=len(train_dataset.series_cols),
        save_path=SAVE_LOC,
        device=device,
    )

    print("\n===============================")
    print(" AUTOREGRESSIVE TEST EVALUATION ")
    print("===============================\n")

    ar_results = evaluate_autoregressive(
        model=model,
        test_dataset=test_dataset,
        horizons=(1, 5, 10, 20),
        max_windows_per_series=50,  # adjust to speed up eval
        lag=LAG,
        device=device,
    )

    for h, metrics in ar_results.items():
        print(f"Horizon {h:>2}: MAE={metrics['MAE']:.4f}, RMSE={metrics['RMSE']:.4f}")


if __name__ == "__main__":
    main()
