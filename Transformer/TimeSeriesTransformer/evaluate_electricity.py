import torch
import numpy as np
import pandas as pd

from torch.utils.data import DataLoader

from transformer import TransformerTimeSeriesWithLearnableTime2Vec

# -------------------------------------------------
# Utilities (copied from training script)
# -------------------------------------------------


def normalize_series(values):
    mean = values.mean(axis=0, keepdims=True)
    std = values.std(axis=0, keepdims=True)
    return (values - mean) / (std + 1e-8), mean.squeeze(), std.squeeze()


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


class ElectricityWindowDataset(torch.utils.data.Dataset):
    def __init__(self, values, timestamps, lag, split, ratios=(0.75, 0.05)):
        self.values = values
        self.timestamps = pd.to_datetime(timestamps)
        self.lag = lag

        n = len(timestamps)
        tr, va = int(ratios[0] * n), int((ratios[0] + ratios[1]) * n)

        if split == "test":
            self.start, self.end = va, n
        else:
            raise ValueError("Evaluation script only supports test split")

        self.num_series = values.shape[1]
        self.windows_per_series = (self.end - self.start) - lag

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
            torch.tensor([y], dtype=torch.float32),
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
        ).astype(torch.float32)

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


# -------------------------------------------------
# Evaluation
# -------------------------------------------------


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
        # compute time features exactly like training
        hours = np.array([t.hour / 23 for t in ts[-lag:]], dtype=np.float32)
        minutes = np.array([(t.minute // 15) / 3 for t in ts[-lag:]], dtype=np.float32)
        wdays = np.array([t.weekday() / 6 for t in ts[-lag:]], dtype=np.float32)
        doy = np.array([t.dayofyear / 365 for t in ts[-lag:]], dtype=np.float32)
        sid_feat = np.full_like(
            hours, sid / len(vals), dtype=np.float32
        )  # sid / num_series

        tfx = np.stack([hours, minutes, wdays, doy, sid_feat], axis=-1)  # shape (lag,5)
        x = torch.tensor(vals[-lag:], device=device)[None, :, None]  # shape (1, lag, 1)
        tfx = torch.tensor(tfx, device=device)[None]  # shape (1, lag, 5)

        tfy = torch.tensor(
            [
                [
                    ts[-1].hour / 23,
                    (ts[-1].minute // 15) / 3,
                    ts[-1].weekday() / 6,
                    ts[-1].dayofyear / 365,
                    sid / len(vals),
                    (step + 1) / steps,
                ]
            ],
            device=device,
            dtype=torch.float32,
        )

        with torch.no_grad():
            pred = model(torch.cat([x, tfx], dim=-1), tfy=tfy).item()

        pred_denorm = pred * series_std[sid] + series_mean[sid]
        preds.append(pred_denorm)
        vals.append(pred)
        ts.append(ts[-1] + (ts[-1] - ts[-2]))

    return preds


def evaluate_ar_sample(model, values, timestamps, lag, series_mean, series_std, device):
    horizons = [1, 2, 5, 10, 15, 20]
    results = {}

    for h in horizons:
        windows = build_ar_test_windows(
            values, timestamps, lag, horizons=(h,), max_windows=100
        )
        mae_list, rmse_list = [], []

        for init_vals, init_ts, true_vals, sid in windows:
            preds = forecast_autoregressive(
                model,
                init_vals,
                init_ts,
                steps=h,
                lag=lag,
                device=device,
                series_mean=series_mean,
                series_std=series_std,
                sid=sid,
            )

            true_vals = true_vals[:h] * series_std[sid] + series_mean[sid]
            preds = np.array(preds)

            err = preds - true_vals
            mae_list.append(np.mean(np.abs(err)))
            rmse_list.append(np.sqrt(np.mean(err**2)))

        results[h] = {
            "MAE": float(np.mean(mae_list)),
            "RMSE": float(np.mean(rmse_list)),
        }

    return results


# -------------------------------------------------
# Main
# -------------------------------------------------


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")

    # --------------------
    # Model config (MUST match training)
    # --------------------
    MODEL_DIM = 384
    NUM_HEADS = 12
    NUM_LAYERS = 6
    DIM_FEEDFORWARD = 1536
    TFx_DIM = 5
    TFy_DIM = 6
    LAG = 96

    CHECKPOINT_PATH = "models/electricity_model_learnable_t2v_4_epochs_12_heads_6_layers_384_dim_1536_dim_feedforward_5_tfx_dim_6_tfy_dim_ddp_fixed"

    # --------------------
    # Load data
    # --------------------
    timestamps, values = load_raw_electricity("data/raw/LD2011_2014.txt")

    values, series_mean, series_std = normalize_series(values)

    test_ds = ElectricityWindowDataset(
        values=values,
        timestamps=timestamps,
        lag=LAG,
        split="test",
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=128,
        shuffle=False,
        num_workers=2,
    )

    # --------------------
    # Load model
    # --------------------
    model = TransformerTimeSeriesWithLearnableTime2Vec(
        t2v_dim=16,
        model_dim=MODEL_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        dropout=0.1,
        dim_feedforward=DIM_FEEDFORWARD,
        tfx_dim=TFx_DIM,
        tfy_dim=TFy_DIM,
    ).to(device)

    state = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(state)

    print("✓ Loaded model weights")

    # --------------------
    # Small AR evaluation
    # --------------------
    results = evaluate_ar_sample(
        model, values, timestamps, LAG, series_mean, series_std, device
    )

    print("\n==== AUTOREGRESSIVE SAMPLE METRICS ====")
    for h, metrics in results.items():
        print(f"H{h:02d} | MAE: {metrics['MAE']:.4f} | RMSE: {metrics['RMSE']:.4f}")


if __name__ == "__main__":
    main()
