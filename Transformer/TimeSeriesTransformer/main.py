import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import glob
from transformer import (
    TransformerTimeSeriesWithLearnableTime2Vec,
)
import pandas as pd

# -------------------------------------------------
# Setup for Apple Silicon
# -------------------------------------------------
torch.set_float32_matmul_precision("medium")
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# -------------------------
# Load and concatenate datasets
# -------------------------
def load_and_concat_datasets(data_dir, lag=12):

    X_train_list, y_train_list = [], []
    TFx_train_list, TFy_train_list = [], []

    X_val_list, y_val_list = [], []
    TFx_val_list, TFy_val_list = [], []

    X_test_list, y_test_list = [], []
    TFx_test_list, TFy_test_list = [], []

    val_sets = []
    test_sets = []

    for file in glob.glob(f"{data_dir}/*.csv"):

        df = pd.read_csv(file, parse_dates=["timestamp"]).sort_values("timestamp")
        n = len(df)

        train_end = int(0.70 * n)
        val_end = int(0.85 * n)

        # ------------- TRAIN -----------------
        Xtr, ytr, TFxtr, TFytr = build_windows(
            df["value"].values[:train_end], df["timestamp"].values[:train_end], lag
        )
        X_train_list.append(Xtr)
        y_train_list.append(ytr)
        TFx_train_list.append(TFxtr)
        TFy_train_list.append(TFytr)

        # ------------- VALIDATION -------------
        Xv, yv, TFxv, TFyv = build_windows(
            df["value"].values[train_end:val_end],
            df["timestamp"].values[train_end:val_end],
            lag,
        )
        X_val_list.append(Xv)
        y_val_list.append(yv)
        TFx_val_list.append(TFxv)
        TFy_val_list.append(TFyv)

        timestamps_val = df["timestamp"].values[train_end:val_end]
        val_sets.append((Xv, yv, TFxv, TFyv, file, timestamps_val))

        # ------------- TEST -------------------
        Xte, yte, TFxte, TFyte = build_windows(
            df["value"].values[val_end:], df["timestamp"].values[val_end:], lag
        )
        X_test_list.append(Xte)
        y_test_list.append(yte)
        TFx_test_list.append(TFxte)
        TFy_test_list.append(TFyte)

        timestamps_test = df["timestamp"].values[val_end:]
        test_sets.append((Xte, yte, TFxte, TFyte, file, timestamps_test))

    # ------------- CONCAT ALL --------------
    X_train = torch.cat(X_train_list)
    y_train = torch.cat(y_train_list)
    TFx_train = torch.cat(TFx_train_list)
    TFy_train = torch.cat(TFy_train_list)

    X_val = torch.cat(X_val_list)
    y_val = torch.cat(y_val_list)
    TFx_val = torch.cat(TFx_val_list)
    TFy_val = torch.cat(TFy_val_list)

    X_test = torch.cat(X_test_list)
    y_test = torch.cat(y_test_list)
    TFx_test = torch.cat(TFx_test_list)
    TFy_test = torch.cat(TFy_test_list)

    return (
        X_train,
        y_train,
        TFx_train,
        TFy_train,
        X_val,
        y_val,
        TFx_val,
        TFy_val,
        X_test,
        y_test,
        TFx_test,
        TFy_test,
        val_sets,
        test_sets,
    )


# -------------------------
# Autoregressive forecasting function
# -------------------------
def forecast_autoregressive(model, init_window_vals, init_window_ts, steps=20, lag=12):
    """
    Autoregressive forecasting using:
    - lag window values (init_window_vals)
    - timestamps for each lag step (init_window_ts)
    """

    model.eval()
    device = next(model.parameters()).device

    current_vals = list(init_window_vals)
    current_ts = list(pd.to_datetime(init_window_ts))

    preds = []

    for _ in range(steps):

        # ---- Prepare lag window ----
        vals = np.array(current_vals[-lag:], dtype=np.float32)
        hours = np.array([t.hour / 23.0 for t in current_ts[-lag:]], dtype=np.float32)
        wdays = np.array(
            [t.weekday() / 6.0 for t in current_ts[-lag:]], dtype=np.float32
        )
        tfx = np.stack([hours, wdays], axis=-1)  # (lag, 2)

        # Combine values + TFx
        x = torch.tensor(vals)[None, :, None]  # (1, lag, 1)
        tfx_tensor = torch.tensor(tfx)[None, :, :]  # (1, lag, 2)
        inp = torch.cat([x, tfx_tensor], dim=-1).to(device)  # (1, lag, 3)

        # ---- Prepare TFy for next timestamp ----
        delta = current_ts[-1] - current_ts[-2]
        next_ts = current_ts[-1] + delta

        tfy = torch.tensor(
            [[next_ts.hour / 23.0, next_ts.weekday() / 6.0]], dtype=torch.float32
        ).to(device)

        # ---- Model forward ----
        with torch.no_grad():
            pred = model(inp, tfy=tfy).item()

        preds.append(pred)

        # ---- Update window ----
        current_vals.append(pred)
        current_ts.append(next_ts)

    return preds


def build_windows(values, timestamps, lag):
    """
    Returns:
        X:    (N, lag, 1)
        y:    (N, 1)
        TFx:  (N, lag, 2)   time features for each input timestep
        TFy:  (N, 2)        time features for the prediction timestamp
    """
    values = np.array(values, dtype=np.float32)
    timestamps = pd.to_datetime(timestamps)

    hours = np.array([t.hour / 23.0 for t in timestamps], dtype=np.float32)
    wdays = np.array([t.weekday() / 6.0 for t in timestamps], dtype=np.float32)
    time_feats = np.stack([hours, wdays], axis=-1)  # (T, 2)

    X_list, y_list, TFx_list, TFy_list = [], [], [], []

    for i in range(len(values) - lag):
        # Inputs
        X_list.append(values[i : i + lag])
        TFx_list.append(time_feats[i : i + lag])

        # Target
        y_list.append(values[i + lag])
        TFy_list.append(time_feats[i + lag])

    X = np.array(X_list, dtype=np.float32)[:, :, None]  # (N, lag, 1)
    y = np.array(y_list, dtype=np.float32)[:, None]  # (N, 1)
    TFx = np.array(TFx_list, dtype=np.float32)  # (N, lag, 2)
    TFy = np.array(TFy_list, dtype=np.float32)  # (N, 2)

    return (
        torch.from_numpy(X),
        torch.from_numpy(y),
        torch.from_numpy(TFx),
        torch.from_numpy(TFy),
    )


def batches(X, y, TFx, TFy, bs):
    idx = torch.randperm(len(X))
    for i in range(0, len(X), bs):
        j = idx[i : i + bs]
        yield X[j], y[j], TFx[j], TFy[j]


def train_model(
    model,
    X_train,
    y_train,
    TFx_train,
    TFy_train,
    X_val,
    y_val,
    TFx_val,
    TFy_val,
    optimizer,
    scheduler,
    criterion,
    epochs=10,
    batch_size=64,
    ar_windows=None,
    ar_eval_every=1,
    save_path="best_model",
):
    n_train = len(X_train)
    n_val = len(X_val)
    best_val = float("inf")

    steps_per_epoch = int(np.ceil(len(X_train) / BATCH_SIZE))

    for epoch in range(1, epochs + 1):

        # ---------- TRAIN ----------
        model.train()
        perm = torch.randperm(n_train)
        total_loss = 0

        for i in range(0, n_train, batch_size):
            idx = perm[i : i + batch_size]

            xb = X_train[idx].to(device)
            tfxb = TFx_train[idx].to(device)
            tfyb = TFy_train[idx].to(device)
            yb = y_train[idx].to(device)

            inp = torch.cat([xb, tfxb], dim=-1)

            optimizer.zero_grad()
            out = model(inp, tfy=tfyb)
            loss = criterion(out, yb)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        train_loss = total_loss / steps_per_epoch

        # ---------- VALID ----------
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for i in range(0, n_val, batch_size):
                xb = X_val[i : i + batch_size].to(device)
                tfxb = TFx_val[i : i + batch_size].to(device)
                tfyb = TFy_val[i : i + batch_size].to(device)
                yb = y_val[i : i + batch_size].to(device)

                inp = torch.cat([xb, tfxb], dim=-1)
                out = model(inp, tfy=tfyb)
                val_loss += criterion(out, yb).item()

        val_loss /= max(1, n_val // batch_size)

        # ---------- AR VALIDATION ----------
        ar_str = ""
        if ar_windows is not None and epoch % ar_eval_every == 0:
            ar_metrics = evaluate_ar_quick(
                model, ar_windows, horizons=(1, 5, 10, 20), lag=lag
            )

            ar_str = " | AR-MAE " + " ".join(
                [f"h{h}:{v:.3f}" for h, v in ar_metrics.items()]
            )

        print(
            f"Epoch {epoch:02d} "
            f"| Train {train_loss:.4f} "
            f"| Val {val_loss:.4f}"
            f"{ar_str}"
        )

        # ---------- CHECKPOINT ----------
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), save_path + ".pth")


def evaluate_test(model, X_test, y_test, TFx_test, TFy_test, batch_size=64):
    model.eval()
    n = len(X_test)
    total_loss = 0
    criterion = nn.MSELoss()

    with torch.no_grad():
        for i in range(0, n, batch_size):

            xb = X_test[i : i + batch_size].to(device)
            tfxb = TFx_test[i : i + batch_size].to(device)
            tfyb = TFy_test[i : i + batch_size].to(device)
            yb = y_test[i : i + batch_size].to(device)

            inp = torch.cat([xb, tfxb], dim=-1)
            out = model(inp, tfy=tfyb)

            total_loss += criterion(out, yb).item()

    return total_loss / max(1, n // batch_size)


# -------------------------
# Evaluation
# -------------------------


def mase(y_true, y_pred, y_train, seasonality):
    naive_err = np.mean(np.abs(y_train[seasonality:] - y_train[:-seasonality]))
    return np.mean(np.abs(y_true - y_pred)) / (naive_err + 1e-8)


def evaluate_autoregressive(
    model,
    test_sets,
    horizons=(1, 5, 10, 20),
    max_windows=50,
    lag=288,
):
    """
    Evaluates autoregressive forecasts over multiple rolling windows.
    Returns horizon-wise MAE and RMSE.
    """
    model.eval()
    results = {h: {"mae": [], "rmse": []} for h in horizons}

    for X, y, TFx, TFy, fname, timestamps in test_sets:

        n_windows = min(max_windows, len(X) - max(horizons))
        for start in range(n_windows):

            init_vals = X[start].squeeze(-1).cpu().numpy()
            init_ts = pd.to_datetime(timestamps[start : start + lag])

            preds = forecast_autoregressive(
                model, init_vals, init_ts, steps=max(horizons), lag=lag
            )

            true = y[start : start + max(horizons)].cpu().numpy().flatten()

            for h in horizons:
                p = np.array(preds[:h])
                t = true[:h]

                mae = np.mean(np.abs(p - t))
                rmse = np.sqrt(np.mean((p - t) ** 2))

                results[h]["mae"].append(mae)
                results[h]["rmse"].append(rmse)

    # Aggregate
    summary = {}
    for h in horizons:
        summary[h] = {
            "MAE": float(np.mean(results[h]["mae"])),
            "RMSE": float(np.mean(results[h]["rmse"])),
        }

    return summary


def select_ar_validation_windows(val_sets, max_windows=5):
    ar_windows = []
    for X, y, TFx, TFy, fname, timestamps in val_sets:
        if len(X) == 0:
            continue

        ar_windows.append(
            (
                X[0].squeeze(-1).cpu().numpy(),
                pd.to_datetime(timestamps[: X.shape[1]]),
                y[:50].cpu().numpy().flatten(),
                fname,
            )
        )

        if len(ar_windows) >= max_windows:
            break

    return ar_windows


def evaluate_ar_quick(model, ar_windows, horizons=(1, 5, 10, 20), lag=288):
    model.eval()
    metrics = {h: [] for h in horizons}

    with torch.no_grad():
        for init_vals, init_ts, true_vals, fname in ar_windows:

            preds = forecast_autoregressive(
                model,
                init_vals,
                init_ts,
                steps=max(horizons),
                lag=lag,
            )

            for h in horizons:
                p = np.array(preds[:h])
                t = true_vals[:h]

                mae = np.mean(np.abs(p - t))
                metrics[h].append(mae)

    return {h: float(np.mean(v)) for h, v in metrics.items()}


# -------------------------
# Load datasets
# -------------------------

lag = 90
DATA_DIR = "data/processed/nab/taxi"

(
    X_train,
    y_train,
    TFx_train,
    TFy_train,
    X_val,
    y_val,
    TFx_val,
    TFy_val,
    X_test,
    y_test,
    TFx_test,
    TFy_test,
    val_sets,
    test_sets,
) = load_and_concat_datasets(DATA_DIR, lag=lag)

# current best config
MODEL_DIM = 128
NUM_HEADS = 16
NUM_LAYERS = 8
EPOCHS = 2
BATCH_SIZE = 64

model = TransformerTimeSeriesWithLearnableTime2Vec(
    t2v_dim=16,
    model_dim=MODEL_DIM,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS,
    dropout=0.1,
).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=1e-3,
    steps_per_epoch=int(np.ceil(len(X_train) / BATCH_SIZE)),
    epochs=EPOCHS,
    pct_start=0.1,
    anneal_strategy="cos",
)

# -------------------------
# Train
# -------------------------
ar_val_windows = select_ar_validation_windows(val_sets, max_windows=3)

train_model(
    model,
    X_train,
    y_train,
    TFx_train,
    TFy_train,
    X_val,
    y_val,
    TFx_val,
    TFy_val,
    optimizer,
    scheduler,
    criterion,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    ar_windows=ar_val_windows,
    ar_eval_every=1,  # or 2–5 for speed
    save_path=f"learnable_t2v_{EPOCHS}_epochs_{NUM_HEADS}_heads_{NUM_LAYERS}_layers_{MODEL_DIM}_dim",
)

print("\n===============================")
print(" AUTOREGRESSIVE TEST EVALUATION ")
print("===============================\n")

ar_results = evaluate_autoregressive(
    model,
    test_sets,
    horizons=(1, 5, 10, 20),
    lag=lag,
)

for h, metrics in ar_results.items():
    print(
        f"Horizon {h:>2}: " f"MAE={metrics['MAE']:.4f}, " f"RMSE={metrics['RMSE']:.4f}"
    )
