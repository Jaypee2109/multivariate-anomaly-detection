import glob
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from dotenv import load_dotenv
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from transformer import (
    TransformerTimeSeriesWithLearnableTime2Vec,
)

# -------------------------------------------------
# Device Setup
# -------------------------------------------------
torch.set_float32_matmul_precision("medium")
DEVICE = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available() else "cpu"
)


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

        train_end = int(0.8 * n)
        val_end = int(0.9 * n)

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
# Autoregressive forecasting
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
        T0 = len(current_ts)
        total_T = len(current_ts) + steps

        abs_idx = np.arange(T0 - lag, T0, dtype=np.float32) / total_T
        t_idx = T0 / total_T

        # ---- Prepare lag window ----
        vals = np.array(current_vals[-lag:], dtype=np.float32)
        hours = np.array([t.hour / 23.0 for t in current_ts[-lag:]], dtype=np.float32)
        minutes = np.array(
            [t.minute / 59.0 for t in current_ts[-lag:]], dtype=np.float32
        )
        wdays = np.array(
            [t.weekday() / 6.0 for t in current_ts[-lag:]], dtype=np.float32
        )

        tfx = np.stack([hours, minutes, wdays, abs_idx], axis=-1)  # (lag, 4)

        x = torch.tensor(vals)[None, :, None]
        tfx_tensor = torch.tensor(tfx)[None, :, :]
        inp = torch.cat([x, tfx_tensor], dim=-1).to(device)  # (1, lag, 4)

        # ---- Prepare TFy for next timestamp ----
        delta = current_ts[-1] - current_ts[-2]
        next_ts = current_ts[-1] + delta

        horizon = (len(preds) + 1) / steps  # ∈ (0,1]

        tfy = torch.tensor(
            [
                [
                    next_ts.hour / 23.0,
                    next_ts.minute / 59.0,
                    next_ts.weekday() / 6.0,
                    t_idx,
                    horizon,
                ]
            ],
            dtype=torch.float32,
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
        TFx:  (N, lag, 3)   time features for each input timestep
        TFy:  (N, 2)        time features for the prediction timestamp
    """
    values = np.array(values, dtype=np.float32)
    timestamps = pd.to_datetime(timestamps)

    T = len(values)
    time_idx = np.arange(T, dtype=np.float32)
    time_idx /= time_idx.max() + 1e-6

    minutes = np.array([t.minute / 59.0 for t in timestamps], dtype=np.float32)
    hours = np.array([t.hour / 23.0 for t in timestamps], dtype=np.float32)
    wdays = np.array([t.weekday() / 6.0 for t in timestamps], dtype=np.float32)

    time_feats = np.stack([hours, minutes, wdays, time_idx], axis=-1)  # (T, 4)

    X_list, y_list, TFx_list, TFy_list = [], [], [], []

    for i in range(len(values) - lag):
        horizon = 1.0 / lag

        # Inputs
        X_list.append(values[i : i + lag])
        TFx_list.append(time_feats[i : i + lag])

        # Target
        y_list.append(values[i + lag])
        TFy_list.append(np.concatenate([time_feats[i + lag], [horizon]], axis=0))

    X = np.array(X_list, dtype=np.float32)[:, :, None]  # (N, lag, 1)
    y = np.array(y_list, dtype=np.float32)[:, None]  # (N, 1)
    TFx = np.array(TFx_list, dtype=np.float32)  # (N, lag, 4)
    TFy = np.array(TFy_list, dtype=np.float32)  # (N, 5)

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

    steps_per_epoch = int(np.ceil(len(X_train) / batch_size))

    for epoch in range(1, epochs + 1):
        # Decay teacher forcing ratio from 1.0 → 0.5 over epochs
        teacher_forcing_ratio = max(0.5, 1.0 - epoch / epochs)

        # ---------- TRAIN ----------
        model.train()
        perm = torch.randperm(n_train)
        total_loss = 0

        for i in range(0, n_train, batch_size):
            idx = perm[i : i + batch_size]

            xb = X_train[idx].to(DEVICE)  # (B, lag, 1)
            tfxb = TFx_train[idx].to(DEVICE)  # (B, lag, time_feats)
            tfyb = TFy_train[idx].to(DEVICE)  # (B, tfy_feats)
            yb = y_train[idx].to(DEVICE)  # (B, 1)

            # --------- Scheduled Sampling ---------
            xb_modified = xb.clone()
            B, lag_len, _ = xb.shape

            for b in range(B):
                for t in range(1, lag_len):
                    if torch.rand(1).item() > teacher_forcing_ratio:
                        # Predict previous value using model
                        with torch.no_grad():
                            prev_inp = torch.cat(
                                [xb_modified[b : b + 1, :t, :], tfxb[b : b + 1, :t, :]],
                                dim=-1,
                            )
                            pred_prev = model(prev_inp, tfy=tfyb[b : b + 1]).squeeze(0)
                        xb_modified[b, t, 0] = pred_prev

            inp = torch.cat([xb_modified, tfxb], dim=-1)

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
                xb = X_val[i : i + batch_size].to(DEVICE)
                tfxb = TFx_val[i : i + batch_size].to(DEVICE)
                tfyb = TFy_val[i : i + batch_size].to(DEVICE)
                yb = y_val[i : i + batch_size].to(DEVICE)

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
            f"Epoch {epoch:02d} | Train {train_loss:.4f} | Val {val_loss:.4f}{ar_str}"
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
            xb = X_test[i : i + batch_size].to(DEVICE)
            tfxb = TFx_test[i : i + batch_size].to(DEVICE)
            tfyb = TFy_test[i : i + batch_size].to(DEVICE)
            yb = y_test[i : i + batch_size].to(DEVICE)

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


def compute_error_threshold(
    model,
    X_train,
    y_train,
    TFx_train,
    TFy_train,
    quantile=0.95,
    batch_size=64,
):
    model.eval()
    errors = []

    with torch.no_grad():
        for i in range(0, len(X_train), batch_size):
            xb = X_train[i : i + batch_size].to(DEVICE)
            tfxb = TFx_train[i : i + batch_size].to(DEVICE)
            tfyb = TFy_train[i : i + batch_size].to(DEVICE)
            yb = y_train[i : i + batch_size].to(DEVICE)

            inp = torch.cat([xb, tfxb], dim=-1)
            preds = model(inp, tfy=tfyb)

            err = torch.abs(preds - yb)
            errors.append(err.cpu().numpy())

    errors = np.concatenate(errors).flatten()
    tau = np.quantile(errors, quantile)

    return float(tau)


def error_based_classification_metrics(
    y_true,
    y_pred,
    tau,
):
    """
    y_true, y_pred: (N,)
    tau: error threshold
    """

    errors = np.abs(y_pred - y_true)

    # Ground truth & predictions
    y_true_cls = (errors > tau).astype(int)
    y_pred_cls = (errors > tau).astype(int)

    # Continuous score for ROC
    y_score = errors

    metrics = {
        "precision": precision_score(y_true_cls, y_pred_cls, zero_division=0),
        "recall": recall_score(y_true_cls, y_pred_cls, zero_division=0),
        "f1": f1_score(y_true_cls, y_pred_cls, zero_division=0),
    }

    if len(np.unique(y_true_cls)) > 1:
        metrics["auc_roc"] = roc_auc_score(y_true_cls, y_score)
    else:
        metrics["auc_roc"] = np.nan

    return metrics


# -------------------------
# Load datasets
# -------------------------

lag = 90
if DEVICE.type in ("cpu", "mps"):
    DATA_DIR = "data/processed/nab/taxi"
else:
    DATA_DIR = "/home/sc.uni-leipzig.de/uj74reda/Transformer/taxi/taxi"
print("DATA_DIR:", DATA_DIR)

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
EPOCHS = 10
BATCH_SIZE = 64

model = TransformerTimeSeriesWithLearnableTime2Vec(
    t2v_dim=16,
    model_dim=MODEL_DIM,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS,
    dropout=0.1,
).to(DEVICE)

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


def evaluate_test_error_classification(
    model,
    X_test,
    y_test,
    TFx_test,
    TFy_test,
    tau,
    batch_size=64,
):
    model.eval()

    y_true_all, y_pred_all = [], []

    with torch.no_grad():
        for i in range(0, len(X_test), batch_size):
            xb = X_test[i : i + batch_size].to(DEVICE)
            tfxb = TFx_test[i : i + batch_size].to(DEVICE)
            tfyb = TFy_test[i : i + batch_size].to(DEVICE)

            inp = torch.cat([xb, tfxb], dim=-1)
            preds = model(inp, tfy=tfyb).squeeze(-1)

            y_pred_all.append(preds.cpu().numpy())
            y_true_all.append(y_test[i : i + batch_size].squeeze(-1).cpu().numpy())

    y_true = np.concatenate(y_true_all)
    y_pred = np.concatenate(y_pred_all)

    return error_based_classification_metrics(y_true, y_pred, tau)


def evaluate_autoregressive_error_classification(
    model,
    test_sets,
    tau,
    horizons=(1, 5, 10, 20),
    max_windows=50,
    lag=288,
):
    model.eval()

    results = {
        h: {"precision": [], "recall": [], "f1": [], "auc_roc": []} for h in horizons
    }

    for X, y, TFx, TFy, fname, timestamps in test_sets:
        n_windows = min(max_windows, len(X) - max(horizons))
        for start in range(n_windows):
            init_vals = X[start].squeeze(-1).cpu().numpy()
            init_ts = pd.to_datetime(timestamps[start : start + lag])

            preds = forecast_autoregressive(
                model,
                init_vals,
                init_ts,
                steps=max(horizons),
                lag=lag,
            )

            true_vals = y[start : start + max(horizons)].cpu().numpy().flatten()

            for h in horizons:
                y_p = np.array(preds[:h])
                y_t = true_vals[:h]

                m = error_based_classification_metrics(y_t, y_p, tau)

                for k in results[h]:
                    results[h][k].append(m[k])

    summary = {
        h: {k: float(np.nanmean(v)) for k, v in results[h].items()} for h in horizons
    }

    return summary


# -------------------------
# Train
# -------------------------
ar_val_windows = select_ar_validation_windows(val_sets, max_windows=100)

if DEVICE.type in ("cpu", "mps"):
    SAVE_LOC = f"learnable_t2v_{EPOCHS}_epochs_{NUM_HEADS}_heads_{NUM_LAYERS}_layers_{MODEL_DIM}_dim"
    AR_EVAL_EVERY = 1
else:
    SAVE_LOC = f"/home/sc.uni-leipzig.de/uj74reda/Transformer/models/learnable_t2v_{EPOCHS}_epochs_{NUM_HEADS}_heads_{NUM_LAYERS}_layers_{MODEL_DIM}_dim"
    AR_EVAL_EVERY = 5

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
    ar_eval_every=AR_EVAL_EVERY,  # or 2–5 for speed
    save_path=SAVE_LOC,
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
    print(f"Horizon {h:>2}: MAE={metrics['MAE']:.4f}, RMSE={metrics['RMSE']:.4f}")


TAU = compute_error_threshold(
    model,
    X_train,
    y_train,
    TFx_train,
    TFy_train,
    quantile=0.95,
)

print(f"Error threshold τ = {TAU:.4f}")


ar_cls = evaluate_autoregressive_error_classification(
    model,
    test_sets,
    tau=TAU,
    horizons=(1, 5, 10, 20),
    lag=lag,
)

print("\nAR error-based classification:")
for h, m in ar_cls.items():
    print(
        f"H{h:>2} | "
        f"P={m['precision']:.3f} "
        f"R={m['recall']:.3f} "
        f"F1={m['f1']:.3f} "
        f"AUC={m['auc_roc']:.3f}"
    )
