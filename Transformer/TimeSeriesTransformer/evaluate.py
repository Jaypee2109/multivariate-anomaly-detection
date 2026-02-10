import glob
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from transformer import TransformerTimeSeriesWithLearnableTime2Vec

# -------------------------------------------------
# Device setup (CPU / MPS / CUDA)
# -------------------------------------------------
torch.set_float32_matmul_precision("medium")

DEVICE = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", DEVICE)

# -------------------------------------------------
# Constants (change freely if needed)
# -------------------------------------------------
LAG = 90

MODEL_DIM = 128
NUM_HEADS = 16
NUM_LAYERS = 8
T2V_DIM = 16
DROPOUT = 0.1

BATCH_SIZE = 64

# Placeholder checkpoint path
MODEL_CHECKPOINT_PATH = "models/learnable_t2v_10_epochs_16_heads_8_layers_128_dim.pth"


# -------------------------------------------------
# Data loading utilities (unchanged)
# -------------------------------------------------
def build_windows(values, timestamps, lag):
    values = np.array(values, dtype=np.float32)
    timestamps = pd.to_datetime(timestamps)

    T = len(values)
    time_idx = np.arange(T, dtype=np.float32)
    time_idx /= time_idx.max() + 1e-6

    minutes = np.array([t.minute / 59.0 for t in timestamps], dtype=np.float32)
    hours = np.array([t.hour / 23.0 for t in timestamps], dtype=np.float32)
    wdays = np.array([t.weekday() / 6.0 for t in timestamps], dtype=np.float32)

    time_feats = np.stack([hours, minutes, wdays, time_idx], axis=-1)

    X_list, y_list, TFx_list, TFy_list = [], [], [], []

    for i in range(len(values) - lag):
        horizon = 1.0 / lag

        X_list.append(values[i : i + lag])
        TFx_list.append(time_feats[i : i + lag])

        y_list.append(values[i + lag])
        TFy_list.append(np.concatenate([time_feats[i + lag], [horizon]], axis=0))

    X = np.array(X_list, dtype=np.float32)[:, :, None]
    y = np.array(y_list, dtype=np.float32)[:, None]
    TFx = np.array(TFx_list, dtype=np.float32)
    TFy = np.array(TFy_list, dtype=np.float32)

    return (
        torch.from_numpy(X),
        torch.from_numpy(y),
        torch.from_numpy(TFx),
        torch.from_numpy(TFy),
    )


def load_and_concat_datasets(data_dir, lag):
    X_train_list, y_train_list, TFx_train_list, TFy_train_list = [], [], [], []
    X_val_list, y_val_list, TFx_val_list, TFy_val_list = [], [], [], []
    X_test_list, y_test_list, TFx_test_list, TFy_test_list = [], [], [], []

    val_sets, test_sets = [], []

    for file in glob.glob(f"{data_dir}/*.csv"):
        df = pd.read_csv(file, parse_dates=["timestamp"]).sort_values("timestamp")
        n = len(df)

        train_end = int(0.8 * n)
        val_end = int(0.9 * n)

        # Train
        Xtr, ytr, TFxtr, TFytr = build_windows(
            df["value"].values[:train_end],
            df["timestamp"].values[:train_end],
            lag,
        )

        # Val
        Xv, yv, TFxv, TFyv = build_windows(
            df["value"].values[train_end:val_end],
            df["timestamp"].values[train_end:val_end],
            lag,
        )

        # Test
        Xte, yte, TFxte, TFyte = build_windows(
            df["value"].values[val_end:],
            df["timestamp"].values[val_end:],
            lag,
        )

        X_train_list.append(Xtr)
        y_train_list.append(ytr)
        TFx_train_list.append(TFxtr)
        TFy_train_list.append(TFytr)

        X_val_list.append(Xv)
        y_val_list.append(yv)
        TFx_val_list.append(TFxv)
        TFy_val_list.append(TFyv)

        X_test_list.append(Xte)
        y_test_list.append(yte)
        TFx_test_list.append(TFxte)
        TFy_test_list.append(TFyte)

        val_sets.append(
            (Xv, yv, TFxv, TFyv, file, df["timestamp"].values[train_end:val_end])
        )
        test_sets.append(
            (Xte, yte, TFxte, TFyte, file, df["timestamp"].values[val_end:])
        )

    return (
        torch.cat(X_train_list),
        torch.cat(y_train_list),
        torch.cat(TFx_train_list),
        torch.cat(TFy_train_list),
        torch.cat(X_val_list),
        torch.cat(y_val_list),
        torch.cat(TFx_val_list),
        torch.cat(TFy_val_list),
        torch.cat(X_test_list),
        torch.cat(y_test_list),
        torch.cat(TFx_test_list),
        torch.cat(TFy_test_list),
        val_sets,
        test_sets,
    )


# -------------------------------------------------
# Evaluation helpers (unchanged logic)
# -------------------------------------------------
def evaluate_test_mse(model, X_test, y_test, TFx_test, TFy_test):
    model.eval()
    criterion = nn.MSELoss()
    total = 0.0

    with torch.no_grad():
        for i in range(0, len(X_test), BATCH_SIZE):
            xb = X_test[i : i + BATCH_SIZE].to(DEVICE)
            tfxb = TFx_test[i : i + BATCH_SIZE].to(DEVICE)
            tfyb = TFy_test[i : i + BATCH_SIZE].to(DEVICE)
            yb = y_test[i : i + BATCH_SIZE].to(DEVICE)

            inp = torch.cat([xb, tfxb], dim=-1)
            preds = model(inp, tfy=tfyb)
            total += criterion(preds, yb).item()

    return total / max(1, len(X_test) // BATCH_SIZE)


def compute_error_threshold(
    model, X_train, y_train, TFx_train, TFy_train, quantile=0.99
):
    model.eval()
    errors = []

    with torch.no_grad():
        for i in range(0, len(X_train), BATCH_SIZE):
            xb = X_train[i : i + BATCH_SIZE].to(DEVICE)
            tfxb = TFx_train[i : i + BATCH_SIZE].to(DEVICE)
            tfyb = TFy_train[i : i + BATCH_SIZE].to(DEVICE)
            yb = y_train[i : i + BATCH_SIZE].to(DEVICE)

            inp = torch.cat([xb, tfxb], dim=-1)
            preds = model(inp, tfy=tfyb)
            errors.append(torch.abs(preds - yb).cpu().numpy())

    errors = np.concatenate(errors).flatten()
    return float(np.quantile(errors, quantile))


def error_based_classification_metrics(y_true, y_pred, tau):
    errors = np.abs(y_pred - y_true)
    y_true_cls = (errors > tau).astype(int)
    y_pred_cls = (errors > tau).astype(int)

    metrics = {
        "precision": precision_score(y_true_cls, y_pred_cls, zero_division=0),
        "recall": recall_score(y_true_cls, y_pred_cls, zero_division=0),
        "f1": f1_score(y_true_cls, y_pred_cls, zero_division=0),
    }

    if len(np.unique(y_true_cls)) > 1:
        metrics["auc_roc"] = roc_auc_score(y_true_cls, errors)
    else:
        metrics["auc_roc"] = np.nan

    return metrics


# -------------------------------------------------
# Multi-step evaluation
# -------------------------------------------------
def evaluate_multi_step(model, X_test, y_test, TFx_test, TFy_test, steps=5):
    """
    Multi-step forecasting: predicts `steps` ahead recursively.
    """
    model.eval()
    mse_list = []

    with torch.no_grad():
        for i in range(0, len(X_test) - steps + 1, BATCH_SIZE):
            xb = X_test[i : i + BATCH_SIZE].to(DEVICE)
            tfxb = TFx_test[i : i + BATCH_SIZE].to(DEVICE)
            tfyb = TFy_test[i : i + BATCH_SIZE].to(DEVICE)

            batch_preds = []
            batch_input = xb.clone()

            # Predict recursively for 'steps'
            for s in range(steps):
                inp = torch.cat([batch_input, tfxb], dim=-1)
                pred = model(inp, tfy=tfyb)
                batch_preds.append(pred.cpu().numpy())

                # Shift window for next step
                if s < steps - 1:
                    batch_input = torch.cat(
                        [batch_input[:, 1:, :], pred[:, None, :]], dim=1
                    )

            # Compute MSE per step
            y_true_steps = y_test[i : i + BATCH_SIZE, :steps].cpu().numpy()
            batch_preds = np.stack(batch_preds, axis=1)  # shape: (batch, steps, 1)
            mse_step = np.mean(
                (batch_preds[:, :, 0] - y_true_steps[:, :steps]) ** 2, axis=0
            )
            mse_list.append(mse_step)

    # Average MSE across all batches
    mse_array = np.mean(np.stack(mse_list, axis=0), axis=0)
    return mse_array


# -------------------------------------------------
# Error summary (no labels)
# -------------------------------------------------
def error_summary(y_true, y_pred, tau):
    """
    Summarize prediction errors for unlabeled data.

    Args:
        y_true (np.ndarray): True values.
        y_pred (np.ndarray): Predicted values.
        tau (float): Threshold for unusually high errors.

    Returns:
        dict: mean_error, max_error, percent_exceed_tau
    """
    errors = np.abs(y_pred - y_true)
    exceed_count = np.sum(errors > tau)
    exceed_percent = exceed_count / len(errors) * 100

    return {
        "mean_error": float(np.mean(errors)),
        "max_error": float(np.max(errors)),
        "percent_exceed_tau": float(exceed_percent),
    }


# -------------------------------------------------
# Main evaluation
# -------------------------------------------------
if __name__ == "__main__":

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
    ) = load_and_concat_datasets(DATA_DIR, lag=LAG)

    # Model
    model = TransformerTimeSeriesWithLearnableTime2Vec(
        t2v_dim=T2V_DIM,
        model_dim=MODEL_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(DEVICE)

    print("Loading checkpoint:", MODEL_CHECKPOINT_PATH)
    model.load_state_dict(torch.load(MODEL_CHECKPOINT_PATH, map_location=DEVICE))
    model.eval()

    # ---- One-step test MSE ----
    mse = evaluate_test_mse(model, X_test, y_test, TFx_test, TFy_test)
    print(f"\nOne-step Test MSE: {mse:.6f}")

    # ---- Error threshold τ ----
    tau = compute_error_threshold(
        model,
        X_train,
        y_train,
        TFx_train,
        TFy_train,
        quantile=0.95,
    )
    print(f"Error threshold τ (95%): {tau:.6f}")

    # ---- Error-based classification (one-step) ----
    with torch.no_grad():
        preds = []
        trues = []

        for i in range(0, len(X_test), BATCH_SIZE):
            xb = X_test[i : i + BATCH_SIZE].to(DEVICE)
            tfxb = TFx_test[i : i + BATCH_SIZE].to(DEVICE)
            tfyb = TFy_test[i : i + BATCH_SIZE].to(DEVICE)

            inp = torch.cat([xb, tfxb], dim=-1)
            preds.append(model(inp, tfy=tfyb).cpu().numpy())
            trues.append(y_test[i : i + BATCH_SIZE].cpu().numpy())

    y_pred = np.concatenate(preds).flatten()
    y_true = np.concatenate(trues).flatten()

    cls_metrics = error_based_classification_metrics(y_true, y_pred, tau)

    print("\nOne-step error-based classification:")
    for k, v in cls_metrics.items():
        print(f"{k:>10}: {v:.4f}")

    # ---- Multi-step test MSE ----
    multi_step_horizon = 20  # Number of steps ahead
    multi_step_mse = evaluate_multi_step(
        model, X_test, y_test, TFx_test, TFy_test, steps=multi_step_horizon
    )
    print(f"\nMulti-step Test MSE ({multi_step_horizon} steps):")
    for step, mse_val in enumerate(multi_step_mse, 1):
        print(f" Step {step}: {mse_val:.6f}")

    # ---- One-step error summary (no labels) ----
    summary_metrics = error_summary(y_true, y_pred, tau)

    print("\nOne-step error summary (no anomaly labels):")
    for k, v in summary_metrics.items():
        print(f"{k:>20}: {v:.4f}")
