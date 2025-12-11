import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import glob
from transformer import TransformerTimeSeries
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

    current_vals = list(init_window_vals)
    current_ts = list(pd.to_datetime(init_window_ts))

    preds = []

    for _ in range(steps):

        # ---- Build lag window ----
        vals = np.array(current_vals[-lag:], dtype=np.float32)

        hours = np.array([t.hour / 23.0 for t in current_ts[-lag:]], dtype=np.float32)
        wdays = np.array(
            [t.weekday() / 6.0 for t in current_ts[-lag:]], dtype=np.float32
        )
        tfx = np.stack([hours, wdays], axis=-1)  # (lag, 2)

        # Prepare input (value + TFx)
        x = torch.tensor(vals)[None, :, None]  # (1, lag, 1)
        tfx_tensor = torch.tensor(tfx)[None, :, :]  # (1, lag, 2)
        inp = torch.cat([x, tfx_tensor], dim=-1).to(device)  # (1, lag, 3)

        # ---- Build TFy for NEXT timestamp ----
        delta = current_ts[-1] - current_ts[-2]
        next_ts = current_ts[-1] + delta

        next_hour = next_ts.hour / 23.0
        next_wday = next_ts.weekday() / 6.0

        tfy = torch.tensor([[next_hour, next_wday]], dtype=torch.float32).to(
            device
        )  # (1, 2)

        # ---- Forward pass with TFy ----
        with torch.no_grad():
            pred = model(inp, tfy=tfy).item()

        preds.append(pred)

        # ---- Extend window ----
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
):

    n_train = len(X_train)
    n_val = len(X_val)
    best_val = float("inf")

    for epoch in range(1, epochs + 1):

        # ---------- TRAIN ----------
        model.train()
        perm = torch.randperm(n_train)

        total_loss = 0

        for i in range(0, n_train, batch_size):
            idx = perm[i : i + batch_size]

            xb = X_train[idx].to(device)  # (B, lag, 1)
            tfxb = TFx_train[idx].to(device)  # (B, lag, 2)
            tfyb = TFy_train[idx].to(device)  # (B, 2)
            yb = y_train[idx].to(device)

            # concat lag-value + lag-timefeatures
            inp = torch.cat([xb, tfxb], dim=-1)  # (B, lag, 3)

            optimizer.zero_grad()

            # ← pass TFy into model
            out = model(inp, tfy=tfyb)

            loss = criterion(out, yb)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        train_loss = total_loss / (n_train // batch_size)

        # ---------- VALID ----------
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for i in range(0, n_val, batch_size):

                xb = X_val[i : i + batch_size].to(device)
                tfxb = TFx_val[i : i + batch_size].to(device)
                tfyb = TFy_val[i : i + batch_size].to(device)
                yb = y_val[i : i + batch_size].to(device)

                inp = torch.cat([xb, tfxb], dim=-1)

                # ← pass TFy during validation as well
                out = model(inp, tfy=tfyb)

                val_loss += criterion(out, yb).item()

        val_loss /= n_val // batch_size

        print(f"Epoch {epoch:02d} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), "best_model.pth")


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

        # ---- Prepare TFy for NEXT timestamp ----
        delta = current_ts[-1] - current_ts[-2]  # assume uniform frequency
        next_ts = current_ts[-1] + delta

        tfy = torch.tensor(
            [[next_ts.hour / 23.0, next_ts.weekday() / 6.0]], dtype=torch.float32
        ).to(
            device
        )  # (1, 2)

        # ---- Model forward ----
        with torch.no_grad():
            pred = model(inp, tfy=tfy).item()

        preds.append(pred)

        # ---- Update window ----
        current_vals.append(pred)
        current_ts.append(next_ts)

    return preds


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
# Load datasets
# -------------------------
lag = 90
data_dir = "data/processed/nab/realTweets/realTweets"

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
) = load_and_concat_datasets(data_dir, lag=lag)


MODEL_DIM = 128
NUM_HEADS = 16
NUM_LAYERS = 8
EPOCHS = 5
BATCH_SIZE = 64

model = TransformerTimeSeries(
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
    steps_per_epoch=len(X_train),
    epochs=EPOCHS,
    pct_start=0.1,
    anneal_strategy="cos",
)

# -------------------------
# Train
# -------------------------
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
)

# -------------------------
# Load best model & evaluate on TEST set
# -------------------------
# model.load_state_dict(torch.load("best_model.pth"))
# test_loss = evaluate_test(
#     model, X_test, y_test, TFx_test, TFy_test, batch_size=BATCH_SIZE
# )

# print("\n===============================")
# print("        FINAL TEST LOSS        ")
# print("===============================")
# print(f"Test MSE: {test_loss:.6f}")
# print("===============================\n")

for X_test_i, y_test_i, TFx_test_i, TFy_test_i, fname, timestamps in test_sets:
    print("\n# -----------------------------------")
    print(f"# Sample autoregressive prediction (TEST SET) for {fname}")
    print("# -----------------------------------")

    init_vals = X_test_i[0].squeeze(-1).cpu().numpy()
    init_ts = pd.to_datetime(timestamps[: len(init_vals)])

    steps = 20
    preds = forecast_autoregressive(model, init_vals, init_ts, steps, lag=lag)
    true_vals = y_test_i[:steps].cpu().numpy().flatten().tolist()

    print("\n=== SAMPLE TEST PREDICTIONS (first 20 steps) ===")
    print(f"{'Step':>4} | {'Prediction':>12} | {'Actual':>12}")
    print("-" * 38)

    for i in range(steps):
        p = preds[i]
        t = true_vals[i]
        print(f"{i+1:>4} | {p:12.6f} | {t:12.6f}")
