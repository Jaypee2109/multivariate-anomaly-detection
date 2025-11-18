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

    all_X, all_y, all_TF = [], [], []
    val_sets = []

    for file in glob.glob(f"{data_dir}/*.csv"):
        df = pd.read_csv(file, parse_dates=["timestamp"]).sort_values("timestamp")

        split = int(0.8 * len(df))

        # train
        X, y, TF = build_windows(
            df["value"].values[:split],
            df["timestamp"].values[:split],
            lag,
        )

        all_X.append(X)
        all_y.append(y)
        all_TF.append(TF)

        # val
        Xv, yv, TFv = build_windows(
            df["value"].values[split:],
            df["timestamp"].values[split:],
            lag,
        )

        val_sets.append((Xv, yv, TFv, file, df["timestamp"].values[split:]))

    X_all = torch.cat(all_X, dim=0)
    y_all = torch.cat(all_y, dim=0)
    TF_all = torch.cat(all_TF, dim=0)

    return X_all, y_all, TF_all, val_sets


# -------------------------
# Autoregressive forecasting function
# -------------------------
def forecast_autoregressive(model, init_window_vals, init_window_ts, steps=20, lag=12):
    """
    model: trained TransformerTimeSeries
    init_window_vals: numpy array of last 'lag' values
    init_window_ts: corresponding datetime timestamps
    steps: how many future steps to predict
    """
    model.eval()
    preds = []
    window_vals = init_window_vals.copy()

    for _ in range(steps):
        # Prepare lag tensor
        x_lag = (
            torch.tensor(window_vals[-lag:], dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(-1)
        )  # (1, lag, 1)

        # Prepare time features for the last lag
        last_ts = init_window_ts[len(window_vals) - 1]
        hour = torch.tensor([[last_ts.hour / 23.0]], dtype=torch.float32)
        weekday = torch.tensor([[last_ts.weekday() / 6.0]], dtype=torch.float32)
        tf = torch.cat([hour, weekday], dim=-1).unsqueeze(0)  # (1, 1, 2)

        # Model forward
        with torch.no_grad():
            pred = model(x_lag.to(device), time_features=tf.to(device))
        pred_val = pred.item()
        preds.append(pred_val)

        # Append to window for next step
        window_vals = list(window_vals) + [pred_val]

    return preds


def compute_time_features(ts_list):
    hours = np.array([t.hour / 23.0 for t in ts_list], dtype=np.float32)
    weekdays = np.array([t.weekday() / 6.0 for t in ts_list], dtype=np.float32)
    return np.stack([hours, weekdays], axis=-1)  # (seq_len, 2)


def build_windows(values, timestamps, lag):
    """
    Returns:
        X:  (N, lag, 1)
        y:  (N, 1)
        TF: (N, lag, 2)
    """
    values = np.array(values, dtype=np.float32)
    timestamps = pd.to_datetime(timestamps)

    hours = np.array([t.hour / 23.0 for t in timestamps], dtype=np.float32)
    weekdays = np.array([t.weekday() / 6.0 for t in timestamps], dtype=np.float32)
    time_feats = np.stack([hours, weekdays], axis=-1)

    X_list = []
    y_list = []
    TF_list = []

    for i in range(len(values) - lag):
        X_list.append(values[i : i + lag])
        y_list.append(values[i + lag])
        TF_list.append(time_feats[i : i + lag])

    X = np.array(X_list, dtype=np.float32)[:, :, None]  # (N, lag, 1)
    y = np.array(y_list, dtype=np.float32)[:, None]  # (N, 1)
    TF = np.array(TF_list, dtype=np.float32)  # (N, lag, 2)

    # THEN torchify
    return (
        torch.from_numpy(X),
        torch.from_numpy(y),
        torch.from_numpy(TF),
    )


def batches(X, y, TF, bs):
    idx = torch.randperm(len(X))
    for i in range(0, len(X), bs):
        j = idx[i : i + bs]
        yield X[j], y[j], TF[j]


def train_model(
    model,
    X_train,
    y_train,
    TF_train,
    X_val,
    y_val,
    TF_val,
    optimizer,
    scheduler,
    criterion,
    epochs=20,
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
            tfb = TF_train[idx].to(device)  # (B, lag, 2)

            inp = torch.cat([xb, tfb], dim=-1)  # (B, lag, 3)

            yb = y_train[idx].to(device)

            optimizer.zero_grad()
            out = model(inp)

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
                tfb = TF_val[i : i + batch_size].to(device)
                yb = y_val[i : i + batch_size].to(device)

                inp = torch.cat([xb, tfb], dim=-1)

                out = model(inp)
                val_loss += criterion(out, yb).item()

        val_loss /= n_val // batch_size

        print(f"Epoch {epoch:02d} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), "best_model.pth")

        if device.type == "mps":
            pass  # MPS has no empty_cache()


def forecast_autoregressive(model, init_vals, init_ts, steps, lag=12):

    model.eval()

    current_vals = list(init_vals)
    current_ts = list(pd.to_datetime(init_ts))

    preds = []

    for _ in range(steps):

        # build current lag window
        vals = np.array(current_vals[-lag:], dtype=np.float32)
        hours = np.array([t.hour / 23.0 for t in current_ts[-lag:]], dtype=np.float32)
        wdays = np.array(
            [t.weekday() / 6.0 for t in current_ts[-lag:]], dtype=np.float32
        )

        tf = np.stack([hours, wdays], axis=-1)  # (lag, 2)

        x = torch.tensor(vals)[None, :, None]  # (1, lag, 1)
        tf = torch.tensor(tf)[None, :, :]  # (1, lag, 2)

        inp = torch.cat([x, tf], dim=-1).to(device)

        with torch.no_grad():
            pred = model(inp).item()

        preds.append(pred)

        # extend window
        current_vals.append(pred)

        delta = current_ts[-1] - current_ts[-2]
        current_ts.append(current_ts[-1] + delta)

    return preds


# -------------------------
# Load datasets
# -------------------------
lag = 90
data_dir = "data/processed/nab/realTweets/realTweets"

X_train, y_train, TF_train, val_sets = load_and_concat_datasets(data_dir, lag=lag)

MODEL_DIM = 256
NUM_HEADS = 16
NUM_LAYERS = 16
EPOCHS = 8
BATCH_SIZE = 64

model = TransformerTimeSeries(
    input_dim=1 + 2,  # raw value + 2 time_feats AFTER lagging
    model_dim=MODEL_DIM,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS,
    dropout=0.1,
).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=1e-3,  # peak LR
    steps_per_epoch=len(X_train),
    epochs=EPOCHS,
    pct_start=0.1,  # 10% of training = warmup
    anneal_strategy="cos",  # cosine cooldown
)


# for now just use same val set — later fix properly
train_model(
    model,
    X_train,
    y_train,
    TF_train,
    X_train,
    y_train,
    TF_train,
    optimizer,
    scheduler,
    criterion,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
)


# -------------------------
# Sample prediction display
# -------------------------
for X_val, y_val, TF_val, fname, _ in val_sets:
    print("\n# -----------------------------------")
    print(f"# Sample autoregressive prediction for {fname}")
    print("# -----------------------------------")

    # Use the timestamps, not time features
    init_vals = X_val[0].squeeze(-1).cpu().numpy()
    init_ts = pd.to_datetime(val_sets[0][4][: len(init_vals)])  # <- correct

    steps = 20

    preds = forecast_autoregressive(model, init_vals, init_ts, steps=steps)

    # True future values
    true_vals = y_val[:steps].cpu().numpy().flatten().tolist()

    print("\n=== SAMPLE PREDICTIONS (first 20 steps) ===")
    print(f"{'Step':>4} | {'Prediction':>12} | {'Actual':>12}")
    print("-" * 38)

    for i in range(steps):
        p = preds[i]
        t = true_vals[i] if i < len(true_vals) else None
        if t is None:
            print(f"{i+1:>4} | {p:12.6f} | {'(no data)':>12}")
        else:
            print(f"{i+1:>4} | {p:12.6f} | {t:12.6f}")
