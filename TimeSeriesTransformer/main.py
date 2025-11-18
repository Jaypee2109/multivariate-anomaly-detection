import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import glob
from datetime import datetime
from transformer import TransformerTimeSeries
import pandas as pd

# -------------------------------------------------
# Setup for Apple Silicon (M1/M2/M3)
# -------------------------------------------------
torch.set_float32_matmul_precision("medium")
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# -------------------------
# Load and concatenate datasets
# -------------------------
def load_and_concat_datasets(data_dir):
    all_train_X, all_train_y, all_train_tf = [], [], []
    val_sets = []

    for file in glob.glob(f"{data_dir}/*.csv"):
        df = pd.read_csv(file, parse_dates=["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)

        # Split train / val 80/20
        split_idx = int(0.8 * len(df))
        train_df = df.iloc[:split_idx]
        val_df = df.iloc[split_idx:]

        # Convert to tensors
        train_values = torch.tensor(train_df["value"].values, dtype=torch.float32)
        val_values = torch.tensor(val_df["value"].values, dtype=torch.float32)

        # Lag features
        lag = 12

        def create_lag_tensor(series):
            return torch.stack(
                [
                    series[i : -(lag - i - 1)] if i < lag - 1 else series[i:]
                    for i in range(lag)
                ],
                dim=-1,
            )

        X_train = create_lag_tensor(train_values[:-1]).unsqueeze(
            -1
        )  # (seq_len_lag, lag)
        y_train = train_values[lag:].unsqueeze(-1)

        X_val = create_lag_tensor(val_values[:-1]).unsqueeze(-1)
        y_val = val_values[lag:].unsqueeze(-1)

        # Time features
        def create_time_features(df):
            """
            df: pandas DataFrame with 'timestamp' column
            Returns: torch tensor of shape (seq_len, 2) with normalized hour and weekday
            """
            ts = pd.to_datetime(
                df["timestamp"].values
            )  # ensures each is a pd.Timestamp
            hours = torch.tensor(
                [t.hour / 23.0 for t in ts], dtype=torch.float32
            ).unsqueeze(-1)
            weekdays = torch.tensor(
                [t.weekday() / 6.0 for t in ts], dtype=torch.float32
            ).unsqueeze(-1)
            return torch.cat([hours, weekdays], dim=-1)

        tf_train = create_time_features(train_df)
        tf_val = create_time_features(val_df)

        # Add to master training set
        all_train_X.append(X_train)
        all_train_y.append(y_train)
        all_train_tf.append(tf_train)

        # Save val set for later predictions
        val_sets.append((X_val, y_val, tf_val, file, val_df["timestamp"].values))

    # Concatenate all datasets along the first dimension (time axis)
    X_train_all = torch.cat(all_train_X, dim=0)
    y_train_all = torch.cat(all_train_y, dim=0)
    TF_train_all = torch.cat(all_train_tf, dim=0)

    return X_train_all, y_train_all, TF_train_all, val_sets


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


def build_windows(values, timestamps, seq_len):
    """
    Returns:
      X: (num_windows, seq_len, 1)
      y: (num_windows, 1)
      TF: (num_windows, seq_len, 2)
    """
    time_features = compute_time_features(timestamps)

    X_list = []
    y_list = []
    TF_list = []

    for i in range(len(values) - seq_len - 1):
        seq = values[i : i + seq_len]
        nxt = values[i + seq_len]
        ts_feats = time_features[i : i + seq_len]

        X_list.append(seq)
        y_list.append(nxt)
        TF_list.append(ts_feats)

    # Convert lists to numpy arrays FIRST (fast)
    X = np.array(X_list, dtype=np.float32)[:, :, None]  # (N, seq_len, 1)
    y = np.array(y_list, dtype=np.float32)[:, None]  # (N, 1)
    TF = np.array(TF_list, dtype=np.float32)  # (N, seq_len, 2)

    # THEN convert to torch tensors ONCE (no warning)
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
    criterion,
    epochs=20,
    batch_size=64,
):

    n_train = len(X_train)
    n_val = len(X_val)

    best_val = float("inf")

    for epoch in range(1, epochs + 1):

        # ==========================
        # TRAIN
        # ==========================
        model.train()
        perm = torch.randperm(n_train)

        train_loss = 0
        batches = 0

        for i in range(0, n_train, batch_size):
            idx = perm[i : i + batch_size]

            xb = X_train[idx].to(device)
            yb = y_train[idx].to(device)
            tfb = TF_train[idx].to(device)

            optimizer.zero_grad()
            out = model(xb, time_features=tfb)
            loss = criterion(out, yb)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            batches += 1

        avg_train = train_loss / batches

        # ==========================
        # VALIDATION
        # ==========================
        model.eval()
        val_loss = 0
        val_batches = 0

        with torch.no_grad():
            for i in range(0, n_val, batch_size):
                xb = X_val[i : i + batch_size].to(device)
                yb = y_val[i : i + batch_size].to(device)
                tfb = TF_val[i : i + batch_size].to(device)

                out = model(xb, time_features=tfb)
                loss = criterion(out, yb)

                val_loss += loss.item()
                val_batches += 1

        avg_val = val_loss / val_batches

        # ==========================
        # STATUS
        # ==========================
        print(f"Epoch {epoch:02d} | Train: {avg_train:.4f} | Val: {avg_val:.4f}")

        if avg_val < best_val:
            best_val = avg_val
            torch.save(model.state_dict(), "best_model.pth")

        if device.type == "mps":
            torch.mps.empty_cache()


def forecast_autoregressive(model, init_window_vals, init_window_ts, steps):
    """
    init_window_vals: (150,) last seq_len values
    init_window_ts:   (150,) timestamps aligned to those values
    steps: number of steps to forecast
    """
    model.eval()

    seq_len = len(init_window_vals)

    current_vals = init_window_vals.copy()
    current_ts = init_ts.tolist()

    predictions = []

    for _ in range(steps):
        # Build time features for the current full window
        tf = compute_time_features(current_ts)
        tf = torch.tensor(tf, dtype=torch.float32)[None, :, :]

        x = torch.tensor(current_vals, dtype=torch.float32)[:, None]
        x = x[None, :, :]

        x = x.to(device)
        tf = tf.to(device)

        with torch.no_grad():
            pred = model(x, tf).item()

        predictions.append(pred)

        # Slide window forward by 1
        current_vals = np.append(current_vals[1:], pred)
        current_ts = current_ts[1:] + [
            current_ts[-1] + (current_ts[-1] - current_ts[-2])
        ]

    return predictions


# -------------------------
# Load datasets
# -------------------------
data_dir = "data/processed/nab/realTweets/realTweets"
X_train_all, y_train_all, TF_train_all, val_sets = load_and_concat_datasets(data_dir)

# -------------------------
# Initialize model
# -------------------------
model = TransformerTimeSeries(
    input_dim=12 + 2,  # 12 lags + 2 time features
    model_dim=128,
    num_heads=8,
    num_layers=2,
    dropout=0.1,
).to(device)

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# -------------------------
# Train the model
# -------------------------
train_model(
    model,
    X_train_all,
    y_train_all,
    TF_train_all,
    X_train_all,
    y_train_all,
    TF_train_all,  # Using same as val just for now
    optimizer,
    criterion,
    epochs=25,
    batch_size=64,
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
