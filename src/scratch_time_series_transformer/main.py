import os
from dotenv import load_dotenv
from huggingface_hub import login
import torch
from datasets import load_dataset
import torch.nn as nn
import torch.optim as optim
from time_series_transformer import TransformerTimeSeries

# -------------------------------------------------
# Setup for Apple Silicon (M1/M2/M3)
# -------------------------------------------------
torch.set_float32_matmul_precision("medium")
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# -------------------------------------------------
# Hugging Face login
# -------------------------------------------------
load_dotenv()
token = os.getenv("TOKEN")
login(token)

# -------------------------------------------------
# Load a specific Chronos dataset
# -------------------------------------------------
dataset = load_dataset("autogluon/chronos_datasets", "monash_m3_monthly")

print(dataset)

# -------------------------------------------------
# Prepare input/output sequences
# -------------------------------------------------
window_size = 24
horizon = 2

# Take only the first half of the training set
num_train_series = len(dataset["train"])
half_train = dataset["train"].select(range(num_train_series))


# -------------------------------------------------
# Fast vectorized windowing with stride + sampling
# -------------------------------------------------
def create_windows_vectorized(target, window_size, horizon, stride=4, max_windows=300):
    """
    target: 1D list of float values
    returns X: (num_windows, window_size, 1)
            y: (num_windows, horizon)
    """
    t = torch.tensor(target, dtype=torch.float32)
    total_len = window_size + horizon
    # All sliding windows (vectorized, no loops)
    windows = t.unfold(0, total_len, stride)  # shape: (num_windows, total_len)

    if windows.shape[0] == 0:
        return None, None  # skip very short series

    # Split into X and y
    X = windows[:, :window_size].unsqueeze(-1)  # (num_windows, window_size, 1)
    y = windows[:, window_size:]  # (num_windows, horizon)

    # Subsample: keep at most max_windows
    if X.shape[0] > max_windows:
        idx = torch.randperm(X.shape[0])[:max_windows]
        X = X[idx]
        y = y[idx]

    return X, y


# -------------------------------------------------
# Build dataset using first N series
# -------------------------------------------------
N_SERIES = num_train_series
STRIDE = 4
MAX_WINDOWS = 1000000

X_list = []
y_list = []

subset = dataset["train"].select(range(N_SERIES))

for example in subset:
    target = example["target"]
    if isinstance(target, dict):
        target = target["values"]
    target = list(map(float, target))

    Xv, yv = create_windows_vectorized(
        target,
        window_size=window_size,
        horizon=horizon,
        stride=STRIDE,
        max_windows=MAX_WINDOWS,
    )

    if Xv is not None:
        X_list.append(Xv)
        y_list.append(yv)

# Concatenate all windows from all series
X = torch.cat(X_list, dim=0)
y = torch.cat(y_list, dim=0)

print("Final dataset shapes:", X.shape, y.shape)

# -------------------------------------------------
# Train/validation split
# -------------------------------------------------
train_size = int(0.8 * len(X))
X_train, y_train = X[:train_size], y[:train_size]
X_val, y_val = X[train_size:], y[train_size:]
# -------------------------------------------------
# Model setup
# -------------------------------------------------
model = TransformerTimeSeries(
    input_dim=1,
    model_dim=64,
    num_heads=4,
    num_layers=2,
    output_dim=horizon,
).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# -------------------------------------------------
# Mini-batch training loop
# -------------------------------------------------
epochs = 10
batch_size = 16


def batch_iter(X, y, batch_size):
    for i in range(0, len(X), batch_size):
        yield X[i : i + batch_size], y[i : i + batch_size]


for epoch in range(epochs):
    model.train()
    total_loss = 0.0

    for xb, yb in batch_iter(X_train, y_train, batch_size):
        xb, yb = xb.to(device), yb.to(device)

        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        total_loss += loss.item() * len(xb)

    avg_train_loss = total_loss / len(X_train)

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for xb, yb in batch_iter(X_val, y_val, batch_size):
            xb, yb = xb.to(device), yb.to(device)
            val_loss += criterion(model(xb), yb).item() * len(xb)
    avg_val_loss = val_loss / len(X_val)

    print(
        f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}"
    )

    # 🔹 Clear GPU memory between epochs
    if device.type == "mps":
        torch.mps.empty_cache()


# -------------------------------------------------
# Example prediction
# -------------------------------------------------
def predict_future(model, initial_window, forecast_steps, window_size, horizon, device):
    """
    model: trained model
    initial_window: tensor of shape (window_size, 1)
    forecast_steps: how many total steps to predict
    window_size: input window size (e.g., 24)
    horizon: model's output size (e.g., 12)
    """
    model.eval()
    preds = []

    # Current window to feed into the model
    current = initial_window.clone().to(device)  # shape (window_size, 1)

    steps_done = 0

    with torch.no_grad():
        while steps_done < forecast_steps:
            # Model expects shape (batch, seq_len, features)
            inp = current.unsqueeze(0)  # (1, window_size, 1)
            out = model(inp)  # (1, horizon)

            # Clamp negatives
            out = torch.clamp(out, min=0)

            # Convert to 1D tensor
            out_vals = out[0]  # shape: (horizon,)

            # Append to results
            preds.extend(out_vals.cpu().tolist())

            # Slide window forward by horizon steps
            new_values = torch.cat([current[:, 0], out_vals])[-window_size:]
            current = new_values.unsqueeze(-1)

            steps_done += horizon

    # Trim in case we went past forecast_steps
    return preds[:forecast_steps]


# ------------------------------------------------------------
# Predict the next 20 time steps using autoregressive rollout
# ------------------------------------------------------------
initial_window = X_val[0].cpu()  # shape (window_size, 1)
future_steps = 20

pred_long = predict_future(
    model=model,
    initial_window=initial_window,
    forecast_steps=future_steps,
    window_size=window_size,
    horizon=horizon,
    device=device,
)

# ------------------------------------------------------------
# Extract TRUE future values for comparison
# ------------------------------------------------------------
# y_val holds true next-step targets per window.
# For multi-step comparison, we need the raw time series from which X_val[0] came.

# Recover the original target sequence for the corresponding validation window
# X_val[k] corresponds to window starting at index: (train_size + k*stride)
stride = STRIDE  # same stride used during window creation
raw_series = None

# Find which raw series X_val[0] belongs to
idx = train_size  # first validation window index in the concatenated set
window_global_index = idx

# Determine which original series & offset this window came from
cum = 0
for example in subset:
    target = example["target"]
    if isinstance(target, dict):
        target = target["values"]
    target = list(map(float, target))

    tlen = len(target)
    num_windows = max(0, (tlen - (window_size + horizon)) // stride + 1)

    if window_global_index < cum + num_windows:
        # This is the series we want
        series_start_index = (window_global_index - cum) * stride
        raw_series = target
        break

    cum += num_windows

# Extract the correct next "future_steps" true values
true_future = raw_series[
    series_start_index + window_size : series_start_index + window_size + future_steps
]

# Pad if the true series ends before future_steps
if len(true_future) < future_steps:
    true_future += [None] * (future_steps - len(true_future))


# ------------------------------------------------------------
# Print predictions and correct values side-by-side
# ------------------------------------------------------------
print("\nLong forecast (next 20 steps):")
print("Step | Prediction |   Actual")
print("----------------------------------")

for i, pred in enumerate(pred_long):
    truth = true_future[i]
    if truth is None:
        print(f"t+{i+1:<2}: {pred:.4f} |   (no data)")
    else:
        print(f"t+{i+1:<2}: {pred:.4f} | {truth:.4f}")

print("\nPred list:", pred_long)
print("True list:", true_future)
