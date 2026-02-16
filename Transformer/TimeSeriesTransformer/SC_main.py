import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import pandas as pd
import numpy as np

from transformer import TransformerTimeSeries


# -------------
# Device Setup
# -------------
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# -------------
# Data Loading
# -------------
def load_and_concat_datasets(data_dir, lag=12):
    X_train_list, y_train_list = [], []
    TFx_train_list, TFy_train_list = [], []

    X_val_list, y_val_list = [], []
    TFx_val_list, TFy_val_list = [], []

    val_sets = []

    for file in glob.glob(f"{data_dir}/*.csv"):
        df = pd.read_csv(file, parse_dates=["timestamp"]).sort_values("timestamp")
        split = int(0.8 * len(df))

        Xtr, ytr, TFxtr, TFytr = build_windows(
            df["value"].values[:split],
            df["timestamp"].values[:split],
            lag,
        )
        X_train_list.append(Xtr)
        y_train_list.append(ytr)
        TFx_train_list.append(TFxtr)
        TFy_train_list.append(TFytr)

        Xv, yv, TFxv, TFyv = build_windows(
            df["value"].values[split:],
            df["timestamp"].values[split:],
            lag,
        )
        X_val_list.append(Xv)
        y_val_list.append(yv)
        TFx_val_list.append(TFxv)
        TFy_val_list.append(TFyv)

        timestamps = df["timestamp"].values[split:]
        val_sets.append((Xv, yv, TFxv, TFyv, file, timestamps))

    X_train = torch.cat(X_train_list)
    y_train = torch.cat(y_train_list)
    TFx_train = torch.cat(TFx_train_list)
    TFy_train = torch.cat(TFy_train_list)

    X_val = torch.cat(X_val_list)
    y_val = torch.cat(y_val_list)
    TFx_val = torch.cat(TFx_val_list)
    TFy_val = torch.cat(TFy_val_list)

    return (
        X_train,
        y_train,
        TFx_train,
        TFy_train,
        X_val,
        y_val,
        TFx_val,
        TFy_val,
        val_sets,
    )


def build_windows(values, timestamps, lag):
    values = np.array(values, dtype=np.float32)
    timestamps = pd.to_datetime(timestamps)

    hours = np.array([t.hour / 23.0 for t in timestamps], dtype=np.float32)
    wdays = np.array([t.weekday() / 6.0 for t in timestamps], dtype=np.float32)
    time_feats = np.stack([hours, wdays], axis=-1)

    X_list, y_list, TFx_list, TFy_list = [], [], [], []

    for i in range(len(values) - lag):
        X_list.append(values[i : i + lag])
        TFx_list.append(time_feats[i : i + lag])
        y_list.append(values[i + lag])
        TFy_list.append(time_feats[i + lag])

    X = torch.tensor(np.array(X_list)[:, :, None])
    y = torch.tensor(np.array(y_list)[:, None])
    TFx = torch.tensor(np.array(TFx_list))
    TFy = torch.tensor(np.array(TFy_list))

    return X.float(), y.float(), TFx.float(), TFy.float()


# ---------------------------
# Autoregressive forecasting
# ---------------------------
def forecast_autoregressive(model, init_vals, init_ts, steps=20, lag=12):
    model.eval()
    device = next(model.parameters()).device

    vals = list(init_vals)
    ts = list(pd.to_datetime(init_ts))
    preds = []

    for _ in range(steps):
        window_vals = np.array(vals[-lag:], dtype=np.float32)
        hours = np.array([t.hour / 23.0 for t in ts[-lag:]], dtype=np.float32)
        wdays = np.array([t.weekday() / 6.0 for t in ts[-lag:]], dtype=np.float32)
        tfx = np.stack([hours, wdays], axis=-1)

        x = torch.tensor(window_vals)[None, :, None].to(device)
        tfx = torch.tensor(tfx)[None, :, :].to(device)
        inp = torch.cat([x, tfx], dim=-1)

        next_ts = ts[-1] + (ts[-1] - ts[-2])
        tfy = (
            torch.tensor([[next_ts.hour / 23.0, next_ts.weekday() / 6.0]])
            .float()
            .to(device)
        )

        with torch.no_grad():
            pred = model(inp, tfy=tfy).item()

        preds.append(pred)
        vals.append(pred)
        ts.append(next_ts)

    return preds


# -------------------
# Training
# -------------------
def train_single_device(args):
    device = get_device()
    print(f"Using device: {device}")

    (
        X_train,
        y_train,
        TFx_train,
        TFy_train,
        X_val,
        y_val,
        TFx_val,
        TFy_val,
        val_sets,
    ) = load_and_concat_datasets(args["data_dir"], lag=args["lag"])

    train_ds = TensorDataset(X_train, y_train, TFx_train, TFy_train)
    val_ds = TensorDataset(X_val, y_val, TFx_val, TFy_val)

    train_dl = DataLoader(
        train_ds,
        batch_size=args["batch_size"],
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=args["batch_size"],
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    model = TransformerTimeSeries(
        t2v_dim=32,
        model_dim=args["model_dim"],
        num_heads=args["num_heads"],
        num_layers=args["num_layers"],
        dropout=0.2,
    )

    # Wrap in DataParallel for multi-GPU
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args["lr"])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args["lr"],
        epochs=args["epochs"],
        steps_per_epoch=len(train_dl),
    )

    best_val = float("inf")

    for epoch in range(args["epochs"]):
        model.train()
        for xb, yb, tfxb, tfyb in train_dl:
            xb, yb, tfxb, tfyb = (
                xb.to(device),
                yb.to(device),
                tfxb.to(device),
                tfyb.to(device),
            )
            inp = torch.cat([xb, tfxb], dim=-1)

            optimizer.zero_grad()
            out = model(inp, tfy=tfyb)
            loss = criterion(out, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        model.eval()
        val_loss = 0.0
        count = 0

        with torch.no_grad():
            for xb, yb, tfxb, tfyb in val_dl:
                xb, yb, tfxb, tfyb = (
                    xb.to(device),
                    yb.to(device),
                    tfxb.to(device),
                    tfyb.to(device),
                )
                inp = torch.cat([xb, tfxb], dim=-1)
                out = model(inp, tfy=tfyb)
                val_loss += criterion(out, yb).item()
                count += 1

        val_loss /= count
        print(f"Epoch {epoch+1}/{args['epochs']} - Val Loss: {val_loss:.6f}")

        # Save checkpoint
        os.makedirs(args["save_dir"], exist_ok=True)
        ckpt_path = os.path.join(args["save_dir"], f"model_epoch_{epoch}.pth")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved best model to {ckpt_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", type=str, default="data/processed/nab/realTweets/realTweets"
    )
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--lag", type=int, default=90)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--model_dim", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-5)

    args = vars(parser.parse_args())

    train_single_device(args)
