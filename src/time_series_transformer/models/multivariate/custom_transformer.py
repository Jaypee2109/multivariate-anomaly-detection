"""Custom Transformer Forecaster for multivariate time-series anomaly detection.

- Derives minute_of_hour and hour_of_day from absolute position indices
  (assuming 1-minute sampling) and feeds them through learnable Time2Vec
- Changes from univariate (1 value) to multivariate (n_features) input/output
- Uses forecast error (MSE) as anomaly score

Architecture
------------
Features + Time2Vec(minute_of_hour, hour_of_day) → Linear → Transformer Encoder
(causal mask) → Cross-Attention (query = Time2Vec(next_pos), key/value = encoder
output) → MLP Decoder → next-step prediction (n_features)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from time_series_transformer.exceptions import DataValidationError, ModelNotFittedError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Time2Vec — from Lars's implementation (learnableTime2Vec.py)
# ---------------------------------------------------------------------------


class LearnableTime2Vec(nn.Module):
    """Learnable Time2Vec positional encoding.

    Combines a linear component with learnable sinusoidal components
    to encode temporal position information.

    Input: ``(..., in_dim)`` → Output: ``(..., out_dim)``
    """

    def __init__(self, in_dim: int = 1, out_dim: int = 16):
        super().__init__()
        self.w0 = nn.Parameter(torch.randn(in_dim))
        self.b0 = nn.Parameter(torch.randn(1))
        self.W = nn.Parameter(torch.randn(out_dim - 1, in_dim))
        self.B = nn.Parameter(torch.randn(out_dim - 1))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t.to(self.w0.dtype)
        v0 = torch.matmul(t, self.w0) + self.b0
        z = torch.einsum("...i,ki->...k", t, self.W) + self.B
        vp = torch.sin(z)
        return torch.cat([v0.unsqueeze(-1), vp], dim=-1)  # (..., out_dim)


# ---------------------------------------------------------------------------
# Transformer Forecaster Module
# ---------------------------------------------------------------------------


def _generate_causal_mask(seq_len: int) -> torch.Tensor:
    """Upper-triangular boolean mask: ``True`` = masked (future positions)."""
    return torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()


class CustomTransformerForecaster(nn.Module):
    """Transformer encoder + cross-attention decoder for next-step forecasting.

    Architecture adapted from Lars's Time2Vec transformer:

    1. Input features concatenated with Time2Vec temporal encoding
    2. Linear projection to ``model_dim``
    3. Transformer encoder with causal self-attention mask
    4. Cross-attention: query from Time2Vec(next position),
       key/value from encoder output
    5. MLP decoder predicts next timestep's feature values

    Time2Vec receives two derived time features per position:
    ``minute_of_hour`` and ``hour_of_day``, computed from absolute
    position indices assuming 1-minute sampling intervals.
    """

    def __init__(
        self,
        n_features: int,
        t2v_dim: int = 16,
        model_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_features = n_features
        self.model_dim = model_dim

        # Time2Vec with 2 derived time features (minute_of_hour, hour_of_day)
        self.time2vec = LearnableTime2Vec(in_dim=2, out_dim=t2v_dim)

        # Input: features + time2vec → model_dim
        self.input_proj = nn.Linear(n_features + t2v_dim, model_dim)

        # Query projection (Time2Vec of next position → model_dim)
        self.query_proj = nn.Linear(t2v_dim, model_dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Cross-attention: query from target position, key/value from encoder
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=model_dim, num_heads=num_heads, batch_first=True
        )

        # Decoder MLP
        self.decoder = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.ReLU(),
            nn.Linear(model_dim, n_features),
        )

    @staticmethod
    def _derive_time_features(abs_positions: torch.Tensor) -> torch.Tensor:
        """Derive (minute_of_hour, hour_of_day) from absolute minute indices.

        Args:
            abs_positions: integer tensor of any shape

        Returns:
            ``(..., 2)`` with minute_of_hour in [0, 1] and hour_of_day in [0, 1].
        """
        minute_of_hour = (abs_positions % 60).float() / 59.0
        hour_of_day = ((abs_positions // 60) % 24).float() / 23.0
        return torch.stack([minute_of_hour, hour_of_day], dim=-1)

    def forward(self, src: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        """Forward pass: predict the next timestep.

        Args:
            src: ``(B, seq_len, n_features)``
            offsets: ``(B,)`` — absolute starting index (in minutes) of each window

        Returns:
            ``(B, n_features)`` — predicted feature values at the next step.
        """
        _, seq_len, _ = src.shape
        device = src.device

        # Absolute positions for each step in the window
        abs_pos = offsets.unsqueeze(1) + torch.arange(seq_len, device=device).unsqueeze(0)
        time_feats = self._derive_time_features(abs_pos)  # (B, seq_len, 2)

        # Time2Vec temporal encoding
        t2v = self.time2vec(time_feats)  # (B, seq_len, t2v_dim)

        # Concat features + time2vec, then project
        x = torch.cat([src, t2v], dim=-1)
        x = self.input_proj(x)  # (B, seq_len, model_dim)

        # Causal transformer encoder
        mask = _generate_causal_mask(seq_len).to(device)
        x = self.transformer_encoder(x, mask=mask)

        # Query: Time2Vec of the *next* position (one step beyond window)
        next_abs = (offsets + seq_len).unsqueeze(1)  # (B, 1)
        next_time = self._derive_time_features(next_abs)  # (B, 1, 2)
        query_t2v = self.time2vec(next_time)  # (B, 1, t2v_dim)
        query = self.query_proj(query_t2v)  # (B, 1, model_dim)

        # Cross-attention: query attends to full encoder output
        attended, _ = self.cross_attention(query, x, x)  # (B, 1, model_dim)

        return self.decoder(attended.squeeze(1))  # (B, n_features)


# ---------------------------------------------------------------------------
# Detector wrapper (same API as TranAD / LSTM Autoencoder detectors)
# ---------------------------------------------------------------------------


@dataclass
class CustomTransformerDetector:
    """Custom Transformer forecast-based multivariate anomaly detector.

    Adapted from Lars's Time2Vec + cross-attention architecture.  Trains
    to predict the next timestep from a lookback window, then uses the
    forecast error (MSE) as the anomaly score — points where the model
    predicts poorly are flagged as anomalous.
    """

    lookback: int = 30
    t2v_dim: int = 16
    model_dim: int = 64
    num_heads: int = 4
    num_layers: int = 2
    dim_feedforward: int = 256
    dropout: float = 0.1
    batch_size: int = 64
    lr: float = 1e-3
    epochs: int = 15
    error_quantile: float = 0.99
    score_metric: str = "mse"
    device: str = "auto"

    def __post_init__(self) -> None:
        if self.device == "auto":
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            if self.device.startswith("cuda") and not torch.cuda.is_available():
                logger.warning("CUDA not available, falling back to CPU.")
                self._device = "cpu"
            else:
                self._device = self.device

        self.model: CustomTransformerForecaster | None = None
        self._trained: bool = False
        self.n_features_: int | None = None
        self.threshold_: float | None = None

    # ------------------------------------------------------------------ helpers

    def _make_windows(
        self, arr: np.ndarray, start_offset: int = 0
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build sliding windows for forecast training.

        Args:
            arr: ``(n_timesteps, n_features)``
            start_offset: absolute minute index of the first row in *arr*
                (default 0 for training data).

        Returns:
            ``(X, y, offsets)`` where ``X`` is ``(n_windows, lookback, n_features)``,
            ``y`` is ``(n_windows, n_features)`` (next-step targets), and
            ``offsets`` is ``(n_windows,)`` absolute starting indices.
        """
        T = self.lookback
        if len(arr) <= T:
            raise DataValidationError(f"Series length {len(arr)} too short for lookback={T}.")
        windows = np.lib.stride_tricks.sliding_window_view(arr, (T + 1, arr.shape[1]))
        windows = windows.squeeze(axis=1)
        X = windows[:, :T, :].astype(np.float32)
        y = windows[:, T, :].astype(np.float32)
        offsets = (np.arange(len(X)) + start_offset).astype(np.int64)
        return X, y, offsets

    def _per_window_errors(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Compute per-window forecast errors."""
        if self.score_metric == "mae":
            return np.mean(np.abs(y_true - y_pred), axis=1)
        return np.mean((y_true - y_pred) ** 2, axis=1)

    def _overlap_average(self, window_errors: np.ndarray, n: int) -> np.ndarray:
        """Map window-level errors to per-timestep scores via overlap averaging."""
        T = self.lookback
        accum = np.zeros(n, dtype=np.float64)
        counts = np.zeros(n, dtype=np.float64)
        for i, err in enumerate(window_errors):
            accum[i : i + T + 1] += err
            counts[i : i + T + 1] += 1
        return np.where(counts > 0, accum / counts, 0.0)

    # ------------------------------------------------------------------ public API

    def fit(self, X: pd.DataFrame) -> None:
        self.n_features_ = X.shape[1]
        arr = X.values.astype(np.float32)
        X_win, y_win, off_win = self._make_windows(arr)

        dataset = TensorDataset(
            torch.from_numpy(X_win),
            torch.from_numpy(y_win),
            torch.from_numpy(off_win),
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)

        self.model = CustomTransformerForecaster(
            n_features=self.n_features_,
            t2v_dim=self.t2v_dim,
            model_dim=self.model_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
        ).to(self._device)

        n_batches = math.ceil(len(dataset) / self.batch_size)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            opt,
            max_lr=self.lr,
            steps_per_epoch=n_batches,
            epochs=self.epochs,
            pct_start=0.1,
            anneal_strategy="cos",
        )
        loss_fn = nn.MSELoss()

        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            n_samples = 0
            for batch_x, batch_y, batch_off in loader:
                batch_x = batch_x.to(self._device)
                batch_y = batch_y.to(self._device)
                batch_off = batch_off.to(self._device)
                opt.zero_grad()
                pred = self.model(batch_x, batch_off)
                loss = loss_fn(pred, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
                scheduler.step()
                epoch_loss += loss.item() * batch_x.size(0)
                n_samples += batch_x.size(0)
            if (epoch + 1) % max(1, self.epochs // 3) == 0 or epoch == 0:
                logger.info(
                    "Epoch %d/%d, loss=%.6f",
                    epoch + 1,
                    self.epochs,
                    epoch_loss / n_samples,
                )

        # Compute threshold on training forecast errors
        self.model.eval()
        with torch.no_grad():
            all_x = torch.from_numpy(X_win).to(self._device)
            all_off = torch.from_numpy(off_win).to(self._device)
            preds = self.model(all_x, all_off).cpu().numpy()

        errors = self._per_window_errors(y_win, preds)
        self.threshold_ = float(np.quantile(errors, self.error_quantile))
        logger.info(
            "Custom Transformer threshold: quantile(%.3f)=%.6f (train mean=%.6f, std=%.6f)",
            self.error_quantile,
            self.threshold_,
            float(np.mean(errors)),
            float(np.std(errors)),
        )
        self._trained = True

    def decision_function(self, X: pd.DataFrame) -> pd.Series:
        if not self._trained or self.model is None:
            raise ModelNotFittedError("Call fit() first.")

        arr = X.values.astype(np.float32)
        X_win, y_win, off_win = self._make_windows(arr)

        self.model.eval()
        with torch.no_grad():
            all_x = torch.from_numpy(X_win).to(self._device)
            all_off = torch.from_numpy(off_win).to(self._device)
            preds = self.model(all_x, all_off).cpu().numpy()

        window_errors = self._per_window_errors(y_win, preds)
        timestep_scores = self._overlap_average(window_errors, len(arr))

        return pd.Series(
            timestep_scores.astype(np.float32),
            index=X.index,
            name="anomaly_score",
        )

    def predict(self, X: pd.DataFrame) -> pd.Series:
        scores = self.decision_function(X)
        threshold = (
            self.threshold_ if self.threshold_ is not None else scores.quantile(self.error_quantile)
        )
        return (scores >= threshold).rename("is_anomaly")

    # ------------------------------------------------------------------ checkpointing

    def save_checkpoint(self, path: str | Path) -> None:
        if not self._trained or self.model is None:
            raise ModelNotFittedError("Cannot save: model not trained.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "threshold": self.threshold_,
                "n_features": self.n_features_,
                "hyperparams": {
                    "lookback": self.lookback,
                    "t2v_dim": self.t2v_dim,
                    "model_dim": self.model_dim,
                    "num_heads": self.num_heads,
                    "num_layers": self.num_layers,
                    "dim_feedforward": self.dim_feedforward,
                    "dropout": self.dropout,
                    "batch_size": self.batch_size,
                    "lr": self.lr,
                    "epochs": self.epochs,
                    "error_quantile": self.error_quantile,
                    "score_metric": self.score_metric,
                },
            },
            path,
        )
        logger.info("Saved Custom Transformer checkpoint to %s", path)

    @classmethod
    def load_checkpoint(cls, path: str | Path) -> CustomTransformerDetector:
        path = Path(path)
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        hp = ckpt["hyperparams"]
        detector = cls(**hp)
        detector.n_features_ = ckpt["n_features"]
        detector.threshold_ = ckpt["threshold"]
        detector.model = CustomTransformerForecaster(
            n_features=ckpt["n_features"],
            t2v_dim=hp["t2v_dim"],
            model_dim=hp["model_dim"],
            num_heads=hp["num_heads"],
            num_layers=hp["num_layers"],
            dim_feedforward=hp["dim_feedforward"],
            dropout=hp["dropout"],
        ).to(detector._device)
        detector.model.load_state_dict(ckpt["state_dict"])
        detector._trained = True
        logger.info("Loaded Custom Transformer checkpoint from %s", path)
        return detector
