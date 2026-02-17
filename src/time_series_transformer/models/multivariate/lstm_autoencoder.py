"""LSTM Autoencoder for multivariate time-series anomaly detection."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from time_series_transformer.exceptions import DataValidationError, ModelNotFittedError

logger = logging.getLogger(__name__)


class LSTMAutoencoder(nn.Module):
    """LSTM-based autoencoder for multivariate time series.

    Encoder: LSTM → last hidden → linear → latent
    Decoder: latent (repeated T times) → LSTM → linear → reconstruction
    """

    def __init__(
        self,
        n_features: int,
        hidden_size: int = 64,
        latent_dim: int = 32,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim

        # Encoder
        self.encoder_lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.encoder_fc = nn.Linear(hidden_size, latent_dim)

        # Decoder
        self.decoder_lstm = nn.LSTM(
            input_size=latent_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.decoder_fc = nn.Linear(hidden_size, n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, n_features) → reconstruction of same shape."""
        batch_size, seq_len, _ = x.shape

        # Encode
        _, (h_n, _) = self.encoder_lstm(x)
        latent = self.encoder_fc(h_n[-1])  # (batch, latent_dim)

        # Decode: repeat latent across time
        decoder_input = latent.unsqueeze(1).repeat(1, seq_len, 1)
        decoder_out, _ = self.decoder_lstm(decoder_input)
        reconstruction = self.decoder_fc(decoder_out)
        return reconstruction


@dataclass
class LSTMAutoencoderAnomalyDetector:
    """LSTM autoencoder-based multivariate anomaly detector.

    - ``fit()``: train autoencoder to reconstruct windows of
      ``(lookback, n_features)``
    - ``decision_function()``: reconstruction error as anomaly score
    - ``predict()``: threshold at ``error_quantile``
    """

    lookback: int = 30
    hidden_size: int = 64
    latent_dim: int = 32
    num_layers: int = 1
    dropout: float = 0.0
    batch_size: int = 64
    lr: float = 1e-3
    epochs: int = 30
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

        self.model: LSTMAutoencoder | None = None
        self._trained: bool = False
        self.n_features_: int | None = None
        self.threshold_: float | None = None

    # ------------------------------------------------------------------ helpers

    def _make_windows(self, arr: np.ndarray) -> np.ndarray:
        """arr: (n_timesteps, n_features) → (n_windows, lookback, n_features)"""
        T = self.lookback
        if len(arr) <= T:
            raise DataValidationError(
                f"Series length {len(arr)} too short for lookback={T}."
            )
        windows = np.lib.stride_tricks.sliding_window_view(arr, (T, arr.shape[1]))
        # sliding_window_view produces (n_windows, 1, T, n_features) — squeeze
        return windows.squeeze(axis=1).astype(np.float32)

    def _window_errors(self, windows: np.ndarray, recon: np.ndarray) -> np.ndarray:
        """Compute per-window reconstruction errors using the configured metric."""
        if self.score_metric == "mae":
            return np.mean(np.abs(windows - recon), axis=(1, 2))
        return np.mean((windows - recon) ** 2, axis=(1, 2))

    def _overlap_average(self, window_errors: np.ndarray, n: int) -> np.ndarray:
        """Map window-level errors to per-timestep scores via overlap averaging."""
        T = self.lookback
        accum = np.zeros(n, dtype=np.float64)
        counts = np.zeros(n, dtype=np.float64)
        for i, err in enumerate(window_errors):
            accum[i : i + T] += err
            counts[i : i + T] += 1
        return np.where(counts > 0, accum / counts, 0.0)

    # ------------------------------------------------------------------ public API

    def fit(self, X: pd.DataFrame) -> None:
        self.n_features_ = X.shape[1]
        arr = X.values.astype(np.float32)
        windows = self._make_windows(arr)

        dataset = TensorDataset(torch.from_numpy(windows))
        loader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, drop_last=False,
        )

        self.model = LSTMAutoencoder(
            n_features=self.n_features_,
            hidden_size=self.hidden_size,
            latent_dim=self.latent_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
        ).to(self._device)

        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            n_samples = 0
            for (batch_x,) in loader:
                batch_x = batch_x.to(self._device)
                opt.zero_grad()
                recon = self.model(batch_x)
                loss = loss_fn(recon, batch_x)
                loss.backward()
                opt.step()
                epoch_loss += loss.item() * batch_x.size(0)
                n_samples += batch_x.size(0)
            if (epoch + 1) % 10 == 0:
                logger.info(
                    "Epoch %d/%d, loss=%.6f",
                    epoch + 1,
                    self.epochs,
                    epoch_loss / n_samples,
                )

        # Compute threshold on training overlap-averaged scores (same as
        # decision_function) so the threshold matches the prediction scale.
        self.model.eval()
        with torch.no_grad():
            all_windows = torch.from_numpy(windows).to(self._device)
            recon = self.model(all_windows).cpu().numpy()

        window_errors = self._window_errors(windows, recon)
        train_scores = self._overlap_average(window_errors, len(arr))

        self.threshold_ = float(np.quantile(train_scores, self.error_quantile))
        logger.info(
            "AE threshold: quantile(%.3f)=%.6f (train mean=%.6f, std=%.6f)",
            self.error_quantile, self.threshold_,
            float(np.mean(train_scores)), float(np.std(train_scores)),
        )
        self._trained = True

    def decision_function(self, X: pd.DataFrame) -> pd.Series:
        if not self._trained or self.model is None:
            raise ModelNotFittedError("Call fit() first.")

        arr = X.values.astype(np.float32)
        windows = self._make_windows(arr)

        self.model.eval()
        with torch.no_grad():
            all_windows = torch.from_numpy(windows).to(self._device)
            recon = self.model(all_windows).cpu().numpy()

        window_errors = self._window_errors(windows, recon)
        timestep_scores = self._overlap_average(window_errors, len(arr))

        return pd.Series(
            timestep_scores.astype(np.float32),
            index=X.index,
            name="anomaly_score",
        )

    def predict(self, X: pd.DataFrame) -> pd.Series:
        scores = self.decision_function(X)
        threshold = self.threshold_ if self.threshold_ is not None else scores.quantile(self.error_quantile)
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
                    "hidden_size": self.hidden_size,
                    "latent_dim": self.latent_dim,
                    "num_layers": self.num_layers,
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
        logger.info("Saved LSTM-AE checkpoint to %s", path)

    @classmethod
    def load_checkpoint(cls, path: str | Path) -> LSTMAutoencoderAnomalyDetector:
        path = Path(path)
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        hp = ckpt["hyperparams"]
        detector = cls(**hp)
        detector.n_features_ = ckpt["n_features"]
        detector.threshold_ = ckpt["threshold"]
        detector.model = LSTMAutoencoder(
            n_features=ckpt["n_features"],
            hidden_size=hp["hidden_size"],
            latent_dim=hp["latent_dim"],
            num_layers=hp["num_layers"],
            dropout=hp["dropout"],
        ).to(detector._device)
        detector.model.load_state_dict(ckpt["state_dict"])
        detector._trained = True
        logger.info("Loaded LSTM-AE checkpoint from %s", path)
        return detector
