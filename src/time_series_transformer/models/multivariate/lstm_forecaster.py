"""LSTM Forecaster for multivariate time-series anomaly detection."""

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


class MultivariateLSTMForecaster(nn.Module):
    """LSTM forecaster for multivariate time series.

    Given a window of ``(seq_len, n_features)``, predict the next
    ``(n_features,)`` values.
    """

    def __init__(
        self,
        n_features: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, n_features) → (batch, n_features)."""
        out, _ = self.lstm(x)
        last_hidden = out[:, -1, :]  # (batch, hidden_size)
        return self.fc(last_hidden)  # (batch, n_features)


@dataclass
class LSTMForecasterMultivariateDetector:
    """LSTM forecast-based multivariate anomaly detector.

    - ``fit()``: train LSTM to predict the next timestep from a lookback window
    - ``decision_function()``: forecast error (MSE) as anomaly score
    - ``predict()``: threshold at ``error_quantile``
    """

    lookback: int = 30
    hidden_size: int = 64
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

        self.model: MultivariateLSTMForecaster | None = None
        self._trained: bool = False
        self.n_features_: int | None = None
        self.threshold_: float | None = None

    # ------------------------------------------------------------------ helpers

    def _make_windows(self, arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """arr: (n_timesteps, n_features) → X (n_windows, lookback, n_features), y (n_windows, n_features)."""
        T = self.lookback
        if len(arr) <= T:
            raise DataValidationError(
                f"Series length {len(arr)} too short for lookback={T}."
            )
        # Windows of length lookback+1: input is [:lookback], target is [lookback]
        windows = np.lib.stride_tricks.sliding_window_view(
            arr, (T + 1, arr.shape[1])
        )
        windows = windows.squeeze(axis=1)  # (n_windows, T+1, n_features)
        X = windows[:, :T, :].astype(np.float32)  # (n_windows, T, n_features)
        y = windows[:, T, :].astype(np.float32)  # (n_windows, n_features)
        return X, y

    def _per_window_errors(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Compute per-window forecast errors using the configured metric."""
        if self.score_metric == "mae":
            return np.mean(np.abs(y_true - y_pred), axis=1)
        return np.mean((y_true - y_pred) ** 2, axis=1)

    # ------------------------------------------------------------------ public API

    def fit(self, X: pd.DataFrame) -> None:
        self.n_features_ = X.shape[1]
        arr = X.values.astype(np.float32)
        X_win, y_win = self._make_windows(arr)

        dataset = TensorDataset(
            torch.from_numpy(X_win), torch.from_numpy(y_win),
        )
        loader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, drop_last=False,
        )

        self.model = MultivariateLSTMForecaster(
            n_features=self.n_features_,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
        ).to(self._device)

        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            n_samples = 0
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self._device)
                batch_y = batch_y.to(self._device)
                opt.zero_grad()
                pred = self.model(batch_x)
                loss = loss_fn(pred, batch_y)
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

        # Compute threshold on training forecast errors
        self.model.eval()
        with torch.no_grad():
            all_x = torch.from_numpy(X_win).to(self._device)
            preds = self.model(all_x).cpu().numpy()

        errors = self._per_window_errors(y_win, preds)
        self.threshold_ = float(np.quantile(errors, self.error_quantile))
        logger.info(
            "Forecaster threshold: quantile(%.3f)=%.6f (train mean=%.6f, std=%.6f)",
            self.error_quantile, self.threshold_,
            float(np.mean(errors)), float(np.std(errors)),
        )
        self._trained = True

    def decision_function(self, X: pd.DataFrame) -> pd.Series:
        if not self._trained or self.model is None:
            raise ModelNotFittedError("Call fit() first.")

        arr = X.values.astype(np.float32)
        X_win, y_win = self._make_windows(arr)

        self.model.eval()
        with torch.no_grad():
            all_x = torch.from_numpy(X_win).to(self._device)
            preds = self.model(all_x).cpu().numpy()

        window_errors = self._per_window_errors(y_win, preds)  # (n_windows,)

        # Map window errors back to per-timestep scores by averaging
        # across all windows that contain each timestep.
        n = len(arr)
        T = self.lookback
        accum = np.zeros(n, dtype=np.float64)
        counts = np.zeros(n, dtype=np.float64)
        for i, err in enumerate(window_errors):
            accum[i : i + T + 1] += err
            counts[i : i + T + 1] += 1
        timestep_scores = np.where(counts > 0, accum / counts, 0.0)

        return pd.Series(
            timestep_scores.astype(np.float32),
            index=X.index,
            name="anomaly_score",
        )

    def predict(self, X: pd.DataFrame) -> pd.Series:
        scores = self.decision_function(X)
        threshold = (
            self.threshold_
            if self.threshold_ is not None
            else scores.quantile(self.error_quantile)
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
                    "hidden_size": self.hidden_size,
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
        logger.info("Saved LSTM Forecaster checkpoint to %s", path)

    @classmethod
    def load_checkpoint(cls, path: str | Path) -> LSTMForecasterMultivariateDetector:
        path = Path(path)
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        hp = ckpt["hyperparams"]
        detector = cls(**hp)
        detector.n_features_ = ckpt["n_features"]
        detector.threshold_ = ckpt["threshold"]
        detector.model = MultivariateLSTMForecaster(
            n_features=ckpt["n_features"],
            hidden_size=hp["hidden_size"],
            num_layers=hp["num_layers"],
            dropout=hp["dropout"],
        ).to(detector._device)
        detector.model.load_state_dict(ckpt["state_dict"])
        detector._trained = True
        logger.info("Loaded LSTM Forecaster checkpoint from %s", path)
        return detector
