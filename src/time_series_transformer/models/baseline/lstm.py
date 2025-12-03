from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class LSTMForecaster(nn.Module):
    """
    Simple univariate LSTM forecaster: given a window of length T,
    predict the next value.
    """

    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 32,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, 1)
        out, _ = self.lstm(x)
        # use last hidden state
        last_hidden = out[:, -1, :]
        pred = self.fc(last_hidden)  # (batch, 1)
        return pred


@dataclass
class LSTMForecastAnomalyDetector:
    """
    LSTM forecasting baseline:

    - fit(): train LSTM to predict next value from previous `lookback` values
    - decision_function(): forecast error as anomaly score
    - predict(): label anomalies as scores >= error_quantile
    """

    lookback: int = 48
    hidden_size: int = 32
    num_layers: int = 1
    dropout: float = 0.0
    batch_size: int = 64
    lr: float = 1e-3
    epochs: int = 20
    error_quantile: float = 0.99
    device: str = "auto"  # "auto", "cpu", or "cuda"

    def __post_init__(self):
        # Resolve device robustly
        if self.device == "auto":
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            if self.device.startswith("cuda") and not torch.cuda.is_available():
                print("[LSTM] CUDA requested but not available, falling back to CPU.")
                self._device = "cpu"
            else:
                self._device = self.device

        self.model: Optional[LSTMForecaster] = None
        self.mean_: Optional[float] = None
        self.std_: Optional[float] = None
        self._trained: bool = False

    # ------- helpers -------

    def _standardize(self, y: pd.Series) -> np.ndarray:
        arr = y.values.astype(np.float32)
        if self.mean_ is None or self.std_ is None:
            self.mean_ = float(arr.mean())
            self.std_ = float(arr.std() + 1e-8)
        return (arr - self.mean_) / self.std_

    def _make_windows(self, arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        arr: 1D standardized values
        returns X: (n_samples, lookback, 1), y: (n_samples, 1)
        """
        T = self.lookback
        if len(arr) <= T:
            raise ValueError("Series too short for given lookback.")
        xs, ys = [], []
        for i in range(len(arr) - T):
            xs.append(arr[i : i + T])
            ys.append(arr[i + T])
        X = np.array(xs, dtype=np.float32)[..., None]  # (n, T, 1)
        y = np.array(ys, dtype=np.float32)[..., None]  # (n, 1)
        return X, y

    # ------- public API -------

    def fit(self, y_train: pd.Series) -> None:
        """
        Train LSTM to forecast next value from previous `lookback` values.
        """
        arr = self._standardize(y_train)
        X, y = self._make_windows(arr)

        dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
        )

        self.model = LSTMForecaster(
            input_size=1,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
        ).to(self._device)

        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        self.model.train()
        for _ in range(self.epochs):
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self._device)
                batch_y = batch_y.to(self._device)
                opt.zero_grad()
                pred = self.model(batch_x)
                loss = loss_fn(pred, batch_y)
                loss.backward()
                opt.step()

        self._trained = True

    def _compute_errors_for_series(self, y: pd.Series) -> pd.Series:
        """
        Compute forecast errors for a given series.
        Returns a Series aligned to y.index with NaN for the first `lookback` points.
        """
        if not self._trained or self.model is None:
            raise RuntimeError("Call fit() before decision_function().")

        arr = self._standardize(y)
        X, y_true = self._make_windows(arr)

        self.model.eval()
        with torch.no_grad():
            X_t = torch.from_numpy(X).to(self._device)
            preds = self.model(X_t).cpu().numpy().squeeze(-1)

        y_true = y_true.squeeze(-1)
        errors = np.abs(preds - y_true)  # MAE

        full_errors = np.full_like(arr, np.nan, dtype=np.float32)
        full_errors[self.lookback :] = errors

        return pd.Series(full_errors, index=y.index)

    def decision_function(self, y: pd.Series) -> pd.Series:
        """
        Return anomaly scores (forecast error) aligned with y.index.
        """
        return self._compute_errors_for_series(y)

    def predict(self, y: pd.Series) -> pd.Series:
        """
        Convert scores to binary anomalies using the configured error_quantile.
        """
        scores = self.decision_function(y)
        valid_scores = scores.dropna()
        if len(valid_scores) == 0:
            return pd.Series(False, index=y.index)

        thr = valid_scores.quantile(self.error_quantile)
        anom = scores >= thr
        anom = anom.fillna(False)
        return anom
