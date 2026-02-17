"""TranAD (VLDB 2022) for multivariate time-series anomaly detection.

Adapted from https://github.com/imperial-qore/TranAD

Architecture: Transformer encoder with two Transformer decoders and a
self-conditioning mechanism.  Phase 1 reconstructs with zero conditioning;
Phase 2 feeds the squared error from Phase 1 as context to a second decoder.
The final anomaly score is the reconstruction error from the second decoder.

NOTE: The original TranAD uses *custom* Transformer layers (no LayerNorm,
LeakyReLU activation) — NOT PyTorch's built-in TransformerEncoderLayer /
TransformerDecoderLayer which include LayerNorm and use ReLU.  We replicate
the original custom layers below for a faithful comparison.
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
from torch.nn import TransformerDecoder, TransformerEncoder
from torch.utils.data import DataLoader, TensorDataset

from time_series_transformer.exceptions import DataValidationError, ModelNotFittedError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Positional Encoding (faithful to original TranAD implementation)
# ---------------------------------------------------------------------------


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding used in TranAD.

    Note: the original TranAD adds *both* sin and cos to every dimension
    (instead of interleaving).  We keep this behaviour for faithfulness.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model).float() * (-math.log(10000.0) / d_model)
        )
        pe += torch.sin(position * div_term)
        pe += torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # (max_len, 1, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor, pos: int = 0) -> torch.Tensor:
        x = x + self.pe[pos : pos + x.size(0), :]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# Custom Transformer layers — faithful to original TranAD dlutils.py
# (No LayerNorm, LeakyReLU activation instead of ReLU)
# ---------------------------------------------------------------------------


class _TranADEncoderLayer(nn.Module):
    """Transformer encoder layer matching the original TranAD implementation.

    Differences from ``nn.TransformerEncoderLayer``:
    - No ``LayerNorm`` (the original omits it entirely).
    - Uses ``LeakyReLU`` instead of ``ReLU``.
    """

    def __init__(
        self, d_model: int, nhead: int, dim_feedforward: int = 16, dropout: float = 0.0,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.LeakyReLU(True)

    def forward(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        src_key_padding_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        src2 = self.self_attn(src, src, src)[0]
        src = src + self.dropout1(src2)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        return src


class _TranADDecoderLayer(nn.Module):
    """Transformer decoder layer matching the original TranAD implementation.

    Differences from ``nn.TransformerDecoderLayer``:
    - No ``LayerNorm``.
    - Uses ``LeakyReLU`` instead of ``ReLU``.
    """

    def __init__(
        self, d_model: int, nhead: int, dim_feedforward: int = 16, dropout: float = 0.0,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = nn.LeakyReLU(True)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
        tgt_key_padding_mask: torch.Tensor | None = None,
        memory_key_padding_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        tgt2 = self.self_attn(tgt, tgt, tgt)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.multihead_attn(tgt, memory, memory)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt


# ---------------------------------------------------------------------------
# TranAD nn.Module
# ---------------------------------------------------------------------------


class TranADModel(nn.Module):
    """TranAD: Transformer encoder + two decoders with self-conditioning.

    Encoder: shared TransformerEncoder over ``cat(src, c)`` (d_model = 2F).
    Decoder 1: initial reconstruction (Phase 1).
    Decoder 2: self-conditioned reconstruction using Phase-1 error (Phase 2).
    """

    def __init__(
        self,
        n_features: int,
        n_window: int = 10,
        n_heads: int | None = None,
        dim_feedforward: int = 16,
        num_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_features = n_features
        self.n_window = n_window

        # Default: one head per feature (each head sees 2 dims in d_model=2F)
        if n_heads is None or n_heads == 0:
            n_heads = n_features

        self.pos_encoder = PositionalEncoding(2 * n_features, dropout, n_window)

        encoder_layer = _TranADEncoderLayer(
            d_model=2 * n_features,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers)

        decoder_layer1 = _TranADDecoderLayer(
            d_model=2 * n_features,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.transformer_decoder1 = TransformerDecoder(decoder_layer1, num_layers)

        decoder_layer2 = _TranADDecoderLayer(
            d_model=2 * n_features,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.transformer_decoder2 = TransformerDecoder(decoder_layer2, num_layers)

        self.fcn = nn.Sequential(nn.Linear(2 * n_features, n_features), nn.Sigmoid())

    def encode(
        self, src: torch.Tensor, c: torch.Tensor, tgt: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode *src* concatenated with conditioning signal *c*."""
        src = torch.cat((src, c), dim=2)  # (seq, batch, 2F)
        src = src * math.sqrt(self.n_features)
        src = self.pos_encoder(src)
        memory = self.transformer_encoder(src)
        tgt = tgt.repeat(1, 1, 2)  # (1, batch, 2F)
        return tgt, memory

    def forward(
        self, src: torch.Tensor, tgt: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with two-phase self-conditioning.

        Args:
            src: (seq_len, batch, n_features) — full window.
            tgt: (1, batch, n_features) — last timestep of the window.

        Returns:
            (x1, x2) each of shape (1, batch, n_features).
        """
        # Phase 1 — reconstruct without anomaly scores
        c = torch.zeros_like(src)
        x1 = self.fcn(self.transformer_decoder1(*self.encode(src, c, tgt)))

        # Phase 2 — self-conditioned using squared error from Phase 1
        c = (x1 - src) ** 2
        x2 = self.fcn(self.transformer_decoder2(*self.encode(src, c, tgt)))

        return x1, x2


# ---------------------------------------------------------------------------
# Detector wrapper (same API as LSTMAutoencoderAnomalyDetector)
# ---------------------------------------------------------------------------


@dataclass
class TranADAnomalyDetector:
    """TranAD-based multivariate anomaly detector.

    Wraps :class:`TranADModel` in a scikit-learn-like interface that matches
    the existing detector pattern (``fit`` / ``decision_function`` / ``predict``).

    Default hyper-parameters match the original TranAD paper for SMD.
    """

    lookback: int = 10
    n_heads: int = 0  # 0 = auto (= n_features)
    dim_feedforward: int = 16
    num_layers: int = 1
    dropout: float = 0.1
    batch_size: int = 128
    lr: float = 1e-4
    epochs: int = 5
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

        self.model: TranADModel | None = None
        self._trained: bool = False
        self.n_features_: int | None = None
        self.threshold_: float | None = None

    # ------------------------------------------------------------------ helpers

    def _make_windows(self, arr: np.ndarray) -> np.ndarray:
        """arr: (n_timesteps, n_features) -> (n_windows, lookback, n_features)"""
        T = self.lookback
        if len(arr) <= T:
            raise DataValidationError(
                f"Series length {len(arr)} too short for lookback={T}."
            )
        windows = np.lib.stride_tricks.sliding_window_view(arr, (T, arr.shape[1]))
        return windows.squeeze(axis=1).astype(np.float32)

    def _window_scores(self, windows: np.ndarray) -> np.ndarray:
        """Run inference and return per-window anomaly scores."""
        self.model.eval()
        # windows: (n_windows, lookback, n_features)
        # Transformer expects seq-first: (lookback, n_windows, n_features)
        src = torch.from_numpy(windows).permute(1, 0, 2).to(self._device)
        tgt = src[-1, :, :].unsqueeze(0)  # (1, n_windows, n_features)

        with torch.no_grad():
            _, x2 = self.model(src, tgt)

        # MSE between Phase-2 reconstruction and actual last timestep
        loss_fn = nn.MSELoss(reduction="none")
        per_elem = loss_fn(x2, tgt)  # (1, n_windows, n_features)
        per_window = per_elem.squeeze(0).mean(dim=1)  # (n_windows,)
        return per_window.cpu().numpy()

    def _overlap_average(self, window_scores: np.ndarray, n: int) -> np.ndarray:
        """Map window-level scores to per-timestep scores via overlap averaging."""
        T = self.lookback
        accum = np.zeros(n, dtype=np.float64)
        counts = np.zeros(n, dtype=np.float64)
        for i, s in enumerate(window_scores):
            accum[i : i + T] += s
            counts[i : i + T] += 1
        return np.where(counts > 0, accum / counts, 0.0)

    # ------------------------------------------------------------------ public API

    def fit(self, X: pd.DataFrame) -> None:
        self.n_features_ = X.shape[1]
        arr = X.values.astype(np.float32)
        windows = self._make_windows(arr)

        n_heads = self.n_heads if self.n_heads > 0 else self.n_features_

        self.model = TranADModel(
            n_features=self.n_features_,
            n_window=self.lookback,
            n_heads=n_heads,
            dim_feedforward=self.dim_feedforward,
            num_layers=self.num_layers,
            dropout=self.dropout,
        ).to(self._device)

        dataset = TensorDataset(torch.from_numpy(windows))
        loader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, drop_last=False,
        )

        # AdamW with weight decay — matches the original TranAD repo
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.9)
        loss_fn = nn.MSELoss()

        self.model.train()
        for epoch in range(self.epochs):
            n = epoch + 1
            epoch_loss = 0.0
            n_samples = 0
            for (batch_x,) in loader:
                batch_x = batch_x.to(self._device)
                local_bs = batch_x.shape[0]

                # seq-first: (lookback, batch, n_features)
                window = batch_x.permute(1, 0, 2)
                elem = window[-1, :, :].unsqueeze(0)  # (1, batch, n_features)

                opt.zero_grad()
                x1, x2 = self.model(window, elem)

                # TranAD loss: shift weight from decoder1 -> decoder2 over epochs
                l1 = loss_fn(x1, elem)
                l2 = loss_fn(x2, elem)
                loss = (1 / n) * l1 + (1 - 1 / n) * l2

                loss.backward(retain_graph=True)
                opt.step()

                epoch_loss += loss.item() * local_bs
                n_samples += local_bs

            scheduler.step()

            if (epoch + 1) % max(1, self.epochs // 3) == 0 or epoch == 0:
                logger.info(
                    "Epoch %d/%d, loss=%.6f",
                    epoch + 1,
                    self.epochs,
                    epoch_loss / n_samples,
                )

        # Compute threshold from training reconstruction errors
        train_scores = self._window_scores(windows)
        timestep_scores = self._overlap_average(train_scores, len(arr))
        self.threshold_ = float(np.quantile(timestep_scores, self.error_quantile))
        logger.info(
            "TranAD threshold: quantile(%.3f)=%.6f (train mean=%.6f, std=%.6f)",
            self.error_quantile,
            self.threshold_,
            float(np.mean(timestep_scores)),
            float(np.std(timestep_scores)),
        )
        self._trained = True

    def decision_function(self, X: pd.DataFrame) -> pd.Series:
        if not self._trained or self.model is None:
            raise ModelNotFittedError("Call fit() first.")

        arr = X.values.astype(np.float32)
        windows = self._make_windows(arr)
        window_scores = self._window_scores(windows)
        timestep_scores = self._overlap_average(window_scores, len(arr))

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
                    "n_heads": self.n_heads,
                    "dim_feedforward": self.dim_feedforward,
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
        logger.info("Saved TranAD checkpoint to %s", path)

    @classmethod
    def load_checkpoint(cls, path: str | Path) -> TranADAnomalyDetector:
        path = Path(path)
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        hp = ckpt["hyperparams"]
        detector = cls(**hp)
        detector.n_features_ = ckpt["n_features"]
        detector.threshold_ = ckpt["threshold"]

        n_heads = hp["n_heads"] if hp["n_heads"] > 0 else ckpt["n_features"]
        detector.model = TranADModel(
            n_features=ckpt["n_features"],
            n_window=hp["lookback"],
            n_heads=n_heads,
            dim_feedforward=hp["dim_feedforward"],
            num_layers=hp["num_layers"],
            dropout=hp["dropout"],
        ).to(detector._device)
        detector.model.load_state_dict(ckpt["state_dict"])
        detector._trained = True
        logger.info("Loaded TranAD checkpoint from %s", path)
        return detector
