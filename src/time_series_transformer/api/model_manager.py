"""Model lifecycle management for the inference server."""

from __future__ import annotations

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from time_series_transformer.exceptions import ConfigurationError
from time_series_transformer.models.baseline.base import BaseAnomalyDetector
from time_series_transformer.models.baseline.lstm import LSTMForecastAnomalyDetector

logger = logging.getLogger(__name__)

# slug -> (checkpoint filename, custom loader class or None for joblib default)
MODEL_LOADERS: dict[str, tuple[str, type | None]] = {
    "arima": ("arima_residual.joblib", None),
    "isolation_forest": ("isolation_forest.joblib", None),
    "rolling_zscore": ("rolling_zscore.joblib", None),
    "lstm": ("lstm_checkpoint.pt", LSTMForecastAnomalyDetector),
}

MODEL_DISPLAY_NAMES: dict[str, str] = {
    "arima": "ARIMA Residual",
    "isolation_forest": "Isolation Forest",
    "rolling_zscore": "Rolling Z-Score",
    "lstm": "LSTM Forecast",
}


class ModelManager:
    """Holds loaded model instances and provides inference methods."""

    def __init__(self) -> None:
        self._models: dict[str, Any] = {}
        self._load_times: dict[str, str] = {}
        self.checkpoint_dir: str = ""

    @property
    def loaded_model_names(self) -> list[str]:
        return list(self._models.keys())

    def load_from_directory(self, checkpoint_dir: Path) -> list[str]:
        """Scan *checkpoint_dir* for known checkpoint files and load them all.

        Returns list of loaded model slugs.
        """
        self.checkpoint_dir = str(checkpoint_dir)
        loaded: list[str] = []
        for slug, (filename, loader_cls) in MODEL_LOADERS.items():
            ckpt_path = checkpoint_dir / filename
            if not ckpt_path.exists():
                logger.debug("Checkpoint not found for %s: %s", slug, ckpt_path)
                continue
            try:
                if loader_cls is not None:
                    model = loader_cls.load_checkpoint(ckpt_path)
                else:
                    model = BaseAnomalyDetector.load_checkpoint(ckpt_path)
                self._models[slug] = model
                self._load_times[slug] = datetime.now().isoformat()
                loaded.append(slug)
                logger.info("Loaded %s from %s", slug, ckpt_path)
            except Exception:
                logger.exception("Failed to load %s from %s", slug, ckpt_path)
        return loaded

    def get_model(self, slug: str) -> Any:
        if slug not in self._models:
            raise ConfigurationError(f"Model '{slug}' is not loaded.")
        return self._models[slug]

    def detect(
        self,
        y: pd.Series,
        model_slugs: list[str] | None = None,
    ) -> dict[str, tuple[pd.Series, pd.Series, float]]:
        """Run anomaly detection with specified (or all) loaded models.

        Returns ``{slug: (anomalies_bool_series, scores_series, latency_ms)}``.
        """
        slugs = model_slugs or self.loaded_model_names
        results: dict[str, tuple[pd.Series, pd.Series, float]] = {}
        for slug in slugs:
            model = self.get_model(slug)
            t0 = time.time()
            scores = model.decision_function(y)
            anomalies = model.predict(y)
            latency_ms = (time.time() - t0) * 1000
            results[slug] = (anomalies, scores, latency_ms)
        return results

    def get_model_info(self, slug: str) -> dict[str, Any]:
        """Return introspection dict for a loaded model."""
        model = self.get_model(slug)
        params: dict[str, Any] = {}
        if hasattr(model, "__dataclass_fields__"):
            for k in model.__dataclass_fields__:
                val = getattr(model, k, None)
                if val is not None and not callable(val):
                    params[k] = val
        else:
            for attr in vars(model):
                if attr.startswith("_") or attr.endswith("_"):
                    continue
                val = getattr(model, attr)
                if val is None or callable(val):
                    continue
                if isinstance(val, (str, int, float, bool, tuple)):
                    params[attr] = val
        return {
            "name": slug,
            "display_name": MODEL_DISPLAY_NAMES.get(slug, slug),
            "model_class": type(model).__name__,
            "parameters": params,
        }
