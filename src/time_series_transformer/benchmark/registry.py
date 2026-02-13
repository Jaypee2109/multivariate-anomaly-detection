"""Model factory registry for benchmarking."""

from __future__ import annotations

import logging
from typing import Callable

logger = logging.getLogger(__name__)

# name → callable that returns a fresh model instance
_REGISTRY: dict[str, Callable] = {}


def register_model(name: str, factory: Callable) -> None:
    """Register a model factory under *name*."""
    _REGISTRY[name] = factory


def get_factory(name: str) -> Callable:
    """Return the factory for *name*, or raise ``ValueError``."""
    if name not in _REGISTRY:
        raise ValueError(
            f"Model {name!r} not registered. Available: {list(_REGISTRY)}"
        )
    return _REGISTRY[name]


def list_models() -> list[str]:
    """Return all registered model names."""
    return list(_REGISTRY)


# ------------------------------------------------------------------
# Built-in models
# ------------------------------------------------------------------

def _register_defaults() -> None:
    from time_series_transformer.config import (
        ARIMA_ORDER,
        ARIMA_Z_THRESH,
        ISO_CONTAMINATION,
        LSTM_BATCH_SIZE,
        LSTM_DROPOUT,
        LSTM_EPOCHS,
        LSTM_ERROR_QUANTILE,
        LSTM_HIDDEN_SIZE,
        LSTM_LOOKBACK,
        LSTM_LR,
        LSTM_NUM_LAYERS,
        RANDOM_STATE,
        ROLLING_WINDOW,
        ROLLING_Z_THRESH,
    )
    from time_series_transformer.models.baseline.arima import (
        ARIMAResidualAnomalyDetector,
    )
    from time_series_transformer.models.baseline.isolation_forest import (
        IsolationForestAnomalyDetector,
    )
    from time_series_transformer.models.baseline.lstm import (
        LSTMForecastAnomalyDetector,
    )
    from time_series_transformer.models.baseline.rolling_zscore import (
        RollingZScoreAnomalyDetector,
    )

    register_model(
        "arima",
        lambda: ARIMAResidualAnomalyDetector(
            order=ARIMA_ORDER, z_thresh=ARIMA_Z_THRESH,
        ),
    )
    register_model(
        "isolation_forest",
        lambda: IsolationForestAnomalyDetector(
            contamination=ISO_CONTAMINATION, random_state=RANDOM_STATE,
        ),
    )
    register_model(
        "lstm",
        lambda: LSTMForecastAnomalyDetector(
            lookback=LSTM_LOOKBACK,
            hidden_size=LSTM_HIDDEN_SIZE,
            num_layers=LSTM_NUM_LAYERS,
            dropout=LSTM_DROPOUT,
            batch_size=LSTM_BATCH_SIZE,
            lr=LSTM_LR,
            epochs=LSTM_EPOCHS,
            error_quantile=LSTM_ERROR_QUANTILE,
            device="auto",
        ),
    )
    register_model(
        "rolling_zscore",
        lambda: RollingZScoreAnomalyDetector(
            window=ROLLING_WINDOW, z_thresh=ROLLING_Z_THRESH,
        ),
    )


_register_defaults()
