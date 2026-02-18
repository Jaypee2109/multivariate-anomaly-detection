"""Model factory registry for benchmarking."""

from __future__ import annotations

import logging
from collections.abc import Callable

logger = logging.getLogger(__name__)

# name → callable that returns a fresh model instance
_REGISTRY: dict[str, Callable] = {}
_MULTIVARIATE: set[str] = set()


def register_model(
    name: str,
    factory: Callable,
    *,
    multivariate: bool = False,
) -> None:
    """Register a model factory under *name*."""
    _REGISTRY[name] = factory
    if multivariate:
        _MULTIVARIATE.add(name)


def get_factory(name: str) -> Callable:
    """Return the factory for *name*, or raise ``ValueError``."""
    if name not in _REGISTRY:
        raise ValueError(f"Model {name!r} not registered. Available: {list(_REGISTRY)}")
    return _REGISTRY[name]


def list_models() -> list[str]:
    """Return all registered model names."""
    return list(_REGISTRY)


def is_multivariate(name: str) -> bool:
    """Return *True* if *name* is a multivariate model."""
    return name in _MULTIVARIATE


def list_multivariate_models() -> list[str]:
    """Return names of all registered multivariate models."""
    return [m for m in _REGISTRY if m in _MULTIVARIATE]


def list_univariate_models() -> list[str]:
    """Return names of all registered univariate models."""
    return [m for m in _REGISTRY if m not in _MULTIVARIATE]


# ------------------------------------------------------------------
# Built-in univariate models
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
            order=ARIMA_ORDER,
            z_thresh=ARIMA_Z_THRESH,
        ),
    )
    register_model(
        "isolation_forest",
        lambda: IsolationForestAnomalyDetector(
            contamination=ISO_CONTAMINATION,
            random_state=RANDOM_STATE,
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
            window=ROLLING_WINDOW,
            z_thresh=ROLLING_Z_THRESH,
        ),
    )


# ------------------------------------------------------------------
# Built-in multivariate models
# ------------------------------------------------------------------


def _register_multivariate_defaults() -> None:
    from time_series_transformer.config import (
        LSTM_AE_BATCH_SIZE,
        LSTM_AE_DROPOUT,
        LSTM_AE_EPOCHS,
        LSTM_AE_ERROR_QUANTILE,
        LSTM_AE_HIDDEN_SIZE,
        LSTM_AE_LATENT_DIM,
        LSTM_AE_LOOKBACK,
        LSTM_AE_LR,
        LSTM_AE_NUM_LAYERS,
        MULTI_ISO_CONTAMINATION,
        RANDOM_STATE,
        VAR_AGGREGATION,
        VAR_IC,
        VAR_MAXLAGS,
        VAR_Z_THRESH,
    )
    from time_series_transformer.models.multivariate.isolation_forest import (
        MultivariateIsolationForestDetector,
    )
    from time_series_transformer.models.multivariate.lstm_autoencoder import (
        LSTMAutoencoderAnomalyDetector,
    )
    from time_series_transformer.models.multivariate.var import (
        VARResidualAnomalyDetector,
    )

    register_model(
        "var",
        lambda: VARResidualAnomalyDetector(
            maxlags=VAR_MAXLAGS,
            ic=VAR_IC,
            z_thresh=VAR_Z_THRESH,
            aggregation=VAR_AGGREGATION,
        ),
        multivariate=True,
    )
    register_model(
        "multi_isolation_forest",
        lambda: MultivariateIsolationForestDetector(
            contamination=MULTI_ISO_CONTAMINATION,
            random_state=RANDOM_STATE,
        ),
        multivariate=True,
    )
    register_model(
        "lstm_autoencoder",
        lambda: LSTMAutoencoderAnomalyDetector(
            lookback=LSTM_AE_LOOKBACK,
            hidden_size=LSTM_AE_HIDDEN_SIZE,
            latent_dim=LSTM_AE_LATENT_DIM,
            num_layers=LSTM_AE_NUM_LAYERS,
            dropout=LSTM_AE_DROPOUT,
            batch_size=LSTM_AE_BATCH_SIZE,
            lr=LSTM_AE_LR,
            epochs=LSTM_AE_EPOCHS,
            error_quantile=LSTM_AE_ERROR_QUANTILE,
            device="auto",
        ),
        multivariate=True,
    )

    from time_series_transformer.config import (
        LSTM_FC_BATCH_SIZE,
        LSTM_FC_DROPOUT,
        LSTM_FC_EPOCHS,
        LSTM_FC_ERROR_QUANTILE,
        LSTM_FC_HIDDEN_SIZE,
        LSTM_FC_LOOKBACK,
        LSTM_FC_LR,
        LSTM_FC_NUM_LAYERS,
    )
    from time_series_transformer.models.multivariate.lstm_forecaster import (
        LSTMForecasterMultivariateDetector,
    )

    register_model(
        "lstm_forecaster",
        lambda: LSTMForecasterMultivariateDetector(
            lookback=LSTM_FC_LOOKBACK,
            hidden_size=LSTM_FC_HIDDEN_SIZE,
            num_layers=LSTM_FC_NUM_LAYERS,
            dropout=LSTM_FC_DROPOUT,
            batch_size=LSTM_FC_BATCH_SIZE,
            lr=LSTM_FC_LR,
            epochs=LSTM_FC_EPOCHS,
            error_quantile=LSTM_FC_ERROR_QUANTILE,
            device="auto",
        ),
        multivariate=True,
    )

    from time_series_transformer.config import (
        TRANAD_BATCH_SIZE,
        TRANAD_DIM_FF,
        TRANAD_DROPOUT,
        TRANAD_EPOCHS,
        TRANAD_ERROR_QUANTILE,
        TRANAD_LOOKBACK,
        TRANAD_LR,
        TRANAD_N_HEADS,
        TRANAD_NUM_LAYERS,
        TRANAD_SCORE_METRIC,
    )
    from time_series_transformer.models.multivariate.tranad import (
        TranADAnomalyDetector,
    )

    register_model(
        "tranad",
        lambda: TranADAnomalyDetector(
            lookback=TRANAD_LOOKBACK,
            n_heads=TRANAD_N_HEADS,
            dim_feedforward=TRANAD_DIM_FF,
            num_layers=TRANAD_NUM_LAYERS,
            dropout=TRANAD_DROPOUT,
            batch_size=TRANAD_BATCH_SIZE,
            lr=TRANAD_LR,
            epochs=TRANAD_EPOCHS,
            error_quantile=TRANAD_ERROR_QUANTILE,
            score_metric=TRANAD_SCORE_METRIC,
            device="auto",
        ),
        multivariate=True,
    )

    from time_series_transformer.config import (
        CUSTOM_TF_BATCH_SIZE,
        CUSTOM_TF_DIM_FF,
        CUSTOM_TF_DROPOUT,
        CUSTOM_TF_EPOCHS,
        CUSTOM_TF_ERROR_QUANTILE,
        CUSTOM_TF_LOOKBACK,
        CUSTOM_TF_LR,
        CUSTOM_TF_MODEL_DIM,
        CUSTOM_TF_NUM_HEADS,
        CUSTOM_TF_NUM_LAYERS,
        CUSTOM_TF_SCORE_METRIC,
        CUSTOM_TF_T2V_DIM,
    )
    from time_series_transformer.models.multivariate.custom_transformer import (
        CustomTransformerDetector,
    )

    register_model(
        "custom_transformer",
        lambda: CustomTransformerDetector(
            lookback=CUSTOM_TF_LOOKBACK,
            t2v_dim=CUSTOM_TF_T2V_DIM,
            model_dim=CUSTOM_TF_MODEL_DIM,
            num_heads=CUSTOM_TF_NUM_HEADS,
            num_layers=CUSTOM_TF_NUM_LAYERS,
            dim_feedforward=CUSTOM_TF_DIM_FF,
            dropout=CUSTOM_TF_DROPOUT,
            batch_size=CUSTOM_TF_BATCH_SIZE,
            lr=CUSTOM_TF_LR,
            epochs=CUSTOM_TF_EPOCHS,
            error_quantile=CUSTOM_TF_ERROR_QUANTILE,
            score_metric=CUSTOM_TF_SCORE_METRIC,
            device="auto",
        ),
        multivariate=True,
    )


_register_defaults()
_register_multivariate_defaults()
