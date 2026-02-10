from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from time_series_transformer.config import (
    ARIMA_ORDER,
    ARIMA_Z_THRESH,
    ARTIFACTS_DIR,
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
    TRAIN_RATIO,
)
from time_series_transformer.data_pipeline.data_loading import load_timeseries
from time_series_transformer.evaluation import summarize_anomalies
from time_series_transformer.models.baseline.arima import ARIMAResidualAnomalyDetector
from time_series_transformer.models.baseline.isolation_forest import (
    IsolationForestAnomalyDetector,
)
from time_series_transformer.models.baseline.lstm import LSTMForecastAnomalyDetector
from time_series_transformer.split import train_test_split_series
from time_series_transformer.utils.anomaly_io import save_anomaly_artifacts

logger = logging.getLogger(__name__)

# Mapping from CLI names to display names used as dict keys
MODEL_REGISTRY: dict[str, str] = {
    "arima": "ARIMA Residual",
    "isolation_forest": "Isolation Forest",
    "lstm": "LSTM Forecast",
}


def _seed_everything(seed: int) -> None:
    """Seed all random number generators for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_all_models() -> dict:
    """Construct all available baseline models with current config."""
    return {
        "ARIMA Residual": ARIMAResidualAnomalyDetector(
            order=ARIMA_ORDER,
            z_thresh=ARIMA_Z_THRESH,
        ),
        "Isolation Forest": IsolationForestAnomalyDetector(
            contamination=ISO_CONTAMINATION,
            random_state=RANDOM_STATE,
        ),
        "LSTM Forecast": LSTMForecastAnomalyDetector(
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
    }


def run_pipeline(
    csv_path: Path,
    y_true_labels: pd.Series | None = None,
    log_to_mlflow: bool = False,
    model_names: Sequence[str] | None = None,
    save_checkpoints: bool = False,
    load_checkpoint_dir: Path | None = None,
) -> None:
    # Seed for reproducibility
    _seed_everything(RANDOM_STATE)

    # 1. Load data
    y = load_timeseries(csv_path)

    # 2. Train/test split
    y_train, y_test = train_test_split_series(y, train_ratio=TRAIN_RATIO)

    # 3. Define models (optionally filtered)
    all_models = _build_all_models()

    if model_names is not None:
        selected_display = {MODEL_REGISTRY[n] for n in model_names if n in MODEL_REGISTRY}
        models = {k: v for k, v in all_models.items() if k in selected_display}
        if not models:
            logger.warning("No valid models selected. Available: %s", list(MODEL_REGISTRY.keys()))
            return
        logger.info("Training selected models: %s", list(models.keys()))
    else:
        models = all_models

    # Load LSTM checkpoint if requested
    if load_checkpoint_dir is not None and "LSTM Forecast" in models:
        ckpt_path = load_checkpoint_dir / "lstm_checkpoint.pt"
        if ckpt_path.exists():
            logger.info("Loading LSTM checkpoint from %s", ckpt_path)
            models["LSTM Forecast"] = LSTMForecastAnomalyDetector.load_checkpoint(ckpt_path)
        else:
            logger.warning("LSTM checkpoint not found at %s, training from scratch.", ckpt_path)

    # Checkpoint directory
    ckpt_dir = ARTIFACTS_DIR / "checkpoints"
    if save_checkpoints:
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    # 4. MLflow setup
    mlflow_mod = None
    if log_to_mlflow:
        try:
            import mlflow

            from time_series_transformer.mlflow_utils import (
                log_anomaly_summary,
                log_data_hash,
                log_environment_info,
                log_params_from_model,
                log_point_metrics,
                log_range_metrics,
                setup_mlflow,
            )

            mlflow_mod = mlflow
            setup_mlflow()
        except ImportError:
            logger.warning("mlflow not installed, skipping tracking.")
            log_to_mlflow = False

    scores_dict = {}
    anomalies_dict = {}

    # 5. Train and evaluate each model
    for name, model in models.items():
        run_ctx = (
            mlflow_mod.start_run(run_name=f"{name} — {csv_path.stem}")
            if log_to_mlflow
            else nullcontext()
        )

        with run_ctx:
            if log_to_mlflow:
                log_environment_info()
                log_data_hash(csv_path)
                mlflow_mod.log_params(
                    {
                        "dataset": csv_path.name,
                        "train_ratio": TRAIN_RATIO,
                        "random_state": RANDOM_STATE,
                        "train_size": len(y_train),
                        "test_size": len(y_test),
                    }
                )
                log_params_from_model(name, model)

            # Skip fit if model was loaded from checkpoint
            already_loaded = (
                name == "LSTM Forecast"
                and load_checkpoint_dir is not None
                and hasattr(model, "_trained")
                and model._trained
            )

            fit_start = time.time()
            if already_loaded:
                logger.info("Skipping fit for %s (loaded from checkpoint)", name)
            else:
                model.fit(y_train)
            fit_duration = time.time() - fit_start

            # Save checkpoint after training
            if save_checkpoints and name == "LSTM Forecast" and not already_loaded:
                ckpt_path = ckpt_dir / "lstm_checkpoint.pt"
                model.save_checkpoint(ckpt_path)
                logger.info("Saved LSTM checkpoint to %s", ckpt_path)

            scores = model.decision_function(y_test)
            anomalies = model.predict(y_test)

            scores_dict[name] = scores
            anomalies_dict[name] = anomalies

            result = summarize_anomalies(
                name,
                y_test,
                anomalies,
                scores,
                y_true_labels=y_true_labels,
            )

            if log_to_mlflow:
                mlflow_mod.log_metric("fit_time_seconds", fit_duration)
                log_anomaly_summary(len(y_test), int(anomalies.astype(bool).sum()))
                if result is not None:
                    pm, rm = result
                    log_point_metrics(pm)
                    if rm is not None:
                        log_range_metrics(rm)

    # 6. Save artifacts
    artifacts_path = ARTIFACTS_DIR / "anomalies" / "baseline_anomalies.csv"
    artifacts_path.parent.mkdir(parents=True, exist_ok=True)
    save_anomaly_artifacts(
        y_test=y_test,
        scores_dict=scores_dict,
        anomalies_dict=anomalies_dict,
        out_path=artifacts_path,
    )
