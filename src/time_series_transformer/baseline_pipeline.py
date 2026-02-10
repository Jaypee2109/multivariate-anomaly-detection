from __future__ import annotations

import time
from contextlib import nullcontext
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

from time_series_transformer.config import (
    ARTIFACTS_DIR,
    RANDOM_STATE,
    TRAIN_RATIO,
    ROLLING_WINDOW,
    ROLLING_Z_THRESH,
    ARIMA_ORDER,
    ARIMA_Z_THRESH,
    ISO_CONTAMINATION,
    LSTM_LOOKBACK,
    LSTM_LR,
    LSTM_BATCH_SIZE,
    LSTM_DROPOUT,
    LSTM_EPOCHS,
    LSTM_ERROR_QUANTILE,
    LSTM_HIDDEN_SIZE,
    LSTM_NUM_LAYERS,
)
from time_series_transformer.data_pipeline.data_loading import load_timeseries
from time_series_transformer.split import train_test_split_series
from time_series_transformer.evaluation import summarize_anomalies
from time_series_transformer.models.baseline.arima import ARIMAResidualAnomalyDetector
from time_series_transformer.models.baseline.rolling_zscore import (
    RollingZScoreAnomalyDetector,
)
from time_series_transformer.models.baseline.isolation_forest import (
    IsolationForestAnomalyDetector,
)
from time_series_transformer.models.baseline.lstm import LSTMForecastAnomalyDetector
from time_series_transformer.utils.anomaly_io import save_anomaly_artifacts


def _seed_everything(seed: int) -> None:
    """Seed all random number generators for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_pipeline(
    csv_path: Path,
    y_true_labels: Optional[pd.Series] = None,
    log_to_mlflow: bool = False,
) -> None:
    # Seed for reproducibility
    _seed_everything(RANDOM_STATE)

    # 1. Load data
    y = load_timeseries(csv_path)

    # 2. Train/test split
    y_train, y_test = train_test_split_series(y, train_ratio=TRAIN_RATIO)

    # 3. Define models
    models = {
        # "Rolling Z-Score": RollingZScoreAnomalyDetector(
        #    window=ROLLING_WINDOW,
        #    z_thresh=ROLLING_Z_THRESH,
        # ),
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

    # 4. MLflow setup
    mlflow_mod = None
    if log_to_mlflow:
        try:
            import mlflow
            from time_series_transformer.mlflow_utils import (
                setup_mlflow,
                log_params_from_model,
                log_point_metrics,
                log_range_metrics,
                log_anomaly_summary,
                log_environment_info,
                log_data_hash,
            )
            mlflow_mod = mlflow
            setup_mlflow()
        except ImportError:
            print("Warning: mlflow not installed, skipping tracking.")
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
                mlflow_mod.log_params({
                    "dataset": csv_path.name,
                    "train_ratio": TRAIN_RATIO,
                    "random_state": RANDOM_STATE,
                    "train_size": len(y_train),
                    "test_size": len(y_test),
                })
                log_params_from_model(name, model)

            fit_start = time.time()
            model.fit(y_train)
            fit_duration = time.time() - fit_start

            scores = model.decision_function(y_test)
            anomalies = model.predict(y_test)

            scores_dict[name] = scores
            anomalies_dict[name] = anomalies

            result = summarize_anomalies(
                name, y_test, anomalies, scores,
                y_true_labels=y_true_labels,
            )

            if log_to_mlflow:
                mlflow_mod.log_metric("fit_time_seconds", fit_duration)
                log_anomaly_summary(
                    len(y_test), int(anomalies.astype(bool).sum())
                )
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
