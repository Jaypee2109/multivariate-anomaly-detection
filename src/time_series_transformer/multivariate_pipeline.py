"""Training pipeline for multivariate anomaly detection on SMD data."""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

import pandas as pd

from time_series_transformer.config import (
    ARTIFACTS_DIR,
    LSTM_AE_BATCH_SIZE,
    LSTM_AE_DROPOUT,
    LSTM_AE_EPOCHS,
    LSTM_AE_ERROR_QUANTILE,
    LSTM_AE_HIDDEN_SIZE,
    LSTM_AE_LATENT_DIM,
    LSTM_AE_LOOKBACK,
    LSTM_AE_LR,
    LSTM_AE_NUM_LAYERS,
    LSTM_AE_SCORE_METRIC,
    LSTM_FC_BATCH_SIZE,
    LSTM_FC_DROPOUT,
    LSTM_FC_EPOCHS,
    LSTM_FC_ERROR_QUANTILE,
    LSTM_FC_HIDDEN_SIZE,
    LSTM_FC_LOOKBACK,
    LSTM_FC_LR,
    LSTM_FC_NUM_LAYERS,
    LSTM_FC_SCORE_METRIC,
    MULTI_ISO_CONTAMINATION,
    RANDOM_STATE,
    SMD_BASE_DIR,
    VAR_AGGREGATION,
    VAR_IC,
    VAR_MAXLAGS,
    VAR_Z_THRESH,
)
from time_series_transformer.data_pipeline.smd_loading import load_smd_machine
from time_series_transformer.evaluation import compute_point_metrics
from time_series_transformer.models.multivariate.isolation_forest import (
    MultivariateIsolationForestDetector,
)
from time_series_transformer.models.multivariate.lstm_autoencoder import (
    LSTMAutoencoderAnomalyDetector,
)
from time_series_transformer.models.multivariate.lstm_forecaster import (
    LSTMForecasterMultivariateDetector,
)
from time_series_transformer.models.multivariate.var import VARResidualAnomalyDetector

logger = logging.getLogger(__name__)

MULTIVARIATE_MODEL_REGISTRY: dict[str, str] = {
    "var": "VAR Residual",
    "multi_isolation_forest": "Isolation Forest (MV)",
    "lstm_autoencoder": "LSTM Autoencoder",
    "lstm_forecaster": "LSTM Forecaster (MV)",
}


def _seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_multivariate_models() -> dict:
    return {
        "VAR Residual": VARResidualAnomalyDetector(
            maxlags=VAR_MAXLAGS,
            ic=VAR_IC,
            z_thresh=VAR_Z_THRESH,
            aggregation=VAR_AGGREGATION,
        ),
        "Isolation Forest (MV)": MultivariateIsolationForestDetector(
            contamination=MULTI_ISO_CONTAMINATION,
            random_state=RANDOM_STATE,
        ),
        "LSTM Autoencoder": LSTMAutoencoderAnomalyDetector(
            lookback=LSTM_AE_LOOKBACK,
            hidden_size=LSTM_AE_HIDDEN_SIZE,
            latent_dim=LSTM_AE_LATENT_DIM,
            num_layers=LSTM_AE_NUM_LAYERS,
            dropout=LSTM_AE_DROPOUT,
            batch_size=LSTM_AE_BATCH_SIZE,
            lr=LSTM_AE_LR,
            epochs=LSTM_AE_EPOCHS,
            error_quantile=LSTM_AE_ERROR_QUANTILE,
            score_metric=LSTM_AE_SCORE_METRIC,
            device="auto",
        ),
        "LSTM Forecaster (MV)": LSTMForecasterMultivariateDetector(
            lookback=LSTM_FC_LOOKBACK,
            hidden_size=LSTM_FC_HIDDEN_SIZE,
            num_layers=LSTM_FC_NUM_LAYERS,
            dropout=LSTM_FC_DROPOUT,
            batch_size=LSTM_FC_BATCH_SIZE,
            lr=LSTM_FC_LR,
            epochs=LSTM_FC_EPOCHS,
            error_quantile=LSTM_FC_ERROR_QUANTILE,
            score_metric=LSTM_FC_SCORE_METRIC,
            device="auto",
        ),
    }


def run_multivariate_pipeline(
    machine_id: str,
    base_dir: Path | None = None,
    model_names: Sequence[str] | None = None,
    save_checkpoints: bool = False,
) -> None:
    """Train and evaluate multivariate anomaly detectors on a single SMD machine."""
    _seed_everything(RANDOM_STATE)

    if base_dir is None:
        base_dir = SMD_BASE_DIR

    # 1. Load data (pre-split, no normalization — SMD is already in [0,1])
    machine_data = load_smd_machine(machine_id, base_dir=base_dir, normalize=False)
    X_train = machine_data.train_df
    X_test = machine_data.test_df
    y_true = machine_data.test_labels

    # 2. Build models (optionally filtered)
    all_models = _build_multivariate_models()
    if model_names is not None:
        selected = {
            MULTIVARIATE_MODEL_REGISTRY[n]
            for n in model_names
            if n in MULTIVARIATE_MODEL_REGISTRY
        }
        models = {k: v for k, v in all_models.items() if k in selected}
        if not models:
            logger.warning(
                "No valid models selected. Available: %s",
                list(MULTIVARIATE_MODEL_REGISTRY.keys()),
            )
            return
    else:
        models = all_models

    # 3. Train and evaluate
    results_df = X_test.copy()
    results_df["is_anomaly"] = y_true.values

    for name, model in models.items():
        logger.info("Training %s on %s …", name, machine_id)
        safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        try:
            t0 = time.time()
            model.fit(X_train)
            fit_time = time.time() - t0
            logger.info("  fit_time=%.2fs", fit_time)

            scores = model.decision_function(X_test)
            anomalies = model.predict(X_test)
        except Exception as e:
            logger.warning("  %s failed on %s: %s — skipping", name, machine_id, e)
            continue

        # Store per-model results for artifact export
        results_df[f"{safe_name}_score"] = scores.values
        results_df[f"{safe_name}_is_anomaly"] = anomalies.values

        n_anom = int(anomalies.astype(bool).sum())
        logger.info(
            "  %s: flagged %d / %d (%.2f%%)",
            name,
            n_anom,
            len(X_test),
            n_anom / len(X_test) * 100,
        )

        pm = compute_point_metrics(y_true=y_true, y_pred=anomalies, scores=scores)
        logger.info(
            "  metrics: precision=%.4f  recall=%.4f  F1=%.4f  AUC-ROC=%s  AUC-PR=%s",
            pm.precision,
            pm.recall,
            pm.f1,
            f"{pm.auc_roc:.4f}" if pm.auc_roc is not None else "N/A",
            f"{pm.auc_pr:.4f}" if pm.auc_pr is not None else "N/A",
        )

        if save_checkpoints:
            ckpt_dir = ARTIFACTS_DIR / "checkpoints" / "multivariate" / machine_id
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            if hasattr(model, "save_checkpoint"):
                ckpt_path = ckpt_dir / f"{safe_name}.pt"
                model.save_checkpoint(ckpt_path)
                logger.info("  saved checkpoint → %s", ckpt_path)

    # 4. Export results artifact for dashboard
    artifact_dir = ARTIFACTS_DIR / "multivariate"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_dir / f"{machine_id}_results.csv"
    results_df.to_csv(artifact_path, index=False)
    logger.info("Saved results artifact → %s", artifact_path)
