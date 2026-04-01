"""Training pipeline for multivariate anomaly detection on SMD data."""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from time_series_transformer.config import (
    ARTIFACTS_DIR,
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
    MULTI_ISO_CONTAMINATION,
    RANDOM_STATE,
    SMD_BASE_DIR,
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
from time_series_transformer.data_pipeline.smd_loading import load_smd_machine
from time_series_transformer.evaluation import (
    compute_best_f1,
    compute_detection_latency,
    compute_point_adjust_metrics,
    compute_point_metrics,
)
from time_series_transformer.models.multivariate.custom_transformer import (
    CustomTransformerDetector,
)
from time_series_transformer.models.multivariate.isolation_forest import (
    MultivariateIsolationForestDetector,
)
from time_series_transformer.models.multivariate.lstm_autoencoder import (
    LSTMAutoencoderAnomalyDetector,
)
from time_series_transformer.models.multivariate.tranad import TranADAnomalyDetector

logger = logging.getLogger(__name__)

MULTIVARIATE_MODEL_REGISTRY: dict[str, str] = {
    "multi_isolation_forest": "Isolation Forest (MV)",
    "lstm_autoencoder": "LSTM Autoencoder",
    "tranad": "TranAD",
    "custom_transformer": "Custom Transformer (T2V)",
    # optional (use --model var / --model lstm_forecaster)
    "var": "VAR Residual",
    "lstm_forecaster": "LSTM Forecaster (MV)",
}

# Models built by default (without --model filter)
_DEFAULT_MODELS = {"multi_isolation_forest", "lstm_autoencoder", "tranad", "custom_transformer"}


def _seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_model(key: str):
    """Lazily build a single model by registry key."""
    if key == "multi_isolation_forest":
        return MultivariateIsolationForestDetector(
            contamination=MULTI_ISO_CONTAMINATION,
            random_state=RANDOM_STATE,
        )
    if key == "lstm_autoencoder":
        return LSTMAutoencoderAnomalyDetector(
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
        )
    if key == "tranad":
        return TranADAnomalyDetector(
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
        )
    if key == "custom_transformer":
        return CustomTransformerDetector(
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
        )
    if key == "var":
        from time_series_transformer.config import (
            VAR_AGGREGATION,
            VAR_IC,
            VAR_MAXLAGS,
            VAR_Z_THRESH,
        )
        from time_series_transformer.models.multivariate.var import (
            VARResidualAnomalyDetector,
        )

        return VARResidualAnomalyDetector(
            maxlags=VAR_MAXLAGS,
            ic=VAR_IC,
            z_thresh=VAR_Z_THRESH,
            aggregation=VAR_AGGREGATION,
        )
    if key == "lstm_forecaster":
        from time_series_transformer.config import (
            LSTM_FC_BATCH_SIZE,
            LSTM_FC_DROPOUT,
            LSTM_FC_EPOCHS,
            LSTM_FC_ERROR_QUANTILE,
            LSTM_FC_HIDDEN_SIZE,
            LSTM_FC_LOOKBACK,
            LSTM_FC_LR,
            LSTM_FC_NUM_LAYERS,
            LSTM_FC_SCORE_METRIC,
        )
        from time_series_transformer.models.multivariate.lstm_forecaster import (
            LSTMForecasterMultivariateDetector,
        )

        return LSTMForecasterMultivariateDetector(
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
        )
    raise ValueError(f"Unknown model key: {key!r}")


def _build_multivariate_models(keys: set[str] | None = None) -> dict:
    """Build models for the given keys (default: _DEFAULT_MODELS)."""
    if keys is None:
        keys = _DEFAULT_MODELS
    return {
        MULTIVARIATE_MODEL_REGISTRY[k]: _build_model(k)
        for k in keys
        if k in MULTIVARIATE_MODEL_REGISTRY
    }


def run_multivariate_pipeline(
    machine_id: str,
    base_dir: Path | None = None,
    model_names: Sequence[str] | None = None,
    save_checkpoints: bool = False,
    log_to_mlflow: bool = False,
) -> None:
    """Train and evaluate multivariate anomaly detectors on a single SMD machine."""
    _seed_everything(RANDOM_STATE)

    # MLflow setup
    mlflow_mod = None
    if log_to_mlflow:
        try:
            import mlflow

            from time_series_transformer.mlflow_utils import (
                log_anomaly_summary,
                log_environment_info,
                setup_mlflow,
            )

            mlflow_mod = mlflow
            setup_mlflow()
        except ImportError:
            logger.warning("mlflow not installed, skipping tracking.")
            log_to_mlflow = False

    if base_dir is None:
        base_dir = SMD_BASE_DIR

    # 1. Load data (pre-split, no normalization — SMD is already in [0,1])
    machine_data = load_smd_machine(machine_id, base_dir=base_dir, normalize=False)
    X_train = machine_data.train_df
    X_test = machine_data.test_df
    y_true = machine_data.test_labels

    # 2. Build models (optionally filtered)
    if model_names is not None:
        keys = {n for n in model_names if n in MULTIVARIATE_MODEL_REGISTRY}
        if not keys:
            logger.warning(
                "No valid models selected. Available: %s",
                list(MULTIVARIATE_MODEL_REGISTRY.keys()),
            )
            return
    else:
        keys = None  # uses _DEFAULT_MODELS
    models = _build_multivariate_models(keys)

    # 3. Train and evaluate
    results_df = X_test.copy()
    results_df["is_anomaly"] = y_true.values

    for name, model in models.items():
        logger.info("Training %s on %s …", name, machine_id)
        safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "")

        run_ctx = (
            mlflow_mod.start_run(run_name=f"{name} — {machine_id}")
            if log_to_mlflow
            else nullcontext()
        )

        with run_ctx:
            if log_to_mlflow:
                log_environment_info()
                mlflow_mod.log_params({
                    "model": name,
                    "machine_id": machine_id,
                    "pipeline": "multivariate",
                })

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

            # Point-level metrics
            pm = compute_point_metrics(y_true=y_true, y_pred=anomalies, scores=scores)
            logger.info(
                "  point:    P=%.4f  R=%.4f  F1=%.4f  AUC-ROC=%s  AUC-PR=%s",
                pm.precision,
                pm.recall,
                pm.f1,
                f"{pm.auc_roc:.4f}" if pm.auc_roc is not None else "N/A",
                f"{pm.auc_pr:.4f}" if pm.auc_pr is not None else "N/A",
            )

            # Point-adjust metrics (TranAD / OmniAnomaly protocol)
            pa = compute_point_adjust_metrics(y_true=y_true, y_pred=anomalies)
            logger.info(
                "  PA:       P=%.4f  R=%.4f  F1=%.4f",
                pa.precision,
                pa.recall,
                pa.f1,
            )

            # Best-F1 threshold search
            bf = compute_best_f1(y_true=y_true, scores=scores)
            logger.info(
                "  best-F1:  F1=%.4f (thr=%.4g)  PA-F1=%.4f",
                bf.f1,
                bf.threshold,
                bf.pa_f1,
            )

            # Detection latency
            dl = compute_detection_latency(y_true=y_true, y_pred=anomalies)
            logger.info(
                "  latency:  mean=%.1f  median=%.1f  detected=%d/%d  missed=%d",
                dl.mean_latency,
                dl.median_latency,
                dl.n_detected,
                dl.n_segments,
                dl.n_missed,
            )

            if log_to_mlflow:
                mlflow_mod.log_metric("fit_time_seconds", fit_time)
                log_anomaly_summary(len(y_true), n_anom)
                mlflow_mod.log_metrics({
                    "point_precision": pm.precision,
                    "point_recall": pm.recall,
                    "point_f1": pm.f1,
                    "pa_precision": pa.precision,
                    "pa_recall": pa.recall,
                    "pa_f1": pa.f1,
                    "best_f1": bf.f1,
                    "best_f1_threshold": bf.threshold,
                    "best_f1_pa_f1": bf.pa_f1,
                    "detection_latency_mean": dl.mean_latency,
                    "detection_latency_median": dl.median_latency,
                })
                if pm.auc_roc is not None:
                    mlflow_mod.log_metric("auc_roc", pm.auc_roc)
                if pm.auc_pr is not None:
                    mlflow_mod.log_metric("auc_pr", pm.auc_pr)

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
