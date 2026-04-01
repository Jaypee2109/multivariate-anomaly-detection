"""MLflow setup and logging utilities for anomaly detection experiments."""

from __future__ import annotations

import hashlib
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import mlflow

from time_series_transformer.config import MLFLOW_EXPERIMENT_NAME, PROJECT_ROOT
from time_series_transformer.evaluation import PointMetrics, RangeMetrics

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")


def setup_mlflow() -> None:
    """Configure MLflow tracking URI and set the default experiment."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)


def _get_git_sha() -> str | None:
    """Return the current git commit SHA, or None if not in a repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def log_environment_info() -> None:
    """Log Python version, torch version, OS, CUDA, and git SHA as MLflow tags."""
    import torch

    tags: dict[str, str] = {
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "platform": platform.platform(),
        "cuda_available": str(torch.cuda.is_available()),
    }

    git_sha = _get_git_sha()
    if git_sha is not None:
        tags["git_sha"] = git_sha

    mlflow.set_tags(tags)


def log_data_hash(csv_path: Path) -> None:
    """Compute SHA-256 of the input CSV and log as MLflow param."""
    h = hashlib.sha256()
    with open(csv_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    mlflow.log_param("data_hash", h.hexdigest()[:16])


def log_params_from_model(name: str, model: Any) -> None:
    """Introspect model hyperparameters and log as MLflow params.

    Skips private attrs (leading _) and fitted-state attrs (trailing _,
    following the scikit-learn convention).
    """
    params: dict[str, str] = {"model_name": name}
    for attr in vars(model):
        if attr.startswith("_") or attr.endswith("_"):
            continue
        val = getattr(model, attr)
        if val is None or callable(val):
            continue
        # Only log simple scalar types (str, int, float, bool, tuple)
        if not isinstance(val, (str, int, float, bool, tuple)):
            continue
        params[attr] = str(val)
    mlflow.log_params(params)


def log_point_metrics(pm: PointMetrics, prefix: str = "point") -> None:
    """Log PointMetrics fields to the active MLflow run."""
    metrics: dict[str, float] = {
        f"{prefix}/precision": pm.precision,
        f"{prefix}/recall": pm.recall,
        f"{prefix}/f1": pm.f1,
    }
    if pm.auc_roc is not None:
        metrics[f"{prefix}/auc_roc"] = pm.auc_roc
    if pm.auc_pr is not None:
        metrics[f"{prefix}/auc_pr"] = pm.auc_pr
    mlflow.log_metrics(metrics)


def log_range_metrics(rm: RangeMetrics, prefix: str = "range") -> None:
    """Log RangeMetrics fields to the active MLflow run."""
    mlflow.log_metrics(
        {
            f"{prefix}/precision": rm.precision,
            f"{prefix}/recall": rm.recall,
            f"{prefix}/f1": rm.f1,
            f"{prefix}/n_gt_ranges": float(rm.n_gt_ranges),
            f"{prefix}/n_pred_ranges": float(rm.n_pred_ranges),
            f"{prefix}/n_tp_ranges": float(rm.n_tp_ranges),
        }
    )


def log_anomaly_summary(n_total: int, n_anomalies: int) -> None:
    """Log basic anomaly statistics."""
    mlflow.log_metrics(
        {
            "test_size": float(n_total),
            "n_anomalies_flagged": float(n_anomalies),
            "anomaly_rate": n_anomalies / n_total if n_total > 0 else 0.0,
        }
    )
