"""MLflow data loading utilities for the dashboard."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px

# Ensure the src package is importable when running from dashboard/
_project_root = Path(__file__).resolve().parents[1]
if str(_project_root / "src") not in sys.path:
    sys.path.insert(0, str(_project_root / "src"))

from time_series_transformer.config import ARTIFACTS_DIR, MLFLOW_EXPERIMENT_NAME  # noqa: E402
from time_series_transformer.mlflow_utils import MLFLOW_TRACKING_URI  # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MLflow run queries
# ---------------------------------------------------------------------------

ARTIFACTS_CSV = ARTIFACTS_DIR / "anomalies" / "baseline_anomalies.csv"


def load_mlflow_runs(
    experiment_name: str = MLFLOW_EXPERIMENT_NAME,
) -> pd.DataFrame | None:
    """Load MLflow runs, keeping the latest run per model.

    Returns a DataFrame with columns like:
      run_id, model_name, metrics.point/f1, params.model_name, ...
    or None if MLflow is unavailable / no runs exist.
    """
    try:
        import mlflow

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        runs = mlflow.search_runs(experiment_names=[experiment_name])
    except Exception:
        logger.warning("Failed to load MLflow runs", exc_info=True)
        return None

    if runs.empty:
        return None

    # Only keep completed runs that have a model_name param
    if "params.model_name" not in runs.columns:
        return None
    runs = runs[runs["params.model_name"].notna()].copy()
    if runs.empty:
        return None

    # Deduplicate: keep the latest run per model
    runs = runs.sort_values("start_time", ascending=False)
    runs = runs.drop_duplicates(subset=["params.model_name"], keep="first")

    # Add a convenience column
    runs["model_name"] = runs["params.model_name"]

    return runs.reset_index(drop=True)


def load_data_run_params(
    experiment_name: str = MLFLOW_EXPERIMENT_NAME,
) -> dict[str, str]:
    """Load shared dataset params (data_hash, dataset, etc.).

    These may live on a separate data-only run (older pipeline) or on each
    model run (current pipeline).  Returns the latest available values.
    """
    try:
        import mlflow

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        runs = mlflow.search_runs(experiment_names=[experiment_name])
    except Exception:
        return {}

    if runs.empty or "params.dataset" not in runs.columns:
        return {}

    data_runs = runs[runs["params.dataset"].notna()].copy()
    if data_runs.empty:
        return {}

    row = data_runs.sort_values("start_time", ascending=False).iloc[0]
    keys = [
        "params.dataset",
        "params.data_hash",
        "params.train_ratio",
        "params.train_size",
        "params.test_size",
    ]
    return {k: str(row[k]) for k in keys if k in row.index and pd.notna(row[k])}


# ---------------------------------------------------------------------------
# Artifacts CSV
# ---------------------------------------------------------------------------


def load_artifacts_csv(path: Path = ARTIFACTS_CSV) -> pd.DataFrame | None:
    """Load the baseline anomalies CSV artifact."""
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, parse_dates=["timestamp"])
        return df
    except Exception:
        logger.warning("Failed to load artifacts CSV: %s", path, exc_info=True)
        return None


def discover_models_from_artifacts(df: pd.DataFrame) -> list[str]:
    """Extract model names from artifact CSV column suffixes.

    Looks for columns matching ``*_score`` and derives model names.
    """
    models = []
    for col in df.columns:
        if col.endswith("_score"):
            name = col[: -len("_score")]
            # Verify the companion _is_anomaly column also exists
            if f"{name}_is_anomaly" in df.columns:
                models.append(name)
    return sorted(models)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def build_color_map(models: list[str]) -> dict[str, str]:
    """Stable color assignment: sorted alphabetically -> Plotly palette."""
    palette = px.colors.qualitative.Plotly
    return {m: palette[i % len(palette)] for i, m in enumerate(sorted(models))}


def enforce_min_one(selected: list[str], fallback: list[str]) -> list[str]:
    """Ensure at least one item is selected."""
    if not selected and fallback:
        return [fallback[0]]
    return selected
