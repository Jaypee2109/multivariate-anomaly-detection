"""SMD dataset discovery and loading utilities for the dashboard."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# Ensure the src package is importable when running from dashboard/
_project_root = Path(__file__).resolve().parents[1]
if str(_project_root / "src") not in sys.path:
    sys.path.insert(0, str(_project_root / "src"))

import plotly.express as px
import plotly.graph_objects as go

from time_series_transformer.config import ARTIFACTS_DIR, SMD_BASE_DIR  # noqa: E402
from time_series_transformer.data_pipeline.smd_loading import SMD_COLUMN_NAMES  # noqa: E402

# Columns that are NOT features in artifact CSVs
_NON_FEATURE_SUFFIXES = ("_score", "_is_anomaly")
_NON_FEATURE_NAMES = {"is_anomaly"}

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MULTIVARIATE_RESULTS_DIR = ARTIFACTS_DIR / "multivariate"

# ---------------------------------------------------------------------------
# SMD result discovery (from pipeline artifact CSVs)
# ---------------------------------------------------------------------------


def discover_smd_results() -> list[dict[str, str]]:
    """Return list of {value, label} dicts for machines with result artifacts."""
    if not MULTIVARIATE_RESULTS_DIR.exists():
        return []
    results = []
    for csv_path in sorted(MULTIVARIATE_RESULTS_DIR.glob("*_results.csv")):
        machine_id = csv_path.stem.replace("_results", "")
        results.append({"value": machine_id, "label": machine_id})
    return results


def load_smd_results(machine_id: str) -> pd.DataFrame | None:
    """Load the artifact CSV for a given machine."""
    path = MULTIVARIATE_RESULTS_DIR / f"{machine_id}_results.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def discover_smd_models(df: pd.DataFrame) -> list[str]:
    """Extract model names from artifact CSV column naming pattern.

    Looks for ``{model}_score`` columns that also have ``{model}_is_anomaly``.
    """
    models = []
    for col in df.columns:
        if col.endswith("_score"):
            name = col[: -len("_score")]
            if f"{name}_is_anomaly" in df.columns:
                models.append(name)
    return sorted(models)


def discover_smd_features(df: pd.DataFrame) -> list[str]:
    """Return feature column names from the results DataFrame.

    Recognises both real column names (cpu_r, load_1, ...) and legacy
    generic names (f0, f1, ...).
    """
    smd_set = set(SMD_COLUMN_NAMES)
    features = [
        c for c in df.columns
        if c in smd_set
        or (c.startswith("f") and c[1:].isdigit())
    ]
    return features


def get_default_machine() -> str | None:
    """Return the first available machine, or None."""
    results = discover_smd_results()
    return results[0]["value"] if results else None


# ---------------------------------------------------------------------------
# Raw SMD data loading (for data analysis page)
# ---------------------------------------------------------------------------


def load_smd_train_test(
    machine_id: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series] | None:
    """Load raw SMD train/test/labels for EDA.

    Returns (train_df, test_df, test_labels) or None if data not available.
    """
    try:
        from time_series_transformer.data_pipeline.smd_loading import load_smd_machine

        data = load_smd_machine(machine_id, base_dir=SMD_BASE_DIR)
        return data.train_df, data.test_df, data.test_labels
    except Exception:
        return None


def list_smd_machines() -> list[dict[str, str]]:
    """Return available SMD machines from the raw dataset directory."""
    try:
        from time_series_transformer.data_pipeline.smd_loading import (
            list_smd_machines as _list,
        )

        machines = _list(SMD_BASE_DIR)
        return [{"value": m, "label": m} for m in machines]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def is_feature_column(col: str) -> bool:
    """Return True if *col* is an SMD feature (real or legacy name)."""
    if col in _NON_FEATURE_NAMES:
        return False
    if any(col.endswith(s) for s in _NON_FEATURE_SUFFIXES):
        return False
    if col in set(SMD_COLUMN_NAMES):
        return True
    if col.startswith("f") and col[1:].isdigit():
        return True
    return False


def build_color_map(models: list[str]) -> dict[str, str]:
    """Stable color assignment: sorted alphabetically -> Plotly palette."""
    palette = px.colors.qualitative.Plotly
    return {m: palette[i % len(palette)] for i, m in enumerate(sorted(models))}


def enforce_min_one(selected: list[str], fallback: list[str]) -> list[str]:
    """Ensure at least one item is selected."""
    if not selected and fallback:
        return [fallback[0]]
    return selected


def anomaly_ranges(mask) -> list[tuple[int, int]]:
    """Convert a boolean anomaly mask to a list of (start, end) index ranges.

    Each range is inclusive on both ends so it can be passed directly to
    ``fig.add_vrect(x0=start, x1=end, ...)``.
    """
    ranges: list[tuple[int, int]] = []
    in_range = False
    start = 0
    for i, v in enumerate(mask):
        if v and not in_range:
            start = i
            in_range = True
        elif not v and in_range:
            ranges.append((start, i - 1))
            in_range = False
    if in_range:
        ranges.append((start, len(mask) - 1))
    return ranges


def add_anomaly_zones(
    fig,
    mask,
    *,
    color: str = "#ff5252",
    opacity: float = 0.15,
    label: str = "Anomaly (GT)",
    row: int | None = None,
    col: int | None = None,
) -> None:
    """Add toggleable anomaly-zone polygons to *fig*.

    Instead of layout shapes (which ignore legend clicks), this builds a
    single ``go.Scatter`` trace with ``fill='toself'`` so the zones can
    be shown/hidden via the legend like any other trace.

    The y-extent is derived from the traces already present in *fig*.
    """
    ranges = anomaly_ranges(mask)
    if not ranges:
        return

    # Derive y-extent from existing traces so zones span the full plot
    all_y: list[float] = []
    for trace in fig.data:
        if hasattr(trace, "y") and trace.y is not None:
            for v in trace.y:
                if isinstance(v, (int, float)) and v == v:  # skip NaN
                    all_y.append(v)
    if all_y:
        y_lo, y_hi = min(all_y), max(all_y)
    else:
        y_lo, y_hi = 0, 1

    # Convert hex colour to rgba fill string
    if color.startswith("#") and len(color) == 7:
        r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
    else:
        r, g, b = 255, 82, 82
    fill_rgba = f"rgba({r}, {g}, {b}, {opacity})"

    # Build a single polygon trace (None-separated rectangles)
    x_poly: list[float | None] = []
    y_poly: list[float | None] = []
    for x0, x1 in ranges:
        x_poly.extend([x0 - 0.5, x0 - 0.5, x1 + 0.5, x1 + 0.5, x0 - 0.5, None])
        y_poly.extend([y_lo, y_hi, y_hi, y_lo, y_lo, None])

    trace_kw = {}
    if row is not None and col is not None:
        trace_kw = {"row": row, "col": col}

    fig.add_trace(
        go.Scatter(
            x=x_poly,
            y=y_poly,
            fill="toself",
            fillcolor=fill_rgba,
            line=dict(width=0),
            name=label,
            showlegend=True,
            hoverinfo="skip",
        ),
        **trace_kw,
    )
