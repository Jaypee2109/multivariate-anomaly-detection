"""NAB dataset discovery and loading utilities for the dashboard."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

# Ensure the src package is importable when running from dashboard/
_project_root = Path(__file__).resolve().parents[1]
if str(_project_root / "src") not in sys.path:
    sys.path.insert(0, str(_project_root / "src"))

from time_series_transformer.config import RAW_DATA_DIR  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NAB_ROOT = RAW_DATA_DIR / "nab"
LABELS_DIR = _project_root / "data" / "labels" / "nab"

CATEGORY_DISPLAY_NAMES: dict[str, str] = {
    "realKnownCause": "Real (Known Cause)",
    "realAWSCloudwatch": "Real (AWS Cloudwatch)",
    "realAdExchange": "Real (Ad Exchange)",
    "realTraffic": "Real (Traffic)",
    "realTweets": "Real (Tweets)",
    "artificialWithAnomaly": "Artificial (With Anomaly)",
    "artificialNoAnomaly": "Artificial (No Anomaly)",
}

# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def discover_categories() -> list[dict[str, str]]:
    """Return list of {value, label} dicts for the category dropdown."""
    if not NAB_ROOT.exists():
        return []
    categories = []
    for d in sorted(NAB_ROOT.iterdir()):
        if d.is_dir() and not d.name.startswith((".", "_")):
            display = CATEGORY_DISPLAY_NAMES.get(d.name, d.name)
            categories.append({"value": d.name, "label": display})
    return categories


def discover_datasets(category: str) -> list[dict[str, str]]:
    """Return list of {value, label} dicts for dataset files in a category."""
    cat_dir = NAB_ROOT / category
    if not cat_dir.exists():
        return []
    datasets = []
    # NAB stores CSVs inside a nested same-name subfolder
    for csv_path in sorted(cat_dir.glob("**/*.csv")):
        if csv_path.name.startswith((".", "_")):
            continue
        stem = csv_path.stem
        label = stem.replace("_", " ").title()
        # Use relative path from NAB_ROOT as the value for uniqueness
        rel = csv_path.relative_to(NAB_ROOT).as_posix()
        datasets.append({"value": rel, "label": label})
    return datasets


def load_dataset(rel_path: str) -> pd.DataFrame | None:
    """Load a NAB CSV by its relative path under NAB_ROOT."""
    csv_path = NAB_ROOT / rel_path
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    return df


def load_labels(category: str, filename: str) -> list[str]:
    """Load ground-truth anomaly timestamps for a dataset.

    Returns list of timestamp strings, or empty list if no labels available.
    """
    label_file = LABELS_DIR / f"{category}.json"
    if not label_file.exists():
        return []
    with open(label_file) as f:
        all_labels: dict[str, list[str]] = json.load(f)
    # Labels are keyed like "realKnownCause/nyc_taxi.csv"
    key = f"{category}/{filename}"
    return all_labels.get(key, [])


def get_default_category() -> str:
    """Return the default category to show on page load."""
    return "realKnownCause"


def get_default_dataset(category: str) -> str | None:
    """Return the first dataset in a category, or None."""
    datasets = discover_datasets(category)
    return datasets[0]["value"] if datasets else None
