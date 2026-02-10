"""Central configuration for time_series_transformer.

Every setting can be overridden via environment variables.
See .env.example for the full list of tunables.
"""

from __future__ import annotations

import os
from pathlib import Path

# Load .env file if python-dotenv is installed (optional)
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Env-var helpers (same pattern as dsai-grp1)
# ---------------------------------------------------------------------------


def _env_int(key: str, default: int) -> int:
    val = os.getenv(key)
    if val is None:
        return default
    try:
        return int(val)
    except ValueError:
        print(f"Warning: invalid int for {key}={val!r}, using default={default}")
        return default


def _env_float(key: str, default: float) -> float:
    val = os.getenv(key)
    if val is None:
        return default
    try:
        return float(val)
    except ValueError:
        print(f"Warning: invalid float for {key}={val!r}, using default={default}")
        return default


def _env_bool(key: str, default: bool) -> bool:
    val = os.getenv(key)
    if val is None:
        return default
    return val.lower() in ("true", "1", "yes", "on")


def _env_str(key: str, default: str) -> str:
    return os.getenv(key, default)


# ---------------------------------------------------------------------------
# Paths (derived from package location — portable across machines)
# ---------------------------------------------------------------------------

PACKAGE_ROOT = Path(__file__).resolve().parent  # .../time_series_transformer
SRC_ROOT = PACKAGE_ROOT.parent  # .../src
PROJECT_ROOT = SRC_ROOT.parent  # .../Transformer

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

KAGGLE_DATASETS = {
    "smd_onmiad": "mgusat/smd-onmiad",
    "nasa_smap_msl": "patrickfleith/nasa-anomaly-detection-dataset-smap-msl",
    "nab": "boltzmannbrain/nab",
}


def ensure_directories() -> None:
    """Create the data directories under the project root if they do not exist."""
    for d in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, ARTIFACTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# General
# ---------------------------------------------------------------------------

TRAIN_RATIO = _env_float("TRAIN_RATIO", 0.7)
RANDOM_STATE = _env_int("RANDOM_STATE", 42)

# ---------------------------------------------------------------------------
# Rolling Z-score
# ---------------------------------------------------------------------------

ROLLING_WINDOW = _env_int("ROLLING_WINDOW", 48)
ROLLING_Z_THRESH = _env_float("ROLLING_Z_THRESH", 1.8)

# ---------------------------------------------------------------------------
# ARIMA
# ---------------------------------------------------------------------------

ARIMA_ORDER = (
    _env_int("ARIMA_P", 2),
    _env_int("ARIMA_D", 1),
    _env_int("ARIMA_Q", 2),
)
ARIMA_Z_THRESH = _env_float("ARIMA_Z_THRESH", 8.5)

# ---------------------------------------------------------------------------
# Isolation Forest
# ---------------------------------------------------------------------------

ISO_CONTAMINATION = _env_float("ISO_CONTAMINATION", 0.004)

# ---------------------------------------------------------------------------
# LSTM baseline
# ---------------------------------------------------------------------------

LSTM_LOOKBACK = _env_int("LSTM_LOOKBACK", 72)
LSTM_HIDDEN_SIZE = _env_int("LSTM_HIDDEN_SIZE", 32)
LSTM_NUM_LAYERS = _env_int("LSTM_NUM_LAYERS", 1)
LSTM_DROPOUT = _env_float("LSTM_DROPOUT", 0.0)
LSTM_EPOCHS = _env_int("LSTM_EPOCHS", 20)
LSTM_BATCH_SIZE = _env_int("LSTM_BATCH_SIZE", 64)
LSTM_LR = _env_float("LSTM_LR", 1e-3)
LSTM_ERROR_QUANTILE = _env_float("LSTM_ERROR_QUANTILE", 0.997)

# ---------------------------------------------------------------------------
# MLflow
# ---------------------------------------------------------------------------

MLFLOW_EXPERIMENT_NAME = _env_str("MLFLOW_EXPERIMENT_NAME", "Anomaly_Detection")
