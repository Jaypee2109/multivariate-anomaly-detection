"""Central configuration for time_series_transformer.

Every setting can be overridden via environment variables.
See .env.example for the full list of tunables.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

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
        logger.warning("Invalid int for %s=%r, using default=%s", key, val, default)
        return default


def _env_float(key: str, default: float) -> float:
    val = os.getenv(key)
    if val is None:
        return default
    try:
        return float(val)
    except ValueError:
        logger.warning("Invalid float for %s=%r, using default=%s", key, val, default)
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
# Env-var overrides allow Docker containers to use mounted volume paths.
# ---------------------------------------------------------------------------

PACKAGE_ROOT = Path(__file__).resolve().parent  # .../time_series_transformer
SRC_ROOT = PACKAGE_ROOT.parent  # .../src
PROJECT_ROOT = SRC_ROOT.parent  # .../Transformer

DATA_DIR = Path(os.environ["DATA_DIR"]) if "DATA_DIR" in os.environ else PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = (
    Path(os.environ["ARTIFACTS_DIR"])
    if "ARTIFACTS_DIR" in os.environ
    else PROJECT_ROOT / "artifacts"
)

KAGGLE_DATASETS = {
    "smd": "mgusat/smd-onmiad",
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
# VAR (multivariate)
# ---------------------------------------------------------------------------

VAR_MAXLAGS = _env_int("VAR_MAXLAGS", 5)
VAR_IC = _env_str("VAR_IC", "aic")
VAR_Z_THRESH = _env_float("VAR_Z_THRESH", 3.0)
VAR_AGGREGATION = _env_str("VAR_AGGREGATION", "max")

# ---------------------------------------------------------------------------
# Multivariate Isolation Forest
# ---------------------------------------------------------------------------

MULTI_ISO_CONTAMINATION = _env_float("MULTI_ISO_CONTAMINATION", 0.01)

# ---------------------------------------------------------------------------
# LSTM Autoencoder (multivariate)
# ---------------------------------------------------------------------------

LSTM_AE_LOOKBACK = _env_int("LSTM_AE_LOOKBACK", 30)
LSTM_AE_HIDDEN_SIZE = _env_int("LSTM_AE_HIDDEN_SIZE", 64)
LSTM_AE_LATENT_DIM = _env_int("LSTM_AE_LATENT_DIM", 32)
LSTM_AE_NUM_LAYERS = _env_int("LSTM_AE_NUM_LAYERS", 1)
LSTM_AE_DROPOUT = _env_float("LSTM_AE_DROPOUT", 0.0)
LSTM_AE_EPOCHS = _env_int("LSTM_AE_EPOCHS", 30)
LSTM_AE_BATCH_SIZE = _env_int("LSTM_AE_BATCH_SIZE", 64)
LSTM_AE_LR = _env_float("LSTM_AE_LR", 1e-3)
LSTM_AE_ERROR_QUANTILE = _env_float("LSTM_AE_ERROR_QUANTILE", 0.99)
LSTM_AE_SCORE_METRIC = _env_str("LSTM_AE_SCORE_METRIC", "mse")

# ---------------------------------------------------------------------------
# LSTM Forecaster (multivariate)
# ---------------------------------------------------------------------------

LSTM_FC_LOOKBACK = _env_int("LSTM_FC_LOOKBACK", 30)
LSTM_FC_HIDDEN_SIZE = _env_int("LSTM_FC_HIDDEN_SIZE", 64)
LSTM_FC_NUM_LAYERS = _env_int("LSTM_FC_NUM_LAYERS", 1)
LSTM_FC_DROPOUT = _env_float("LSTM_FC_DROPOUT", 0.0)
LSTM_FC_EPOCHS = _env_int("LSTM_FC_EPOCHS", 30)
LSTM_FC_BATCH_SIZE = _env_int("LSTM_FC_BATCH_SIZE", 64)
LSTM_FC_LR = _env_float("LSTM_FC_LR", 1e-3)
LSTM_FC_ERROR_QUANTILE = _env_float("LSTM_FC_ERROR_QUANTILE", 0.97)
LSTM_FC_SCORE_METRIC = _env_str("LSTM_FC_SCORE_METRIC", "mse")

# ---------------------------------------------------------------------------
# TranAD (multivariate)
# ---------------------------------------------------------------------------

TRANAD_LOOKBACK = _env_int("TRANAD_LOOKBACK", 30)
TRANAD_N_HEADS = _env_int("TRANAD_N_HEADS", 0)  # 0 = auto (n_features)
TRANAD_DIM_FF = _env_int("TRANAD_DIM_FF", 16)
TRANAD_NUM_LAYERS = _env_int("TRANAD_NUM_LAYERS", 1)
TRANAD_DROPOUT = _env_float("TRANAD_DROPOUT", 0.1)
TRANAD_EPOCHS = _env_int("TRANAD_EPOCHS", 15)
TRANAD_BATCH_SIZE = _env_int("TRANAD_BATCH_SIZE", 128)
TRANAD_LR = _env_float("TRANAD_LR", 1e-4)
TRANAD_ERROR_QUANTILE = _env_float("TRANAD_ERROR_QUANTILE", 0.99)
TRANAD_SCORE_METRIC = _env_str("TRANAD_SCORE_METRIC", "mse")

# ---------------------------------------------------------------------------
# Custom Transformer (multivariate) — adapted from Lars's Time2Vec architecture
# ---------------------------------------------------------------------------

CUSTOM_TF_LOOKBACK = _env_int("CUSTOM_TF_LOOKBACK", 30)
CUSTOM_TF_T2V_DIM = _env_int("CUSTOM_TF_T2V_DIM", 16)
CUSTOM_TF_MODEL_DIM = _env_int("CUSTOM_TF_MODEL_DIM", 64)
CUSTOM_TF_NUM_HEADS = _env_int("CUSTOM_TF_NUM_HEADS", 4)
CUSTOM_TF_NUM_LAYERS = _env_int("CUSTOM_TF_NUM_LAYERS", 2)
CUSTOM_TF_DIM_FF = _env_int("CUSTOM_TF_DIM_FF", 256)
CUSTOM_TF_DROPOUT = _env_float("CUSTOM_TF_DROPOUT", 0.1)
CUSTOM_TF_EPOCHS = _env_int("CUSTOM_TF_EPOCHS", 15)
CUSTOM_TF_BATCH_SIZE = _env_int("CUSTOM_TF_BATCH_SIZE", 64)
CUSTOM_TF_LR = _env_float("CUSTOM_TF_LR", 1e-3)
CUSTOM_TF_ERROR_QUANTILE = _env_float("CUSTOM_TF_ERROR_QUANTILE", 0.99)
CUSTOM_TF_SCORE_METRIC = _env_str("CUSTOM_TF_SCORE_METRIC", "mse")

# ---------------------------------------------------------------------------
# SMD dataset
# ---------------------------------------------------------------------------

SMD_RAW_DIR = RAW_DATA_DIR / "smd_onmiad" / "ServerMachineDataset"
SMD_BASE_DIR = PROCESSED_DATA_DIR / "smd"

# ---------------------------------------------------------------------------
# MLflow
# ---------------------------------------------------------------------------

MLFLOW_EXPERIMENT_NAME = _env_str("MLFLOW_EXPERIMENT_NAME", "Anomaly_Detection")

# ---------------------------------------------------------------------------
# API Server
# ---------------------------------------------------------------------------

API_HOST = _env_str("API_HOST", "127.0.0.1")
API_PORT = _env_int("API_PORT", 8000)
