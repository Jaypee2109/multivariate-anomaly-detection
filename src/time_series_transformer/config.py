from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent  # .../time_series_transformer
SRC_ROOT = PACKAGE_ROOT.parent  # .../src
PROJECT_ROOT = SRC_ROOT.parent  # .../Transformer

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

KAGGLE_DATASETS = {
    "smd_onmiad": "mgusat/smd-onmiad",
    "nasa_smap_msl": "patrickfleith/nasa-anomaly-detection-dataset-smap-msl",
    "nab": "boltzmannbrain/nab",
}


def ensure_directories() -> None:
    """
    Create the data directories under the project root if they do not exist.
    """
    for d in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR]:
        d.mkdir(parents=True, exist_ok=True)


# Train/test split
TRAIN_RATIO = 0.7

# Random seed for reproducibility
RANDOM_STATE = 42

# Rolling Z-score config
ROLLING_WINDOW = 48  # number of points in rolling window
ROLLING_Z_THRESH = 1.8

# ARIMA config (p, d, q)
ARIMA_ORDER = (2, 1, 2)
ARIMA_Z_THRESH = 8.5

# Isolation Forest config
ISO_CONTAMINATION = 0.004  # expected proportion of anomalies

# LSTM baseline config
LSTM_LOOKBACK = 72  # timesteps per window (~1 day)
LSTM_HIDDEN_SIZE = 32
LSTM_NUM_LAYERS = 1
LSTM_DROPOUT = 0.0

LSTM_EPOCHS = 20
LSTM_BATCH_SIZE = 64
LSTM_LR = 1e-3
LSTM_ERROR_QUANTILE = 0.997  # top 1% of errors = anomalies
