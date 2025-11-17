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
