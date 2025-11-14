from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

KAGGLE_DATASETS = {
    "smd_onmiad": "mgusat/smd-onmiad",
    "nasa_smap_msl": "patrickfleith/nasa-anomaly-detection-dataset-smap-msl",
    "nab": "boltzmannbrain/nab",
}


def ensure_directories() -> None:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
