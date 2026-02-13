import logging
import shutil
from pathlib import Path

import kagglehub

from time_series_transformer.config import (
    KAGGLE_DATASETS,
    RAW_DATA_DIR,
    ensure_directories,
)
from time_series_transformer.exceptions import ConfigurationError

logger = logging.getLogger(__name__)


def download_dataset(name: str) -> Path:

    ensure_directories()

    if name not in KAGGLE_DATASETS:
        raise ConfigurationError(f"Unknown dataset: {name}")

    slug = KAGGLE_DATASETS[name]
    target_dir = RAW_DATA_DIR / name

    if target_dir.exists():
        logger.info("Skip '%s', directory already exists: %s", name, target_dir)
        return target_dir

    logger.info("Downloading kaggle dataset '%s' ...", slug)
    cache_path = Path(kagglehub.dataset_download(slug))
    logger.debug("Cache path: %s", cache_path)

    logger.info("Copying to: %s", target_dir)
    shutil.copytree(cache_path, target_dir)

    logger.info("Finished: %s", target_dir)
    return target_dir


def download_all_datasets() -> dict:

    paths = {}
    for name in KAGGLE_DATASETS:
        paths[name] = download_dataset(name)
    return paths


if __name__ == "__main__":
    download_all_datasets()
