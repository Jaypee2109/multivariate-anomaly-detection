import logging
import os
from pathlib import Path

import pandas as pd

from time_series_transformer.config import ensure_directories
from time_series_transformer.data_pipeline.data_download import download_all_datasets
from time_series_transformer.exceptions import DataNotFoundError

logger = logging.getLogger(__name__)


def load_dataset(directory: Path, name: str) -> dict[str, pd.DataFrame]:
    """
    Recursively loads all CSV files under data/raw/<name>/ into a dict:

        { "relative/path/to/file.csv": DataFrame }

    Ignores:
      - macOS metadata folders (__MACOSX)
      - files starting with "._" (resource forks)
    """
    ensure_directories()
    root = directory / name
    if not root.exists():
        raise DataNotFoundError(f"Raw data for dataset '{name}' not found: {root}.\n")

    logger.info("Loading CSVs from: %s", root)
    data: dict[str, pd.DataFrame] = {}

    for dirpath, dirnames, filenames in os.walk(root):
        # skip macOS metadata folder
        dirnames[:] = [d for d in dirnames if d != "__MACOSX"]

        for fname in filenames:
            # skip macOS resource fork files
            if fname.startswith("._"):
                logger.debug("Skip macOS resource file: %s", fname)
                continue

            if not fname.lower().endswith(".csv"):
                continue

            fpath = Path(dirpath) / fname
            rel_path = fpath.relative_to(root).as_posix()

            logger.debug("Reading: %s", rel_path)
            try:
                df = pd.read_csv(fpath)
            except UnicodeDecodeError as e:
                logger.warning("UnicodeDecodeError for %s: %s — skipping", rel_path, e)
                continue

            data[rel_path] = df

    logger.info("Loaded %d CSVs for '%s'.", len(data), name)
    return data


def load_timeseries(path: Path, value_col: str = "value") -> pd.Series:
    """
    Load a CSV time series with columns:
        timestamp,value

    Returns a pandas Series indexed by timestamp.
    """
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.set_index("timestamp").sort_index()
    return df[value_col]


def load_all_datasets(dataset_names: list[str]) -> dict[str, dict[str, pd.DataFrame]]:

    return {name: load_dataset(name) for name in dataset_names}


if __name__ == "__main__":
    download_all_datasets()
    all_data = load_all_datasets(["smd_onmiad", "nasa_smap_msl", "nab"])
    print({k: len(v) for k, v in all_data.items()})
