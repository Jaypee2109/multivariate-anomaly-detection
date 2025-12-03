import os
from pathlib import Path
from typing import Dict

import pandas as pd

from time_series_transformer.config import RAW_DATA_DIR, ensure_directories
from time_series_transformer.data_pipeline.data_download import download_all_datasets


def load_dataset(directory: Path, name: str) -> Dict[str, pd.DataFrame]:
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
        raise FileNotFoundError(f"Raw data for dataset '{name}' not found: {root}.\n")

    print(f"[load_dataset] Load CSV: {root}")
    data: Dict[str, pd.DataFrame] = {}

    for dirpath, dirnames, filenames in os.walk(root):
        # skip macOS metadata folder
        dirnames[:] = [d for d in dirnames if d != "__MACOSX"]

        for fname in filenames:
            # skip macOS resource fork files
            if fname.startswith("._"):
                print(f"[load_dataset] Skip macOS resource file: {fname}")
                continue

            if not fname.lower().endswith(".csv"):
                continue

            fpath = Path(dirpath) / fname
            rel_path = fpath.relative_to(root).as_posix()

            print(f"[load_dataset] Read: {rel_path}")
            try:
                df = pd.read_csv(fpath)
            except UnicodeDecodeError as e:
                print(f"[load_dataset] UnicodeDecodeError for {rel_path}: {e}")
                print("[load_dataset] -> Skipping this file.")
                continue

            data[rel_path] = df

    print(f"[load_dataset] {len(data)} CSV '{name}' loaded.")
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


def load_all_datasets(dataset_names: list[str]) -> dict[str, Dict[str, pd.DataFrame]]:

    return {name: load_dataset(name) for name in dataset_names}


if __name__ == "__main__":

    download_all_datasets()
    all_data = load_all_datasets(["smd_onmiad", "nasa_smap_msl", "nab"])
    print({k: len(v) for k, v in all_data.items()})
