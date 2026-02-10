from __future__ import annotations

from typing import Optional, Sequence

from time_series_transformer.config import (
    KAGGLE_DATASETS,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    ensure_directories,
)
from time_series_transformer.data_pipeline.data_download import download_all_datasets
from time_series_transformer.data_pipeline.data_loading import load_dataset
from time_series_transformer.data_pipeline.preprocessing import (
    preprocess_dataset_dict,
    PreprocessingConfig,
)
from time_series_transformer.data_pipeline.data_save import save_processed_dataset


def run_data_pipeline(datasets: Optional[Sequence[str]] = None) -> None:
    """Download and preprocess datasets.

    Args:
        datasets: Optional list of dataset names to process.
                  If None, all datasets from KAGGLE_DATASETS are processed.
    """
    ensure_directories()

    dataset_names = list(datasets) if datasets else list(KAGGLE_DATASETS.keys())

    print("[pipeline] Load / Synchronize datasets ...")
    download_all_datasets()

    for dataset_name in dataset_names:
        if dataset_name not in KAGGLE_DATASETS:
            print(f"[pipeline] Warning: unknown dataset '{dataset_name}', skipping.")
            continue

        print(f"\n[pipeline] === Process dataset: {dataset_name} ===")

        raw_dict = load_dataset(RAW_DATA_DIR, dataset_name)

        cfg = PreprocessingConfig(
            scale_numeric=True,
            use_datetime_index=True,
            exclude_from_scaling=(),
        )

        processed_dict = preprocess_dataset_dict(dataset_name, raw_dict, cfg)
        save_processed_dataset(dataset_name, processed_dict)

    print("\n[pipeline] Finished. Preprocessed data is saved at:", PROCESSED_DATA_DIR)
