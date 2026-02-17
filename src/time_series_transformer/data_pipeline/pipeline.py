from __future__ import annotations

import logging
from collections.abc import Sequence

from time_series_transformer.config import (
    KAGGLE_DATASETS,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    SMD_RAW_DIR,
    ensure_directories,
)
from time_series_transformer.data_pipeline.data_download import download_all_datasets
from time_series_transformer.data_pipeline.data_loading import load_dataset
from time_series_transformer.data_pipeline.data_save import save_processed_dataset
from time_series_transformer.data_pipeline.preprocessing import (
    PreprocessingConfig,
    preprocess_dataset_dict,
)
from time_series_transformer.data_pipeline.smd_loading import preprocess_smd

logger = logging.getLogger(__name__)


def run_data_pipeline(datasets: Sequence[str] | None = None) -> None:
    """Download and preprocess datasets.

    Args:
        datasets: Optional list of dataset names to process.
                  If None, all datasets from KAGGLE_DATASETS are processed.
    """
    ensure_directories()

    dataset_names = list(datasets) if datasets else list(KAGGLE_DATASETS.keys())

    logger.info("Loading / synchronizing datasets ...")
    download_all_datasets()

    for dataset_name in dataset_names:
        if dataset_name not in KAGGLE_DATASETS:
            logger.warning("Unknown dataset '%s', skipping.", dataset_name)
            continue

        logger.info("=== Processing dataset: %s ===", dataset_name)

        # SMD uses its own preprocessing (already normalised, just needs
        # column names and CSV conversion)
        if dataset_name == "smd":
            if SMD_RAW_DIR.exists():
                preprocess_smd()
            else:
                logger.warning("SMD raw directory not found: %s", SMD_RAW_DIR)
            continue

        raw_dict = load_dataset(RAW_DATA_DIR, dataset_name)

        cfg = PreprocessingConfig(
            scale_numeric=True,
            use_datetime_index=True,
            exclude_from_scaling=(),
        )

        processed_dict = preprocess_dataset_dict(dataset_name, raw_dict, cfg)
        save_processed_dataset(dataset_name, processed_dict)

    logger.info("Finished. Preprocessed data saved at: %s", PROCESSED_DATA_DIR)
