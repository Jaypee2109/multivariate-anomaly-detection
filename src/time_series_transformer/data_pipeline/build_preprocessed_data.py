from pathlib import Path
import pandas as pd

from time_series_transformer.config import (
    KAGGLE_DATASETS,
    PROCESSED_DATA_DIR,
    ensure_directories,
)
from time_series_transformer.data_pipeline.data_download import download_all_datasets
from time_series_transformer.data_pipeline.data_loading import load_dataset
from time_series_transformer.data_pipeline.preprocessing import (
    preprocess_dataset_dict,
    PreprocessingConfig,
)


def save_processed_dataset(
    dataset_name: str,
    processed: dict[str, "pd.DataFrame"],
) -> None:

    base = PROCESSED_DATA_DIR / dataset_name
    base.mkdir(parents=True, exist_ok=True)

    for rel_path, df in processed.items():
        out_path = base / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"[save_processed_dataset] Save: {out_path}")
        df.to_csv(
            out_path, index=True if isinstance(df.index, pd.DatetimeIndex) else False
        )


def main():
    ensure_directories()

    # 1) Download data
    print("[main] Load / Synchronize datasets ...")
    download_all_datasets()

    # 2) Load, preprocess, save
    for dataset_name in KAGGLE_DATASETS.keys():
        print(f"\n[main] === Process dataset: {dataset_name} ===")

        raw_dict = load_dataset(dataset_name)

        # Configuration
        cfg = PreprocessingConfig(
            scale_numeric=True,
            use_datetime_index=True,
            exclude_from_scaling=(),
        )

        processed_dict = preprocess_dataset_dict(dataset_name, raw_dict, cfg)
        save_processed_dataset(dataset_name, processed_dict)

    print("\n[main] Finished. Preprocessed data is saved at:", PROCESSED_DATA_DIR)


if __name__ == "__main__":
    main()
