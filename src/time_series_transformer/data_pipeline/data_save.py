import pandas as pd

from time_series_transformer.config import (
    PROCESSED_DATA_DIR,
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
