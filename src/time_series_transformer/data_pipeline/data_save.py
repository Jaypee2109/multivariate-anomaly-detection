import logging

import pandas as pd

from time_series_transformer.config import (
    PROCESSED_DATA_DIR,
)

logger = logging.getLogger(__name__)


def save_processed_dataset(
    dataset_name: str,
    processed: dict[str, "pd.DataFrame"],
) -> None:

    base = PROCESSED_DATA_DIR / dataset_name
    base.mkdir(parents=True, exist_ok=True)

    for rel_path, df in processed.items():
        out_path = base / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Saving: %s", out_path)
        df.to_csv(out_path, index=isinstance(df.index, pd.DatetimeIndex))
