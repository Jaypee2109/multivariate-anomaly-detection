from time_series_transformer.analysis.eda import run_basic_eda_from_csv
from time_series_transformer.config import (
    RAW_DATA_DIR,
)

if __name__ == "__main__":
    csv_path = RAW_DATA_DIR / "nab" / "realKnownCause" / "realKnownCause" / "nyc_taxi.csv"

    run_basic_eda_from_csv(csv_path, timestamp_col="timestamp", value_col="value")
