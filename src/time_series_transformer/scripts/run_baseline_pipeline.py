from time_series_transformer.baseline_pipeline import run_pipeline

from time_series_transformer.config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
)

if __name__ == "__main__":

    csv_path = (
        RAW_DATA_DIR / "nab" / "realKnownCause" / "realKnownCause" / "nyc_taxi.csv"
    )

    run_pipeline(csv_path)
