from time_series_transformer.analysis.eda import run_eda_pipeline

from time_series_transformer.config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
)


if __name__ == "__main__":

    csv_path = (
        RAW_DATA_DIR / "nab" / "realTweets" / "realTweets" / "Twitter_volume_GOOG.csv"
    )

    run_eda_pipeline(csv_path, timestamp_col="timestamp", value_col="value")
