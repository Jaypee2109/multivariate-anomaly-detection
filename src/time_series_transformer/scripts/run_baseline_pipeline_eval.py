import argparse
from pathlib import Path

from time_series_transformer.data_pipeline.preprocessing import load_csv_to_df
from time_series_transformer.data_pipeline.labels import (
    load_label_times,
    make_point_labels_from_times,
)
from time_series_transformer.split import train_test_split_series
from time_series_transformer.baseline_pipeline import run_pipeline
from time_series_transformer.config import (
    DATA_DIR,
    RAW_DATA_DIR,
    TRAIN_RATIO,
)


DEFAULT_CSV = (
    RAW_DATA_DIR / "nab" / "realKnownCause" / "realKnownCause" / "nyc_taxi.csv"
)
DEFAULT_LABELS = DATA_DIR / "labels" / "nab" / "realKnownCause.json"
DEFAULT_LABELS_KEY = "realKnownCause/nyc_taxi.csv"


def main():
    parser = argparse.ArgumentParser(
        description="Run baseline pipeline with ground-truth evaluation."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
        help="Path to input CSV (default: NAB nyc_taxi)",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=DEFAULT_LABELS,
        help="Path to labels JSON (default: NAB realKnownCause.json)",
    )
    parser.add_argument(
        "--labels-key",
        type=str,
        default=DEFAULT_LABELS_KEY,
        help="Key inside the labels JSON for this file",
    )
    parser.add_argument(
        "--mlflow",
        action="store_true",
        help="Enable MLflow experiment tracking",
    )
    args = parser.parse_args()

    # 1) Load df and put timestamp on the index
    df = load_csv_to_df(args.csv, parse_dates=["timestamp"])
    df = df.set_index("timestamp").sort_index()
    y = df["value"]

    # 2) Load label times & make pointwise labels with DatetimeIndex
    label_times = load_label_times(args.labels, args.labels_key)
    df_for_labels = df.reset_index()
    y_true_all = make_point_labels_from_times(
        df_for_labels, label_times, timestamp_col="timestamp"
    )

    # Align index with y (DatetimeIndex)
    y_true_all.index = y.index

    # 3) Split labels to match test set
    _, y_true_test = train_test_split_series(y_true_all, train_ratio=TRAIN_RATIO)

    # 4) Run pipeline with labels and optional MLflow
    run_pipeline(
        csv_path=args.csv,
        y_true_labels=y_true_test,
        log_to_mlflow=args.mlflow,
    )


if __name__ == "__main__":
    main()
