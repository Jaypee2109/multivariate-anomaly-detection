import argparse
from pathlib import Path

from time_series_transformer.baseline_pipeline import run_pipeline
from time_series_transformer.config import RAW_DATA_DIR

DEFAULT_CSV = RAW_DATA_DIR / "nab" / "realKnownCause" / "realKnownCause" / "nyc_taxi.csv"


def main():
    parser = argparse.ArgumentParser(description="Run baseline anomaly detection pipeline.")
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
        help="Path to input CSV (default: NAB nyc_taxi)",
    )
    parser.add_argument(
        "--mlflow",
        action="store_true",
        help="Enable MLflow experiment tracking",
    )
    args = parser.parse_args()

    run_pipeline(csv_path=args.csv, log_to_mlflow=args.mlflow)


if __name__ == "__main__":
    main()
