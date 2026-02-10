"""train command — train anomaly detection models."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from time_series_transformer.config import DATA_DIR, RAW_DATA_DIR, TRAIN_RATIO

logger = logging.getLogger(__name__)

DEFAULT_CSV = RAW_DATA_DIR / "nab" / "realKnownCause" / "realKnownCause" / "nyc_taxi.csv"
DEFAULT_LABELS = DATA_DIR / "labels" / "nab" / "realKnownCause.json"
DEFAULT_LABELS_KEY = "realKnownCause/nyc_taxi.csv"


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "train",
        help="Train anomaly detection models",
        description="Train baseline anomaly detectors on a time series dataset.",
    )

    # Data arguments
    data_group = parser.add_argument_group("Data")
    data_group.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
        help="Path to input CSV (default: NAB nyc_taxi)",
    )

    # Ground-truth evaluation
    eval_group = parser.add_argument_group("Evaluation (optional)")
    eval_group.add_argument(
        "--labels",
        type=Path,
        default=None,
        help="Path to ground-truth labels JSON",
    )
    eval_group.add_argument(
        "--labels-key",
        type=str,
        default=DEFAULT_LABELS_KEY,
        help="Key inside the labels JSON for this file",
    )

    # Tracking
    parser.add_argument(
        "--mlflow",
        action="store_true",
        help="Enable MLflow experiment tracking",
    )


def run(args: argparse.Namespace) -> None:
    from time_series_transformer.baseline_pipeline import run_pipeline

    # Validate CSV exists
    if not args.csv.exists():
        logger.error("CSV file not found: %s", args.csv)
        logger.error("Run 'python -m time_series_transformer data' to download datasets.")
        sys.exit(1)

    # Load ground-truth labels if provided
    y_true_labels = None
    if args.labels is not None:
        if not args.labels.exists():
            logger.error("Labels file not found: %s", args.labels)
            sys.exit(1)

        from time_series_transformer.data_pipeline.labels import (
            load_label_times,
            make_point_labels_from_times,
        )
        from time_series_transformer.data_pipeline.preprocessing import load_csv_to_df
        from time_series_transformer.split import train_test_split_series

        df = load_csv_to_df(args.csv, parse_dates=["timestamp"])
        df = df.set_index("timestamp").sort_index()
        y = df["value"]

        label_times = load_label_times(args.labels, args.labels_key)
        df_for_labels = df.reset_index()
        y_true_all = make_point_labels_from_times(
            df_for_labels, label_times, timestamp_col="timestamp"
        )
        y_true_all.index = y.index

        _, y_true_labels = train_test_split_series(y_true_all, train_ratio=TRAIN_RATIO)

    try:
        run_pipeline(
            csv_path=args.csv,
            y_true_labels=y_true_labels,
            log_to_mlflow=args.mlflow,
        )
    except Exception as e:
        logger.error("Error during training: %s", e)
        sys.exit(1)
