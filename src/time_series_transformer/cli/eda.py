"""eda command — exploratory data analysis and anomaly visualization."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from time_series_transformer.config import ARTIFACTS_DIR

logger = logging.getLogger(__name__)

DEFAULT_ANOMALIES = ARTIFACTS_DIR / "anomalies" / "baseline_anomalies.csv"


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "eda",
        help="Exploratory data analysis",
        description="Run basic EDA on a time series or visualize detected anomalies.",
    )

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--csv",
        type=Path,
        help="Path to raw time series CSV for basic EDA",
    )
    source.add_argument(
        "--anomalies",
        type=Path,
        nargs="?",
        const=DEFAULT_ANOMALIES,
        help="Path to anomaly artifacts CSV (default: baseline artifacts)",
    )

    parser.add_argument(
        "--no-save-html",
        action="store_true",
        help="Skip saving HTML visualization",
    )


def run(args: argparse.Namespace) -> None:
    if args.csv is not None:
        if not args.csv.exists():
            logger.error("File not found: %s", args.csv)
            sys.exit(1)

        from time_series_transformer.analysis.eda import run_basic_eda_from_csv

        run_basic_eda_from_csv(
            csv_path=args.csv,
            save_html=not args.no_save_html,
        )

    elif args.anomalies is not None:
        if not args.anomalies.exists():
            logger.error("Artifacts file not found: %s", args.anomalies)
            logger.error("Run 'python -m time_series_transformer train' first.")
            sys.exit(1)

        from time_series_transformer.analysis.eda import run_anomaly_eda_from_artifacts

        run_anomaly_eda_from_artifacts(
            artifacts_path=args.anomalies,
            save_html=not args.no_save_html,
        )
