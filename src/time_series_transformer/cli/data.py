"""data command — download and preprocess datasets from Kaggle."""

from __future__ import annotations

import argparse
import logging
import sys

from time_series_transformer.config import KAGGLE_DATASETS

logger = logging.getLogger(__name__)


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "data",
        help="Download and preprocess datasets",
        description="Download datasets from Kaggle and run preprocessing pipeline.",
    )
    parser.add_argument(
        "--dataset",
        choices=list(KAGGLE_DATASETS.keys()),
        help="Process only this dataset (default: all)",
    )


def run(args: argparse.Namespace) -> None:
    from time_series_transformer.data_pipeline.pipeline import run_data_pipeline

    datasets = [args.dataset] if args.dataset else None

    try:
        run_data_pipeline(datasets=datasets)
    except Exception as e:
        logger.error("%s", e)
        sys.exit(1)
