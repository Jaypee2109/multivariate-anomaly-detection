"""Unified CLI entry point for time_series_transformer."""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys

from time_series_transformer.config import PROJECT_ROOT
from time_series_transformer.logging_config import setup_logging
from time_series_transformer.mlflow_utils import MLFLOW_DB_PATH

logger = logging.getLogger(__name__)


def _setup_mlflow_tracking() -> None:
    """Configure MLflow tracking for commands that need it."""
    try:
        from time_series_transformer.mlflow_utils import setup_mlflow

        setup_mlflow()
    except ImportError:
        logger.warning("mlflow not installed.")


def _run_mlflow(args: argparse.Namespace) -> None:
    """Launch MLflow UI."""
    backend_uri = f"sqlite:///{MLFLOW_DB_PATH}"
    cmd = [
        "mlflow",
        "ui",
        "--backend-store-uri",
        backend_uri,
        "--host",
        args.host,
        "--port",
        str(args.port),
    ]
    logger.info("Starting MLflow UI at http://%s:%s", args.host, args.port)
    logger.info("Backend: %s", backend_uri)
    try:
        subprocess.run(cmd, cwd=PROJECT_ROOT)
    except FileNotFoundError:
        logger.error("mlflow CLI not found. Install with: pip install mlflow")
        sys.exit(1)
    except KeyboardInterrupt:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="time_series_transformer",
        description="Time Series Anomaly Detection — Unified CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  %(prog)s data                           Download & preprocess all datasets
  %(prog)s train --mlflow                 Train baselines with MLflow tracking
  %(prog)s train --labels L --labels-key K --mlflow   Train with ground-truth eval
  %(prog)s eda --csv path/to/data.csv     Basic EDA on a time series
  %(prog)s eda --anomalies path/to/artifacts.csv   Visualize anomalies
  %(prog)s info --data path/to/data.csv   Inspect a dataset
  %(prog)s info --run-id <ID>             Inspect an MLflow run
  %(prog)s mlflow                         Start MLflow UI
""",
    )

    # Global verbosity flags
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show debug-level output",
    )
    verbosity.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Show only warnings and errors",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Register command modules
    from time_series_transformer.cli import data, eda, info, train

    data.register(subparsers)
    train.register(subparsers)
    eda.register(subparsers)
    info.register(subparsers)

    # mlflow command (inline — small enough)
    mlflow_parser = subparsers.add_parser(
        "mlflow",
        help="Start MLflow UI for experiment tracking",
        description="Launch the MLflow UI to view and compare experiment runs.",
    )
    mlflow_parser.add_argument("--port", type=int, default=5000, help="Port (default: 5000)")
    mlflow_parser.add_argument("--host", default="127.0.0.1", help="Host (default: 127.0.0.1)")

    # Parse and dispatch
    args = parser.parse_args()

    # Configure logging based on verbosity
    if args.verbose:
        log_level = "DEBUG"
    elif args.quiet:
        log_level = "WARNING"
    else:
        log_level = "INFO"
    setup_logging(log_level)

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    # Setup MLflow only when actually needed
    needs_mlflow = (args.command == "train" and args.mlflow) or (
        args.command == "info" and args.run_id is not None
    )
    if needs_mlflow:
        _setup_mlflow_tracking()

    if args.command == "data":
        data.run(args)
    elif args.command == "train":
        train.run(args)
    elif args.command == "eda":
        eda.run(args)
    elif args.command == "info":
        info.run(args)
    elif args.command == "mlflow":
        _run_mlflow(args)
