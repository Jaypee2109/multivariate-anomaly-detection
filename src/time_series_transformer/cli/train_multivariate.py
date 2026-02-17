"""train-mv command — train multivariate anomaly detection models on SMD."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "train-mv",
        help="Train multivariate anomaly detection models on SMD data",
        description="Train multivariate anomaly detectors on a single SMD machine.",
    )

    parser.add_argument(
        "--machine",
        type=str,
        default="machine-1-1",
        help='SMD machine ID, or "all" to run on every machine (default: machine-1-1)',
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=None,
        help="SMD base directory (default: auto from config)",
    )
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        choices=["var", "multi_isolation_forest", "lstm_autoencoder", "lstm_forecaster", "tranad"],
        help="Train only these models (repeatable, default: all)",
    )
    parser.add_argument(
        "--save-checkpoints",
        action="store_true",
        help="Save model checkpoints after training",
    )
    parser.add_argument(
        "--list-machines",
        action="store_true",
        help="List available SMD machines and exit",
    )


def run(args: argparse.Namespace) -> None:
    from time_series_transformer.config import SMD_BASE_DIR

    if args.list_machines:
        from time_series_transformer.data_pipeline.smd_loading import list_smd_machines

        base = args.base_dir or SMD_BASE_DIR
        machines = list_smd_machines(base)
        print(f"Available machines ({len(machines)}):")
        for m in machines:
            print(f"  {m}")
        return

    from time_series_transformer.data_pipeline.smd_loading import list_smd_machines
    from time_series_transformer.multivariate_pipeline import (
        run_multivariate_pipeline,
    )

    base = args.base_dir or SMD_BASE_DIR

    if args.machine.lower() == "all":
        machines = list_smd_machines(base)
        logger.info("Running pipeline on all %d machines", len(machines))
    else:
        machines = [args.machine]

    for i, machine_id in enumerate(machines, 1):
        logger.info("=== [%d/%d] %s ===", i, len(machines), machine_id)
        try:
            run_multivariate_pipeline(
                machine_id=machine_id,
                base_dir=args.base_dir,
                model_names=args.models,
                save_checkpoints=args.save_checkpoints,
            )
        except Exception as e:
            logger.error("Error on %s: %s", machine_id, e, exc_info=True)
            if len(machines) == 1:
                sys.exit(1)
