"""benchmark command — systematic evaluation across models and datasets."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from time_series_transformer.config import ARTIFACTS_DIR, PROJECT_ROOT

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Registration
# ------------------------------------------------------------------


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "benchmark",
        help="Run systematic evaluation across models and datasets",
        description=(
            "Benchmark anomaly detection models on one or more datasets.\n"
            "Datasets are defined in a YAML config file or passed as individual CSVs."
        ),
    )

    # Datasets
    ds = parser.add_argument_group("Datasets")
    ds.add_argument(
        "--config",
        type=Path,
        default=None,
        metavar="YAML",
        help="YAML config file defining datasets to benchmark",
    )
    ds.add_argument(
        "--csv",
        type=Path,
        action="append",
        dest="csv_files",
        help="Add an individual CSV (no labels). Repeatable.",
    )

    # Models
    ms = parser.add_argument_group("Models")
    ms.add_argument(
        "--model",
        action="append",
        dest="models",
        metavar="NAME",
        help="Model name to include (repeatable, default: all registered)",
    )
    ms.add_argument(
        "--list-models",
        action="store_true",
        help="List registered models and exit",
    )

    # Output
    out = parser.add_argument_group("Output")
    out.add_argument(
        "--output",
        type=Path,
        default=ARTIFACTS_DIR / "benchmark" / "results.csv",
        help="CSV output path (default: artifacts/benchmark/results.csv)",
    )
    out.add_argument(
        "--no-console",
        action="store_true",
        help="Suppress printing results to the console",
    )

    # Tracking
    parser.add_argument(
        "--mlflow",
        action="store_true",
        help="Log each run to MLflow",
    )


# ------------------------------------------------------------------
# Execution
# ------------------------------------------------------------------


def run(args: argparse.Namespace) -> None:
    from time_series_transformer.benchmark import BenchmarkRunner, DatasetSpec
    from time_series_transformer.benchmark.registry import list_models

    # --list-models
    if args.list_models:
        print("Registered models:")
        for m in list_models():
            print(f"  - {m}")
        return

    # Build dataset list
    datasets: list[DatasetSpec] = []

    if args.config is not None:
        datasets.extend(_load_config(args.config))

    if args.csv_files:
        for p in args.csv_files:
            datasets.append(DatasetSpec(name=p.stem, csv_path=p))

    if not datasets:
        logger.error(
            "No datasets specified. Use --config <yaml> or --csv <path>."
        )
        sys.exit(1)

    logger.info("Benchmarking on %d dataset(s)", len(datasets))

    # Run
    runner = BenchmarkRunner(
        datasets=datasets,
        model_names=args.models,
        log_to_mlflow=args.mlflow,
    )
    results = runner.run()

    # Output
    if not args.no_console:
        results.print_summary()

    results.to_csv(args.output)
    logger.info("Results saved to %s", args.output)


# ------------------------------------------------------------------
# YAML config loader
# ------------------------------------------------------------------


def _load_config(config_path: Path) -> list["DatasetSpec"]:
    """Load dataset definitions from a YAML file.

    Expected format::

        datasets:
          - name: NAB_nyc_taxi
            csv: data/raw/nab/realKnownCause/realKnownCause/nyc_taxi.csv
            labels: data/labels/nab/realKnownCause.json       # optional
            labels_key: realKnownCause/nyc_taxi.csv            # optional

    Paths are resolved relative to the project root.
    """
    from time_series_transformer.benchmark import DatasetSpec

    try:
        import yaml
    except ImportError:
        logger.error("PyYAML is required for --config. Install with: pip install pyyaml")
        sys.exit(1)

    if not config_path.exists():
        logger.error("Config file not found: %s", config_path)
        sys.exit(1)

    with config_path.open() as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict) or "datasets" not in raw:
        logger.error("Config must contain a top-level 'datasets' list.")
        sys.exit(1)

    specs: list[DatasetSpec] = []
    for entry in raw["datasets"]:
        name = entry.get("name")
        csv_rel = entry.get("csv")
        if not name or not csv_rel:
            logger.warning("Skipping entry without 'name' or 'csv': %s", entry)
            continue

        csv_path = _resolve(csv_rel)
        if not csv_path.exists():
            logger.warning("CSV not found, skipping: %s", csv_path)
            continue

        labels_path = None
        labels_key = entry.get("labels_key")
        if "labels" in entry:
            labels_path = _resolve(entry["labels"])
            if not labels_path.exists():
                logger.warning("Labels file not found, ignoring: %s", labels_path)
                labels_path = None
                labels_key = None

        specs.append(
            DatasetSpec(
                name=name,
                csv_path=csv_path,
                labels_path=labels_path,
                labels_key=labels_key,
            )
        )

    logger.info("Loaded %d dataset(s) from %s", len(specs), config_path)
    return specs


def _resolve(rel: str) -> Path:
    """Resolve a path relative to the project root."""
    p = Path(rel)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p
