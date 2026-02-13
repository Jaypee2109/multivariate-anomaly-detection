"""info command — inspect datasets or MLflow runs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "info",
        help="Display dataset or model run info",
        description="Inspect a dataset CSV or an MLflow run.",
    )

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--data",
        type=Path,
        help="Path to a dataset CSV file",
    )
    source.add_argument(
        "--run-id",
        type=str,
        help="MLflow run ID to inspect",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show more detail",
    )


def _show_data_info(csv_path: Path, verbose: bool) -> None:
    """Print summary info about a dataset CSV."""
    import pandas as pd

    if not csv_path.exists():
        print(f"Error: File not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(csv_path)
    print(f"File: {csv_path}")
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"Columns: {', '.join(df.columns.tolist())}")
    print(f"Dtypes:\n{df.dtypes.to_string()}")
    print(f"\nMissing values:\n{df.isnull().sum().to_string()}")

    # Try to detect timestamp column and show time range
    for col in ("timestamp", "time", "datetime", "date", "Date_Time"):
        if col in df.columns:
            ts = pd.to_datetime(df[col], errors="coerce")
            if ts.notna().any():
                print(f"\nTime range ({col}): {ts.min()} to {ts.max()}")
            break

    if verbose:
        print(f"\nDescribe:\n{df.describe(include='all').to_string()}")
        print(f"\nHead:\n{df.head(10).to_string()}")


def _show_run_info(run_id: str, verbose: bool) -> None:
    """Print summary info about an MLflow run."""
    try:
        import mlflow

        from time_series_transformer.mlflow_utils import MLFLOW_TRACKING_URI
    except ImportError:
        print("Error: mlflow not installed.", file=sys.stderr)
        sys.exit(1)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()

    try:
        run = client.get_run(run_id)
    except Exception:
        print(f"Error: Run '{run_id}' not found.", file=sys.stderr)
        sys.exit(1)

    info = run.info
    print(f"Run: {info.run_name} (id={info.run_id})")
    print(f"Status: {info.status}")
    print(f"Start: {info.start_time}")

    tags = {k: v for k, v in run.data.tags.items() if not k.startswith("mlflow.")}
    if tags:
        print("\nTags:")
        for k, v in sorted(tags.items()):
            print(f"  {k}: {v}")

    params = run.data.params
    if params:
        print("\nParams:")
        for k, v in sorted(params.items()):
            print(f"  {k}: {v}")

    metrics = run.data.metrics
    if metrics:
        print("\nMetrics:")
        for k, v in sorted(metrics.items()):
            print(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")

    if verbose:
        artifacts = client.list_artifacts(run_id)
        if artifacts:
            print("\nArtifacts:")
            for a in artifacts:
                print(f"  {a.path} ({a.file_size or '?'} bytes)")


def run(args: argparse.Namespace) -> None:
    if args.data is not None:
        _show_data_info(args.data, verbose=args.verbose)
    elif args.run_id is not None:
        _show_run_info(args.run_id, verbose=args.verbose)
