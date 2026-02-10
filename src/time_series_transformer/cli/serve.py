"""serve command -- start the anomaly detection inference API server."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from time_series_transformer.config import ARTIFACTS_DIR

logger = logging.getLogger(__name__)

DEFAULT_CHECKPOINT_DIR = ARTIFACTS_DIR / "checkpoints"


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "serve",
        help="Start the anomaly detection inference API server",
        description="Launch a FastAPI server for anomaly detection on trained models.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=DEFAULT_CHECKPOINT_DIR,
        help=f"Directory containing model checkpoints (default: {DEFAULT_CHECKPOINT_DIR})",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Server host (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Server port (default: 8000)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )


def run(args: argparse.Namespace) -> None:
    import os

    import uvicorn

    os.environ["ANOMALY_CHECKPOINT_DIR"] = str(args.checkpoint_dir.resolve())

    logger.info(
        "Starting inference server on http://%s:%d (checkpoints: %s)",
        args.host,
        args.port,
        args.checkpoint_dir,
    )

    uvicorn.run(
        "time_series_transformer.api.inference_server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
