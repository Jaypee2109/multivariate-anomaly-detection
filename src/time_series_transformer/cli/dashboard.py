"""dashboard command -- start the anomaly detection dashboard."""

from __future__ import annotations

import argparse
import logging

logger = logging.getLogger(__name__)


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "dashboard",
        help="Start the interactive anomaly detection dashboard",
        description="Launch the Dash dashboard (requires the inference API to be running).",
    )
    parser.add_argument(
        "--host", default="127.0.0.1", help="Dashboard host (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port", type=int, default=8050, help="Dashboard port (default: 8050)"
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable Dash debug/hot-reload mode"
    )


def run(args: argparse.Namespace) -> None:
    import sys
    from pathlib import Path

    # Add the dashboard directory to sys.path so Dash can find pages/
    dashboard_dir = Path(__file__).resolve().parents[3] / "dashboard"
    if not dashboard_dir.exists():
        logger.error("Dashboard directory not found: %s", dashboard_dir)
        sys.exit(1)

    sys.path.insert(0, str(dashboard_dir))

    # Import must happen after sys.path adjustment so pages/ is discovered
    from app import app  # noqa: E402

    logger.info(
        "Starting dashboard at http://%s:%d", args.host, args.port
    )
    app.run(host=args.host, port=args.port, debug=args.debug)
