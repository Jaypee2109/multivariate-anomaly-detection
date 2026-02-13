"""Centralized logging configuration for time_series_transformer."""

from __future__ import annotations

import logging
import sys


def setup_logging(level: str = "INFO") -> None:
    """Configure the root logger for the time_series_transformer package.

    Args:
        level: Log level string ("DEBUG", "INFO", "WARNING", "ERROR").
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))

    root_logger = logging.getLogger("time_series_transformer")
    root_logger.setLevel(numeric_level)

    # Avoid duplicate handlers on repeated calls
    if not root_logger.handlers:
        root_logger.addHandler(handler)
    else:
        root_logger.handlers[0] = handler
        root_logger.setLevel(numeric_level)
