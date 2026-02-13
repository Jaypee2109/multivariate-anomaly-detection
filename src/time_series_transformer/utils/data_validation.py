"""Data validation utilities for time series input.

Validates CSV files before they enter the training pipeline,
catching common issues early with clear messages.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Outcome of a data validation run."""

    valid: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)
        self.valid = False

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)

    def log(self) -> None:
        for e in self.errors:
            logger.error("  [FAIL] %s", e)
        for w in self.warnings:
            logger.warning("  [WARN] %s", w)
        if self.valid:
            logger.info("  Data validation passed.")


def validate_timeseries(
    path: Path,
    *,
    timestamp_col: str = "timestamp",
    value_col: str = "value",
    min_rows: int = 100,
    max_nan_ratio: float = 0.1,
) -> ValidationResult:
    """Validate a time series CSV for common issues.

    Checks performed:
      - File exists and is CSV
      - Can be parsed by pandas
      - Required columns present
      - Timestamp parses as datetime
      - Value column is numeric
      - Minimum row count (LSTM needs >= lookback + 1)
      - NaN ratio within limits
      - Timestamps are monotonically increasing
    """
    result = ValidationResult()

    # File existence
    if not path.exists():
        result.add_error(f"File not found: {path}")
        return result

    if path.suffix.lower() != ".csv":
        result.add_error(f"Expected .csv file, got: {path.suffix}")
        return result

    # Parse CSV
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        result.add_error(f"Cannot parse CSV: {exc}")
        return result

    # Required columns
    if timestamp_col not in df.columns:
        result.add_error(
            f"Missing timestamp column '{timestamp_col}'. Available: {df.columns.tolist()}"
        )
    if value_col not in df.columns:
        result.add_error(f"Missing value column '{value_col}'. Available: {df.columns.tolist()}")

    if not result.valid:
        return result

    # Timestamp parsing
    ts = pd.to_datetime(df[timestamp_col], errors="coerce")
    n_bad_ts = ts.isna().sum()
    if n_bad_ts == len(df):
        result.add_error(f"All timestamps failed to parse in column '{timestamp_col}'.")
        return result
    if n_bad_ts > 0:
        result.add_warning(f"{n_bad_ts} timestamps could not be parsed ({n_bad_ts / len(df):.1%}).")

    # Value column numeric
    if not pd.api.types.is_numeric_dtype(df[value_col]):
        result.add_error(
            f"Value column '{value_col}' is not numeric (dtype={df[value_col].dtype})."
        )
        return result

    # Row count
    if len(df) < min_rows:
        result.add_error(
            f"Too few rows: {len(df)} (minimum {min_rows}). LSTM requires at least lookback+1 rows."
        )

    # NaN ratio in value column
    nan_count = df[value_col].isna().sum()
    nan_ratio = nan_count / len(df) if len(df) > 0 else 0.0
    if nan_ratio > max_nan_ratio:
        result.add_error(
            f"NaN ratio in '{value_col}' is {nan_ratio:.1%} (max allowed: {max_nan_ratio:.1%})."
        )
    elif nan_count > 0:
        result.add_warning(f"{nan_count} NaN values in '{value_col}' ({nan_ratio:.1%}).")

    # Monotonic timestamps
    ts_valid = ts.dropna()
    if not ts_valid.is_monotonic_increasing:
        result.add_warning("Timestamps are not monotonically increasing — data will be sorted.")

    return result
