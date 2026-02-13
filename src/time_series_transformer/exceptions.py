"""Custom exception hierarchy for time_series_transformer.

All project exceptions inherit from TransformerError, making it easy
to catch any project-specific error in a single except clause.
Each subclass also inherits from the closest stdlib exception so that
existing bare ``except FileNotFoundError`` etc. still work.
"""

from __future__ import annotations


class TransformerError(Exception):
    """Base exception for the time_series_transformer package."""


class DataNotFoundError(TransformerError, FileNotFoundError):
    """Raised when a required data file or directory is missing."""


class DataValidationError(TransformerError, ValueError):
    """Raised when data fails a validation check (format, schema, NaN ratio, …)."""


class ModelNotFittedError(TransformerError, RuntimeError):
    """Raised when predict/decision_function is called before fit()."""


class ConfigurationError(TransformerError, ValueError):
    """Raised for invalid or missing configuration values."""
