from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import joblib
import pandas as pd


class BaseMultivariateAnomalyDetector(ABC):
    """Common interface for multivariate anomaly detectors.

    All methods accept ``pd.DataFrame`` (rows = timesteps, columns = features)
    instead of ``pd.Series``.  ``decision_function`` and ``predict`` still
    return a ``pd.Series`` (one score / label per timestep).
    """

    @abstractmethod
    def fit(self, X: pd.DataFrame) -> BaseMultivariateAnomalyDetector:
        """Fit on multivariate training data.

        Parameters
        ----------
        X : pd.DataFrame
            Shape ``(n_timesteps, n_features)``.
        """

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Return a boolean Series (index = X.index): True where anomaly."""

    @abstractmethod
    def decision_function(self, X: pd.DataFrame) -> pd.Series:
        """Return anomaly scores per timestep (higher = more anomalous)."""

    def save_checkpoint(self, path: str | Path) -> None:
        """Save the fitted detector to disk via joblib.

        Subclasses (e.g. LSTM Autoencoder) may override with
        framework-specific serialisation.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load_checkpoint(cls, path: str | Path) -> BaseMultivariateAnomalyDetector:
        """Load a fitted detector from disk via joblib."""
        path = Path(path)
        obj = joblib.load(path)
        if not isinstance(obj, BaseMultivariateAnomalyDetector):
            raise TypeError(f"Expected BaseMultivariateAnomalyDetector, got {type(obj).__name__}")
        return obj
