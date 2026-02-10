from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import joblib
import pandas as pd


class BaseAnomalyDetector(ABC):
    """
    Common interface for anomaly detectors.
    """

    @abstractmethod
    def fit(self, y: pd.Series) -> BaseAnomalyDetector:
        """
        Fit the detector on training data.
        """
        pass

    @abstractmethod
    def predict(self, y: pd.Series) -> pd.Series:
        """
        Return a boolean Series (index = y.index):
        True where the point is flagged as an anomaly.
        """
        pass

    @abstractmethod
    def decision_function(self, y: pd.Series) -> pd.Series:
        """
        Return anomaly scores (index = y.index).
        Convention here: higher scores = more anomalous.
        """
        pass

    def save_checkpoint(self, path: str | Path) -> None:
        """Save the fitted detector to disk via joblib.

        Subclasses (e.g. LSTM) may override with framework-specific serialization.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load_checkpoint(cls, path: str | Path) -> BaseAnomalyDetector:
        """Load a fitted detector from disk via joblib."""
        path = Path(path)
        obj = joblib.load(path)
        if not isinstance(obj, BaseAnomalyDetector):
            raise TypeError(
                f"Expected a BaseAnomalyDetector subclass, got {type(obj).__name__}"
            )
        return obj
