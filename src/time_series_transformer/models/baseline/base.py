from abc import ABC, abstractmethod
import pandas as pd


class BaseAnomalyDetector(ABC):
    """
    Common interface for anomaly detectors.
    """

    @abstractmethod
    def fit(self, y: pd.Series) -> "BaseAnomalyDetector":
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
