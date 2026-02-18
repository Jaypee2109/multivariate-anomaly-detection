from __future__ import annotations

import pandas as pd
from sklearn.ensemble import IsolationForest

from time_series_transformer.exceptions import ModelNotFittedError
from time_series_transformer.models.baseline.base import BaseAnomalyDetector


class IsolationForestAnomalyDetector(BaseAnomalyDetector):
    """
    Classical tree-based anomaly detector.
    Treats the value as the feature; you can extend with lags, etc.
    """

    def __init__(self, contamination: float = 0.05, random_state: int | None = None):
        self.contamination = contamination
        self.random_state = random_state
        self.model_: IsolationForest | None = None

    def fit(self, y: pd.Series) -> IsolationForestAnomalyDetector:
        self.model_ = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
        )
        X = y.values.reshape(-1, 1)
        self.model_.fit(X)
        return self

    def decision_function(self, y: pd.Series) -> pd.Series:
        if self.model_ is None:
            raise ModelNotFittedError("Detector not fitted. Call fit() first.")

        X = y.values.reshape(-1, 1)
        # In sklearn, higher decision_function = more normal.
        # We flip the sign so higher = more anomalous.
        scores = -self.model_.decision_function(X)
        return pd.Series(scores, index=y.index)

    def predict(self, y: pd.Series) -> pd.Series:
        if self.model_ is None:
            raise ModelNotFittedError("Detector not fitted. Call fit() first.")

        X = y.values.reshape(-1, 1)
        labels = self.model_.predict(X)  # -1 = anomaly, 1 = normal
        anomalies = labels == -1
        return pd.Series(anomalies, index=y.index)
