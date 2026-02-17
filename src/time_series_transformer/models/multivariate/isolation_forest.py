"""Multivariate Isolation Forest anomaly detector."""

from __future__ import annotations

import pandas as pd
from sklearn.ensemble import IsolationForest

from time_series_transformer.exceptions import ModelNotFittedError
from time_series_transformer.models.multivariate.base import (
    BaseMultivariateAnomalyDetector,
)


class MultivariateIsolationForestDetector(BaseMultivariateAnomalyDetector):
    """Isolation Forest operating on the full multivariate feature matrix.

    sklearn's ``IsolationForest`` natively supports multivariate input,
    so this is a thin wrapper that passes ``X.values`` directly.
    """

    def __init__(
        self,
        contamination: float = 0.05,
        random_state: int | None = None,
    ):
        self.contamination = contamination
        self.random_state = random_state
        self.model_: IsolationForest | None = None

    def fit(self, X: pd.DataFrame) -> MultivariateIsolationForestDetector:
        self.model_ = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
        )
        self.model_.fit(X.values)
        return self

    def decision_function(self, X: pd.DataFrame) -> pd.Series:
        if self.model_ is None:
            raise ModelNotFittedError("Call fit() first.")
        # sklearn: higher = more normal → flip sign
        scores = -self.model_.decision_function(X.values)
        return pd.Series(scores, index=X.index, name="anomaly_score")

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if self.model_ is None:
            raise ModelNotFittedError("Call fit() first.")
        labels = self.model_.predict(X.values)  # -1 = anomaly, 1 = normal
        return pd.Series(labels == -1, index=X.index, name="is_anomaly")
