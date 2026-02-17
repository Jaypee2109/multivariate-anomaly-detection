"""VAR (Vector Autoregression) residual-based multivariate anomaly detector."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR as StatsmodelsVAR

from time_series_transformer.exceptions import ModelNotFittedError
from time_series_transformer.models.multivariate.base import (
    BaseMultivariateAnomalyDetector,
)

logger = logging.getLogger(__name__)


class VARResidualAnomalyDetector(BaseMultivariateAnomalyDetector):
    """Multivariate anomaly detector based on VAR forecast residuals.

    Fits a VAR(p) model on training data.  At test time the forecast
    residuals are z-scored per feature and aggregated across features
    to produce a single anomaly score per timestep.
    """

    def __init__(
        self,
        maxlags: int = 5,
        ic: str | None = "aic",
        z_thresh: float = 3.0,
        aggregation: str = "max",
    ):
        self.maxlags = maxlags
        self.ic = ic
        self.z_thresh = z_thresh
        self.aggregation = aggregation
        self.model_fit_ = None
        self.resid_mean_: np.ndarray | None = None
        self.resid_std_: np.ndarray | None = None
        self.k_ar_: int | None = None

    def fit(self, X: pd.DataFrame) -> VARResidualAnomalyDetector:
        model = StatsmodelsVAR(X.values)
        self.model_fit_ = model.fit(maxlags=self.maxlags, ic=self.ic)
        self.k_ar_ = self.model_fit_.k_ar

        residuals = self.model_fit_.resid
        self.resid_mean_ = residuals.mean(axis=0)
        self.resid_std_ = residuals.std(axis=0, ddof=1)
        # Avoid division by zero for constant features
        self.resid_std_[self.resid_std_ == 0] = 1.0

        logger.info("VAR fitted with lag order=%d", self.k_ar_)
        return self

    def _residual_zscores(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.model_fit_ is None:
            raise ModelNotFittedError("Call fit() first.")

        preds = self.model_fit_.forecast(
            y=self.model_fit_.endog[-self.k_ar_ :],
            steps=len(X),
        )
        preds_df = pd.DataFrame(preds, index=X.index, columns=X.columns)
        residuals = X - preds_df
        z = np.abs((residuals.values - self.resid_mean_) / self.resid_std_)
        return pd.DataFrame(z, index=X.index, columns=X.columns)

    def decision_function(self, X: pd.DataFrame) -> pd.Series:
        z_df = self._residual_zscores(X)
        if self.aggregation == "max":
            scores = z_df.max(axis=1)
        else:
            scores = z_df.mean(axis=1)
        scores.name = "anomaly_score"
        return scores

    def predict(self, X: pd.DataFrame) -> pd.Series:
        scores = self.decision_function(X)
        return (scores > self.z_thresh).rename("is_anomaly")
