import pandas as pd

from time_series_transformer.exceptions import ModelNotFittedError
from time_series_transformer.models.baseline.base import BaseAnomalyDetector


class RollingZScoreAnomalyDetector(BaseAnomalyDetector):
    """
    Simple rolling z-score anomaly detector.
    Uses training data as history, then evaluates test points
    using a rolling mean/std over (history + test).
    """

    def __init__(
        self, window: int = 12, z_thresh: float = 3.0, min_periods: int | None = None
    ):
        self.window = window
        self.z_thresh = z_thresh
        self.min_periods = min_periods if min_periods is not None else window
        self.history_: pd.Series | None = None

    def fit(self, y: pd.Series) -> "RollingZScoreAnomalyDetector":
        # Store training history; no heavy training here
        self.history_ = y.copy()
        return self

    def _compute_zscores(self, y: pd.Series) -> pd.Series:
        """
        Compute z-scores for y using rolling mean/std over
        history_ + y, then return the part corresponding to y.
        """
        if self.history_ is None:
            raise ModelNotFittedError("Detector not fitted. Call fit() first.")

        combined = pd.concat([self.history_, y])
        rolling_mean = combined.rolling(
            self.window, min_periods=self.min_periods
        ).mean()
        rolling_std = combined.rolling(self.window, min_periods=self.min_periods).std()

        # Avoid division by zero
        rolling_std = rolling_std.replace(0, pd.NA)

        z = (combined - rolling_mean) / rolling_std
        z = z.loc[y.index]
        return z

    def decision_function(self, y: pd.Series) -> pd.Series:
        z = self._compute_zscores(y)
        # Higher score = more anomalous
        return z.abs()

    def predict(self, y: pd.Series) -> pd.Series:
        scores = self.decision_function(y)
        return scores > self.z_thresh
