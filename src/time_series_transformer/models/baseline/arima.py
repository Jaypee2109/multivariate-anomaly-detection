import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from time_series_transformer.models.baseline.base import BaseAnomalyDetector


class ARIMAResidualAnomalyDetector(BaseAnomalyDetector):
    """
    Fit an ARIMA model on training data.
    Anomalies are points where forecast residuals are large.
    """

    def __init__(self, order=(2, 0, 2), z_thresh: float = 3.0):
        self.order = order
        self.z_thresh = z_thresh
        self.model_fit_ = None
        self.resid_std_: float | None = None

    def fit(self, y: pd.Series) -> "ARIMAResidualAnomalyDetector":
        model = ARIMA(y, order=self.order)
        self.model_fit_ = model.fit()
        residuals = self.model_fit_.resid
        self.resid_std_ = float(residuals.std(ddof=1))
        return self

    def _residual_zscores(self, y: pd.Series) -> pd.Series:
        if self.model_fit_ is None or self.resid_std_ is None:
            raise RuntimeError("Detector not fitted. Call fit() first.")

        # Forecast forward for the length of y
        preds = self.model_fit_.forecast(steps=len(y))
        preds = pd.Series(preds, index=y.index)

        residuals = y - preds
        z = residuals / self.resid_std_
        return z

    def decision_function(self, y: pd.Series) -> pd.Series:
        z = self._residual_zscores(y)
        return z.abs()

    def predict(self, y: pd.Series) -> pd.Series:
        scores = self.decision_function(y)
        return scores > self.z_thresh
