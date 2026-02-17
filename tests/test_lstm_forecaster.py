"""Tests for multivariate LSTM Forecaster anomaly detector."""

from __future__ import annotations

import pandas as pd
import pytest
import torch


class TestMultivariateLSTMForecasterModel:
    def test_forward_shape(self):
        from time_series_transformer.models.multivariate.lstm_forecaster import (
            MultivariateLSTMForecaster,
        )

        n_features = 5
        model = MultivariateLSTMForecaster(
            n_features=n_features, hidden_size=16, num_layers=1,
        )
        x = torch.randn(4, 10, n_features)  # (batch=4, seq_len=10, features=5)
        out = model(x)
        assert out.shape == (4, n_features)


class TestLSTMForecasterDetector:
    def test_fit_predict(self, sample_multivariate_df):
        from time_series_transformer.models.multivariate.lstm_forecaster import (
            LSTMForecasterMultivariateDetector,
        )

        det = LSTMForecasterMultivariateDetector(
            lookback=10, hidden_size=16, epochs=2, batch_size=16,
        )
        train = sample_multivariate_df.iloc[:70]
        test = sample_multivariate_df.iloc[70:]
        det.fit(train)

        scores = det.decision_function(test)
        preds = det.predict(test)

        assert isinstance(scores, pd.Series)
        assert len(scores) == len(test)
        assert isinstance(preds, pd.Series)
        assert preds.dtype == bool

    def test_checkpoint_roundtrip(self, tmp_path, sample_multivariate_df):
        from time_series_transformer.models.multivariate.lstm_forecaster import (
            LSTMForecasterMultivariateDetector,
        )

        det = LSTMForecasterMultivariateDetector(
            lookback=10, hidden_size=16, epochs=2, batch_size=16,
        )
        det.fit(sample_multivariate_df.iloc[:70])

        path = tmp_path / "lstm_fc.pt"
        det.save_checkpoint(path)

        loaded = LSTMForecasterMultivariateDetector.load_checkpoint(path)
        test = sample_multivariate_df.iloc[70:]
        scores = loaded.decision_function(test)
        assert len(scores) == len(test)
