"""Tests for multivariate anomaly detectors."""

from __future__ import annotations

import pandas as pd
import pytest


class TestVARResidualDetector:
    def test_fit_predict(self, sample_multivariate_df):
        from time_series_transformer.models.multivariate.var import (
            VARResidualAnomalyDetector,
        )

        det = VARResidualAnomalyDetector(maxlags=2, z_thresh=3.0)
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
        from time_series_transformer.models.multivariate.var import (
            VARResidualAnomalyDetector,
        )

        det = VARResidualAnomalyDetector(maxlags=2, z_thresh=3.0)
        det.fit(sample_multivariate_df.iloc[:70])

        path = tmp_path / "var.joblib"
        det.save_checkpoint(path)

        loaded = VARResidualAnomalyDetector.load_checkpoint(path)
        scores = loaded.decision_function(sample_multivariate_df.iloc[70:])
        assert len(scores) == 30


class TestMultivariateIsolationForest:
    def test_fit_predict(self, sample_multivariate_df):
        from time_series_transformer.models.multivariate.isolation_forest import (
            MultivariateIsolationForestDetector,
        )

        det = MultivariateIsolationForestDetector(contamination=0.1, random_state=42)
        det.fit(sample_multivariate_df)

        scores = det.decision_function(sample_multivariate_df)
        preds = det.predict(sample_multivariate_df)

        assert isinstance(scores, pd.Series)
        assert len(scores) == len(sample_multivariate_df)
        assert isinstance(preds, pd.Series)
        assert preds.dtype == bool
        # With contamination=0.1, ~10% should be flagged
        assert 0 < preds.sum() <= len(sample_multivariate_df)


class TestLSTMAutoencoder:
    def test_fit_predict(self, sample_multivariate_df):
        from time_series_transformer.models.multivariate.lstm_autoencoder import (
            LSTMAutoencoderAnomalyDetector,
        )

        det = LSTMAutoencoderAnomalyDetector(
            lookback=10, hidden_size=16, latent_dim=8, epochs=2, batch_size=16,
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
        from time_series_transformer.models.multivariate.lstm_autoencoder import (
            LSTMAutoencoderAnomalyDetector,
        )

        det = LSTMAutoencoderAnomalyDetector(
            lookback=10, hidden_size=16, latent_dim=8, epochs=2, batch_size=16,
        )
        det.fit(sample_multivariate_df.iloc[:70])

        path = tmp_path / "lstm_ae.pt"
        det.save_checkpoint(path)

        loaded = LSTMAutoencoderAnomalyDetector.load_checkpoint(path)
        test = sample_multivariate_df.iloc[70:]
        scores = loaded.decision_function(test)
        assert len(scores) == len(test)
