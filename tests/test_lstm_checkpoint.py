"""Tests for LSTM checkpoint save/load round-trip."""

from __future__ import annotations

import numpy as np
import pytest

from time_series_transformer.exceptions import ModelNotFittedError
from time_series_transformer.models.baseline.lstm import LSTMForecastAnomalyDetector


@pytest.fixture()
def trained_lstm(sample_timeseries):
    """Train a small LSTM detector on sample data."""
    detector = LSTMForecastAnomalyDetector(
        lookback=5,
        hidden_size=4,
        num_layers=1,
        epochs=2,
        batch_size=16,
    )
    detector.fit(sample_timeseries)
    return detector


class TestLSTMCheckpoint:
    def test_save_and_load(self, tmp_path, trained_lstm, sample_timeseries):
        ckpt_path = tmp_path / "lstm_checkpoint.pt"
        trained_lstm.save_checkpoint(ckpt_path)
        assert ckpt_path.exists()

        loaded = LSTMForecastAnomalyDetector.load_checkpoint(ckpt_path)
        assert loaded._trained is True
        assert loaded.lookback == trained_lstm.lookback
        assert loaded.hidden_size == trained_lstm.hidden_size
        assert loaded.mean_ == trained_lstm.mean_
        assert loaded.std_ == trained_lstm.std_

        # Predictions should match
        orig_scores = trained_lstm.decision_function(sample_timeseries)
        loaded_scores = loaded.decision_function(sample_timeseries)
        np.testing.assert_allclose(orig_scores.values, loaded_scores.values, rtol=1e-5, atol=1e-5)

    def test_save_unfitted_raises(self, tmp_path):
        detector = LSTMForecastAnomalyDetector(lookback=5)
        with pytest.raises(ModelNotFittedError):
            detector.save_checkpoint(tmp_path / "bad.pt")

    def test_load_preserves_hyperparams(self, tmp_path, trained_lstm):
        ckpt_path = tmp_path / "lstm_checkpoint.pt"
        trained_lstm.save_checkpoint(ckpt_path)

        loaded = LSTMForecastAnomalyDetector.load_checkpoint(ckpt_path)
        assert loaded.epochs == trained_lstm.epochs
        assert loaded.lr == trained_lstm.lr
        assert loaded.error_quantile == trained_lstm.error_quantile
