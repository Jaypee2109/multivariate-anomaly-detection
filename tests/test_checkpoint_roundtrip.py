"""Tests for save/load checkpoint round-trip on all baseline models."""

from __future__ import annotations

import numpy as np


class TestARIMACheckpoint:
    def test_roundtrip(self, tmp_path, sample_timeseries):
        from time_series_transformer.models.baseline.arima import ARIMAResidualAnomalyDetector

        det = ARIMAResidualAnomalyDetector(order=(1, 0, 1), z_thresh=3.0)
        det.fit(sample_timeseries)

        path = tmp_path / "arima.joblib"
        det.save_checkpoint(path)
        assert path.exists()

        loaded = ARIMAResidualAnomalyDetector.load_checkpoint(path)
        assert loaded.order == det.order
        assert loaded.z_thresh == det.z_thresh

        orig = det.decision_function(sample_timeseries[70:])
        reloaded = loaded.decision_function(sample_timeseries[70:])
        np.testing.assert_allclose(orig.values, reloaded.values, rtol=1e-5)


class TestIsolationForestCheckpoint:
    def test_roundtrip(self, tmp_path, sample_timeseries):
        from time_series_transformer.models.baseline.isolation_forest import (
            IsolationForestAnomalyDetector,
        )

        det = IsolationForestAnomalyDetector(contamination=0.05, random_state=42)
        det.fit(sample_timeseries)

        path = tmp_path / "iso.joblib"
        det.save_checkpoint(path)
        assert path.exists()

        loaded = IsolationForestAnomalyDetector.load_checkpoint(path)
        assert loaded.contamination == det.contamination

        orig = det.decision_function(sample_timeseries)
        reloaded = loaded.decision_function(sample_timeseries)
        np.testing.assert_allclose(orig.values, reloaded.values, rtol=1e-5)


class TestRollingZScoreCheckpoint:
    def test_roundtrip(self, tmp_path, sample_timeseries):
        from time_series_transformer.models.baseline.rolling_zscore import (
            RollingZScoreAnomalyDetector,
        )

        det = RollingZScoreAnomalyDetector(window=12, z_thresh=2.0)
        det.fit(sample_timeseries)

        path = tmp_path / "zscore.joblib"
        det.save_checkpoint(path)
        assert path.exists()

        loaded = RollingZScoreAnomalyDetector.load_checkpoint(path)
        assert loaded.window == det.window
        assert loaded.z_thresh == det.z_thresh

        orig = det.decision_function(sample_timeseries)
        reloaded = loaded.decision_function(sample_timeseries)
        np.testing.assert_allclose(orig.values, reloaded.values, rtol=1e-5)
