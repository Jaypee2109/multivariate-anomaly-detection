"""Tests for the ModelManager class."""

from __future__ import annotations


class TestModelManager:
    def test_load_from_directory(self, tmp_path, sample_timeseries):
        from time_series_transformer.api.model_manager import ModelManager
        from time_series_transformer.models.baseline.isolation_forest import (
            IsolationForestAnomalyDetector,
        )
        from time_series_transformer.models.baseline.rolling_zscore import (
            RollingZScoreAnomalyDetector,
        )

        # Fit and save two models
        iso = IsolationForestAnomalyDetector(contamination=0.05, random_state=42)
        iso.fit(sample_timeseries)
        iso.save_checkpoint(tmp_path / "isolation_forest.joblib")

        zscore = RollingZScoreAnomalyDetector(window=12, z_thresh=2.0)
        zscore.fit(sample_timeseries)
        zscore.save_checkpoint(tmp_path / "rolling_zscore.joblib")

        mgr = ModelManager()
        loaded = mgr.load_from_directory(tmp_path)

        assert "isolation_forest" in loaded
        assert "rolling_zscore" in loaded
        assert len(mgr.loaded_model_names) == 2

    def test_detect_returns_results(self, sample_timeseries):
        from time_series_transformer.api.model_manager import ModelManager
        from time_series_transformer.models.baseline.rolling_zscore import (
            RollingZScoreAnomalyDetector,
        )

        model = RollingZScoreAnomalyDetector(window=12, z_thresh=2.0)
        model.fit(sample_timeseries)

        mgr = ModelManager()
        mgr._models["rolling_zscore"] = model

        results = mgr.detect(sample_timeseries)
        assert "rolling_zscore" in results
        anomalies, scores, latency = results["rolling_zscore"]
        # Rolling Z-Score prepends history, so output may be longer
        assert len(anomalies) >= len(sample_timeseries)
        assert len(scores) >= len(sample_timeseries)
        assert latency > 0

    def test_get_model_info(self, sample_timeseries):
        from time_series_transformer.api.model_manager import ModelManager
        from time_series_transformer.models.baseline.rolling_zscore import (
            RollingZScoreAnomalyDetector,
        )

        model = RollingZScoreAnomalyDetector(window=12, z_thresh=2.0)
        model.fit(sample_timeseries)

        mgr = ModelManager()
        mgr._models["rolling_zscore"] = model

        info = mgr.get_model_info("rolling_zscore")
        assert info["name"] == "rolling_zscore"
        assert info["display_name"] == "Rolling Z-Score"
        assert info["model_class"] == "RollingZScoreAnomalyDetector"
        assert info["parameters"]["window"] == 12
