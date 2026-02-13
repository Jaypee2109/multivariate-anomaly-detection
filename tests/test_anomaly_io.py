"""Tests for anomaly artifact save/load round-trip."""

from __future__ import annotations

import pandas as pd

from time_series_transformer.utils.anomaly_io import (
    load_anomaly_flags_from_artifacts,
    save_anomaly_artifacts,
)


class TestAnomalyIORoundTrip:
    def test_save_and_load(self, tmp_path, sample_timeseries, sample_labels):
        out_path = tmp_path / "artifacts.csv"

        scores_dict = {"TestDetector": pd.Series(0.5, index=sample_timeseries.index)}
        anomalies_dict = {"TestDetector": sample_labels}

        save_anomaly_artifacts(
            y_test=sample_timeseries,
            scores_dict=scores_dict,
            anomalies_dict=anomalies_dict,
            out_path=out_path,
        )

        assert out_path.exists()

        loaded = load_anomaly_flags_from_artifacts(out_path)
        assert "TestDetector" in loaded
        assert loaded["TestDetector"].dtype == bool
        assert loaded["TestDetector"].sum() == sample_labels.sum()

    def test_empty_anomaly_columns_warns(self, tmp_path, sample_timeseries):
        out_path = tmp_path / "no_anomalies.csv"
        df = pd.DataFrame({"timestamp": sample_timeseries.index, "value": sample_timeseries.values})
        df.to_csv(out_path, index=False)

        result = load_anomaly_flags_from_artifacts(out_path)
        assert result == {}
