"""Tests for data validation utility."""

from __future__ import annotations

import pandas as pd

from time_series_transformer.utils.data_validation import ValidationResult, validate_timeseries


class TestValidationResult:
    def test_starts_valid(self):
        vr = ValidationResult()
        assert vr.valid is True
        assert vr.errors == []
        assert vr.warnings == []

    def test_add_error_invalidates(self):
        vr = ValidationResult()
        vr.add_error("something broke")
        assert vr.valid is False
        assert len(vr.errors) == 1

    def test_add_warning_keeps_valid(self):
        vr = ValidationResult()
        vr.add_warning("minor issue")
        assert vr.valid is True
        assert len(vr.warnings) == 1


class TestValidateTimeseries:
    def test_valid_csv(self, tmp_csv):
        vr = validate_timeseries(tmp_csv)
        assert vr.valid is True

    def test_missing_file(self, tmp_path):
        vr = validate_timeseries(tmp_path / "nonexistent.csv")
        assert vr.valid is False
        assert any("not found" in e for e in vr.errors)

    def test_not_csv(self, tmp_path):
        txt = tmp_path / "data.txt"
        txt.write_text("hello")
        vr = validate_timeseries(txt)
        assert vr.valid is False
        assert any(".csv" in e for e in vr.errors)

    def test_missing_timestamp_col(self, tmp_path):
        csv = tmp_path / "no_ts.csv"
        df = pd.DataFrame({"date": ["2020-01-01"], "value": [1.0]})
        df.to_csv(csv, index=False)
        vr = validate_timeseries(csv, timestamp_col="timestamp")
        assert vr.valid is False
        assert any("timestamp" in e for e in vr.errors)

    def test_missing_value_col(self, tmp_path):
        csv = tmp_path / "no_val.csv"
        df = pd.DataFrame({"timestamp": ["2020-01-01"], "amount": [1.0]})
        df.to_csv(csv, index=False)
        vr = validate_timeseries(csv, value_col="value")
        assert vr.valid is False
        assert any("value" in e for e in vr.errors)

    def test_too_few_rows(self, tmp_path):
        csv = tmp_path / "tiny.csv"
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2020-01-01", periods=5, freq="h"),
                "value": range(5),
            }
        )
        df.to_csv(csv, index=False)
        vr = validate_timeseries(csv, min_rows=10)
        assert vr.valid is False
        assert any("Too few rows" in e for e in vr.errors)

    def test_high_nan_ratio(self, tmp_path):
        csv = tmp_path / "nans.csv"
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2020-01-01", periods=100, freq="h"),
                "value": [float("nan")] * 50 + list(range(50)),
            }
        )
        df.to_csv(csv, index=False)
        vr = validate_timeseries(csv, max_nan_ratio=0.1)
        assert vr.valid is False
        assert any("NaN ratio" in e for e in vr.errors)

    def test_non_monotonic_warns(self, tmp_path):
        csv = tmp_path / "shuffled.csv"
        ts = pd.date_range("2020-01-01", periods=100, freq="h")
        df = pd.DataFrame(
            {
                "timestamp": ts[::-1],  # reversed
                "value": range(100),
            }
        )
        df.to_csv(csv, index=False)
        vr = validate_timeseries(csv)
        assert vr.valid is True
        assert any("monotonic" in w.lower() for w in vr.warnings)

    def test_non_numeric_value(self, tmp_path):
        csv = tmp_path / "strings.csv"
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2020-01-01", periods=100, freq="h"),
                "value": ["text"] * 100,
            }
        )
        df.to_csv(csv, index=False)
        vr = validate_timeseries(csv)
        assert vr.valid is False
        assert any("not numeric" in e for e in vr.errors)
