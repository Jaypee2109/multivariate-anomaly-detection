"""Tests for label loading and conversion utilities."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from time_series_transformer.data_pipeline.labels import (
    load_label_times,
    make_point_labels_from_times,
)


class TestLoadLabelTimes:
    def test_loads_timestamps(self, tmp_path):
        data = {"test/file.csv": ["2020-01-01 00:00:00", "2020-01-02 12:00:00"]}
        json_path = tmp_path / "labels.json"
        json_path.write_text(json.dumps(data))

        times = load_label_times(json_path, "test/file.csv")
        assert len(times) == 2
        assert isinstance(times[0], pd.Timestamp)

    def test_missing_key_raises(self, tmp_path):
        data = {"other/file.csv": []}
        json_path = tmp_path / "labels.json"
        json_path.write_text(json.dumps(data))

        with pytest.raises(KeyError, match="not_here"):
            load_label_times(json_path, "not_here")


class TestMakePointLabels:
    def test_marks_correct_timestamps(self):
        timestamps = pd.date_range("2020-01-01", periods=5, freq="h")
        df = pd.DataFrame({"timestamp": timestamps, "value": range(5)})
        label_times = [timestamps[1], timestamps[3]]

        labels = make_point_labels_from_times(df, label_times)
        assert labels.sum() == 2
        assert labels.iloc[1] is True or labels.iloc[1] == True  # noqa: E712
        assert labels.iloc[3] is True or labels.iloc[3] == True  # noqa: E712
        assert labels.iloc[0] is False or labels.iloc[0] == False  # noqa: E712

    def test_returns_boolean_series(self):
        timestamps = pd.date_range("2020-01-01", periods=3, freq="h")
        df = pd.DataFrame({"timestamp": timestamps, "value": [1, 2, 3]})
        labels = make_point_labels_from_times(df, [])
        assert labels.dtype == bool
