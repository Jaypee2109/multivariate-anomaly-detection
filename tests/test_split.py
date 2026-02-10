"""Tests for train/test split utility."""

from __future__ import annotations

from time_series_transformer.split import train_test_split_series


class TestTrainTestSplitSeries:
    def test_default_ratio(self, sample_timeseries):
        y_train, y_test = train_test_split_series(sample_timeseries)
        assert len(y_train) == 70
        assert len(y_test) == 30

    def test_custom_ratio(self, sample_timeseries):
        y_train, y_test = train_test_split_series(sample_timeseries, train_ratio=0.8)
        assert len(y_train) == 80
        assert len(y_test) == 20

    def test_no_overlap(self, sample_timeseries):
        y_train, y_test = train_test_split_series(sample_timeseries)
        assert len(set(y_train.index) & set(y_test.index)) == 0

    def test_preserves_order(self, sample_timeseries):
        y_train, y_test = train_test_split_series(sample_timeseries)
        assert y_train.index[-1] < y_test.index[0]

    def test_covers_all_data(self, sample_timeseries):
        y_train, y_test = train_test_split_series(sample_timeseries)
        assert len(y_train) + len(y_test) == len(sample_timeseries)

    def test_ratio_zero(self, sample_timeseries):
        y_train, y_test = train_test_split_series(sample_timeseries, train_ratio=0.0)
        assert len(y_train) == 0
        assert len(y_test) == 100

    def test_ratio_one(self, sample_timeseries):
        y_train, y_test = train_test_split_series(sample_timeseries, train_ratio=1.0)
        assert len(y_train) == 100
        assert len(y_test) == 0
