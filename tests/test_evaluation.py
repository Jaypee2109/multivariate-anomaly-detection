"""Tests for evaluation module — labels_to_ranges, point metrics, range F1."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from time_series_transformer.evaluation import (
    PointMetrics,
    RangeMetrics,
    compute_point_metrics,
    compute_range_f1_from_labels,
    labels_to_ranges,
)


class TestLabelsToRanges:
    def test_single_contiguous_range(self, sample_timestamps):
        labels = pd.Series(False, index=sample_timestamps, dtype=bool)
        labels.iloc[10:15] = True
        ranges = labels_to_ranges(labels)
        assert len(ranges) == 1
        assert ranges[0] == (sample_timestamps[10], sample_timestamps[14])

    def test_multiple_ranges(self, sample_labels):
        ranges = labels_to_ranges(sample_labels)
        assert len(ranges) == 2

    def test_no_anomalies(self, sample_timestamps):
        labels = pd.Series(False, index=sample_timestamps, dtype=bool)
        assert labels_to_ranges(labels) == []

    def test_all_anomalies(self, sample_timestamps):
        labels = pd.Series(True, index=sample_timestamps, dtype=bool)
        ranges = labels_to_ranges(labels)
        assert len(ranges) == 1
        assert ranges[0] == (sample_timestamps[0], sample_timestamps[-1])

    def test_range_at_end(self, sample_timestamps):
        labels = pd.Series(False, index=sample_timestamps, dtype=bool)
        labels.iloc[-3:] = True
        ranges = labels_to_ranges(labels)
        assert len(ranges) == 1
        assert ranges[0][1] == sample_timestamps[-1]

    def test_requires_datetime_index(self):
        labels = pd.Series([True, False, True], index=[0, 1, 2])
        with pytest.raises(TypeError, match="DatetimeIndex"):
            labels_to_ranges(labels)


class TestComputePointMetrics:
    def test_perfect_predictions(self, sample_timestamps, sample_labels):
        pm = compute_point_metrics(sample_labels, sample_labels)
        assert pm.precision == 1.0
        assert pm.recall == 1.0
        assert pm.f1 == 1.0

    def test_no_predicted_anomalies(self, sample_timestamps, sample_labels):
        y_pred = pd.Series(False, index=sample_timestamps, dtype=bool)
        pm = compute_point_metrics(sample_labels, y_pred)
        assert pm.precision == 0.0
        assert pm.recall == 0.0
        assert pm.f1 == 0.0

    def test_with_scores(self, sample_timestamps, sample_labels):
        rng = np.random.default_rng(42)
        scores = pd.Series(rng.random(100), index=sample_timestamps)
        pm = compute_point_metrics(sample_labels, sample_labels, scores=scores)
        assert pm.auc_roc is not None
        assert pm.auc_pr is not None
        assert 0.0 <= pm.auc_roc <= 1.0
        assert 0.0 <= pm.auc_pr <= 1.0

    def test_returns_point_metrics_type(self, sample_timestamps, sample_labels):
        pm = compute_point_metrics(sample_labels, sample_labels)
        assert isinstance(pm, PointMetrics)


class TestComputeRangeF1:
    def test_perfect_ranges(self, sample_labels):
        rm = compute_range_f1_from_labels(sample_labels, sample_labels)
        assert rm.precision == 1.0
        assert rm.recall == 1.0
        assert rm.f1 == 1.0

    def test_no_overlap(self, sample_timestamps, sample_labels):
        y_pred = pd.Series(False, index=sample_timestamps, dtype=bool)
        y_pred.iloc[50:55] = True  # doesn't overlap 20-24 or 60-62
        rm = compute_range_f1_from_labels(sample_labels, y_pred)
        assert rm.recall == 0.0
        assert rm.n_tp_ranges == 0

    def test_partial_overlap(self, sample_timestamps, sample_labels):
        y_pred = pd.Series(False, index=sample_timestamps, dtype=bool)
        y_pred.iloc[22:27] = True  # overlaps first range only
        rm = compute_range_f1_from_labels(sample_labels, y_pred)
        assert rm.n_tp_ranges == 1
        assert rm.n_pred_ranges == 1
        assert rm.recall == 0.5  # 1 of 2 gt ranges hit

    def test_no_anomalies_anywhere(self, sample_timestamps):
        empty = pd.Series(False, index=sample_timestamps, dtype=bool)
        rm = compute_range_f1_from_labels(empty, empty)
        assert rm.f1 == 1.0
        assert rm.n_gt_ranges == 0

    def test_returns_range_metrics_type(self, sample_labels):
        rm = compute_range_f1_from_labels(sample_labels, sample_labels)
        assert isinstance(rm, RangeMetrics)
