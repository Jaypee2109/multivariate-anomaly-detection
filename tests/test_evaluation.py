"""Tests for evaluation module — labels_to_ranges, point metrics, range F1,
point-adjust, best-F1, detection latency."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from time_series_transformer.evaluation import (
    BestF1Result,
    LatencyResult,
    PointAdjustMetrics,
    PointMetrics,
    RangeMetrics,
    compute_best_f1,
    compute_detection_latency,
    compute_point_adjust_metrics,
    compute_point_metrics,
    compute_range_f1_from_labels,
    labels_to_ranges,
    point_adjust,
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

    def test_integer_index_supported(self):
        labels = pd.Series([False, True, True, False, True], index=[0, 1, 2, 3, 4])
        ranges = labels_to_ranges(labels)
        assert len(ranges) == 2
        assert ranges[0] == (1, 2)
        assert ranges[1] == (4, 4)


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


class TestPointAdjust:
    """Tests for the point-adjust protocol (PA)."""

    def test_single_hit_fills_segment(self):
        """Detecting one point in a segment marks the whole segment."""
        # GT segment at [2..6], prediction hits only index 4
        y_true = np.array([0, 0, 1, 1, 1, 1, 1, 0, 0, 0], dtype=bool)
        y_pred = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0], dtype=bool)
        adj = point_adjust(y_true, y_pred)
        # Entire segment [2..6] should now be True
        np.testing.assert_array_equal(adj[2:7], [True] * 5)
        # Outside segment stays unchanged
        assert not adj[0]
        assert not adj[1]
        assert not adj[7]

    def test_no_hit_leaves_segment_undetected(self):
        """If no point in a segment is predicted, PA does nothing."""
        y_true = np.array([0, 0, 1, 1, 1, 0, 0], dtype=bool)
        y_pred = np.array([0, 0, 0, 0, 0, 0, 0], dtype=bool)
        adj = point_adjust(y_true, y_pred)
        np.testing.assert_array_equal(adj, y_pred)

    def test_multiple_segments(self):
        """PA works independently per segment."""
        # Segment A: [1..3], Segment B: [6..8]
        y_true = np.array([0, 1, 1, 1, 0, 0, 1, 1, 1, 0], dtype=bool)
        # Hit only in segment B (index 7)
        y_pred = np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0], dtype=bool)
        adj = point_adjust(y_true, y_pred)
        # A not detected
        np.testing.assert_array_equal(adj[1:4], [False, False, False])
        # B fully detected
        np.testing.assert_array_equal(adj[6:9], [True, True, True])

    def test_segment_at_end(self):
        """PA handles a segment that extends to the last element."""
        y_true = np.array([0, 0, 1, 1, 1], dtype=bool)
        y_pred = np.array([0, 0, 1, 0, 0], dtype=bool)
        adj = point_adjust(y_true, y_pred)
        np.testing.assert_array_equal(adj[2:5], [True, True, True])

    def test_false_positives_preserved(self):
        """FP predictions outside GT segments are not removed by PA."""
        y_true = np.array([0, 0, 0, 1, 1, 0, 0], dtype=bool)
        y_pred = np.array([1, 0, 0, 1, 0, 0, 1], dtype=bool)
        adj = point_adjust(y_true, y_pred)
        assert adj[0]  # FP preserved
        assert adj[6]  # FP preserved
        np.testing.assert_array_equal(adj[3:5], [True, True])  # segment filled

    def test_perfect_predictions(self, sample_timestamps, sample_labels):
        """Perfect predictions yield PA-F1 = 1.0."""
        pa = compute_point_adjust_metrics(sample_labels, sample_labels)
        assert isinstance(pa, PointAdjustMetrics)
        assert pa.f1 == 1.0
        assert pa.precision == 1.0
        assert pa.recall == 1.0

    def test_pa_inflates_recall(self, sample_timestamps, sample_labels):
        """PA recall should be >= point recall when at least one hit per segment."""
        # Predict only first point of each GT segment
        y_pred = pd.Series(False, index=sample_timestamps, dtype=bool)
        y_pred.iloc[20] = True  # first point of segment 1 (20-24)
        y_pred.iloc[60] = True  # first point of segment 2 (60-62)

        pm = compute_point_metrics(sample_labels, y_pred)
        pa = compute_point_adjust_metrics(sample_labels, y_pred)
        assert pa.recall >= pm.recall
        assert pa.recall == 1.0  # both segments fully detected


class TestBestF1:
    """Tests for best-F1 threshold search."""

    def test_perfect_separation(self, sample_timestamps, sample_labels):
        """When scores perfectly separate anomalies, best F1 should be 1.0."""
        # Score = 1.0 for anomalies, 0.0 for normals
        scores = sample_labels.astype(float)
        result = compute_best_f1(sample_labels, scores)
        assert isinstance(result, BestF1Result)
        assert result.f1 == pytest.approx(1.0)
        assert result.pa_f1 == pytest.approx(1.0)

    def test_random_scores_below_perfect(self, sample_timestamps, sample_labels):
        """Random scores should yield best-F1 < 1.0."""
        rng = np.random.default_rng(42)
        scores = pd.Series(rng.random(100), index=sample_timestamps)
        result = compute_best_f1(sample_labels, scores)
        assert 0.0 <= result.f1 <= 1.0
        assert result.pa_f1 >= result.f1  # PA-F1 >= point F1

    def test_threshold_in_score_range(self, sample_timestamps, sample_labels):
        """Optimal threshold should be within the score range."""
        rng = np.random.default_rng(42)
        scores = pd.Series(rng.random(100), index=sample_timestamps)
        result = compute_best_f1(sample_labels, scores)
        assert scores.min() <= result.threshold <= scores.max()


class TestDetectionLatency:
    """Tests for detection latency metric."""

    def test_immediate_detection(self, sample_timestamps, sample_labels):
        """Perfect predictions → 0 latency."""
        lat = compute_detection_latency(sample_labels, sample_labels)
        assert isinstance(lat, LatencyResult)
        assert lat.mean_latency == 0.0
        assert lat.n_detected == lat.n_segments
        assert lat.n_missed == 0

    def test_missed_segments(self, sample_timestamps, sample_labels):
        """No predictions → all segments missed, latency = segment length."""
        y_pred = pd.Series(False, index=sample_timestamps, dtype=bool)
        lat = compute_detection_latency(sample_labels, y_pred)
        assert lat.n_detected == 0
        assert lat.n_missed == 2  # two GT segments (20-24, 60-62)
        assert lat.mean_latency > 0

    def test_delayed_detection(self):
        """Detection on the last point of a 5-point segment → latency = 4."""
        idx = pd.RangeIndex(10)
        y_true = pd.Series([0, 0, 1, 1, 1, 1, 1, 0, 0, 0], index=idx, dtype=bool)
        y_pred = pd.Series([0, 0, 0, 0, 0, 0, 1, 0, 0, 0], index=idx, dtype=bool)
        lat = compute_detection_latency(y_true, y_pred)
        assert lat.n_segments == 1
        assert lat.n_detected == 1
        assert lat.mean_latency == 4.0  # 5-point segment, detected at offset 4

    def test_no_anomalies(self, sample_timestamps):
        """No GT anomalies → latency is 0, 0 segments."""
        empty = pd.Series(False, index=sample_timestamps, dtype=bool)
        lat = compute_detection_latency(empty, empty)
        assert lat.n_segments == 0
        assert lat.mean_latency == 0.0
