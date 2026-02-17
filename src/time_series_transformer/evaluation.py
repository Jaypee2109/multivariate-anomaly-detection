from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
)


@dataclass
class PointMetrics:
    precision: float
    recall: float
    f1: float
    auc_roc: float | None = None
    auc_pr: float | None = None


def _align_series(
    y_true: pd.Series,
    y_pred: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    """
    Align two Series on a common index (inner join).
    """
    y_true_aligned, y_pred_aligned = y_true.align(y_pred, join="inner")
    return y_true_aligned, y_pred_aligned


def labels_to_ranges(labels: pd.Series) -> list[tuple]:
    """
    Convert a boolean / {0,1} Series into a list of contiguous anomaly ranges
    ``(start, end)``.

    Works with any ordered index (``DatetimeIndex``, ``RangeIndex``, integer, …).
    Contiguous = consecutive True values in the Series.
    """
    labels_bool = labels.astype(bool)

    ranges: list[tuple] = []
    in_range = False
    start: pd.Timestamp | None = None
    prev_t: pd.Timestamp | None = None

    for t, v in labels_bool.items():
        if v and not in_range:
            # start of a new range
            in_range = True
            start = t
        elif not v and in_range:
            # end of current range (previous timestamp)
            assert start is not None
            ranges.append((start, prev_t))  # type: ignore[arg-type]
            in_range = False
            start = None
        prev_t = t

    # if we end inside a range, close it
    if in_range and start is not None and prev_t is not None:
        ranges.append((start, prev_t))

    return ranges


def _ranges_overlap(r1: tuple, r2: tuple) -> bool:
    s1, e1 = r1
    s2, e2 = r2
    return max(s1, s2) <= min(e1, e2)


def compute_point_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
    scores: pd.Series | None = None,
    pos_label: int | bool = 1,
) -> PointMetrics:
    """
    Compute precision/recall/F1 at the point level.
    Optionally also compute ROC-AUC and PR-AUC if `scores` are provided.

    y_true: ground truth anomaly labels (bool or {0,1})
    y_pred: predicted anomaly labels
    scores: anomaly scores (higher = more anomalous)
    """
    y_true_aligned, y_pred_aligned = _align_series(y_true, y_pred)
    y_true_bin = y_true_aligned.astype(int)
    y_pred_bin = y_pred_aligned.astype(int)

    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true_bin,
        y_pred_bin,
        average="binary",
        pos_label=int(pos_label),
        zero_division=0,
    )

    auc_roc = None
    auc_pr = None

    if scores is not None:
        y_true_scores, scores_aligned = _align_series(y_true, scores)
        y_true_scores = y_true_scores.astype(int)

        # Need at least one positive and one negative for AUC-ROC
        if y_true_scores.nunique() == 2:
            try:
                auc_roc = float(roc_auc_score(y_true_scores, scores_aligned))
            except ValueError:
                auc_roc = None

        # PR-AUC (Average Precision) only needs positives
        if y_true_scores.sum() > 0:
            try:
                auc_pr = float(average_precision_score(y_true_scores, scores_aligned))
            except ValueError:
                auc_pr = None

    return PointMetrics(
        precision=float(prec),
        recall=float(rec),
        f1=float(f1),
        auc_roc=auc_roc,
        auc_pr=auc_pr,
    )


@dataclass
class RangeMetrics:
    precision: float
    recall: float
    f1: float
    n_gt_ranges: int
    n_pred_ranges: int
    n_tp_ranges: int


def compute_range_f1_from_labels(
    y_true: pd.Series,
    y_pred: pd.Series,
) -> RangeMetrics:
    """
    Compute F1 at the *range* level.

    - Ground truth and predictions are given as point-wise labels.
    - We first convert them into contiguous ranges (labels_to_ranges),
      then define:

        TP_range: GT range that overlaps at least one predicted range
        FP_range: predicted range with no overlap with any GT range
        FN_range: GT range with no overlap with any predicted range
    """
    # Ensure aligned for timestamps
    y_true_aligned, y_pred_aligned = _align_series(y_true, y_pred)
    y_true_bool = y_true_aligned.astype(bool)
    y_pred_bool = y_pred_aligned.astype(bool)

    gt_ranges = labels_to_ranges(y_true_bool)
    pred_ranges = labels_to_ranges(y_pred_bool)

    n_gt = len(gt_ranges)
    n_pred = len(pred_ranges)

    if n_gt == 0 and n_pred == 0:
        # trivial case: no anomalies anywhere
        return RangeMetrics(
            precision=1.0,
            recall=1.0,
            f1=1.0,
            n_gt_ranges=0,
            n_pred_ranges=0,
            n_tp_ranges=0,
        )

    # GT ranges that are "hit" by at least one predicted range
    gt_hit = [False] * n_gt
    # predicted ranges that overlap any GT range
    pred_hit = [False] * n_pred

    for i, g in enumerate(gt_ranges):
        for j, p in enumerate(pred_ranges):
            if _ranges_overlap(g, p):
                gt_hit[i] = True
                pred_hit[j] = True

    tp_ranges = sum(gt_hit)
    fn_ranges = n_gt - tp_ranges
    fp_ranges = n_pred - sum(pred_hit)

    precision = tp_ranges / (tp_ranges + fp_ranges) if (tp_ranges + fp_ranges) > 0 else 0.0
    recall = tp_ranges / (tp_ranges + fn_ranges) if (tp_ranges + fn_ranges) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return RangeMetrics(
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        n_gt_ranges=n_gt,
        n_pred_ranges=n_pred,
        n_tp_ranges=tp_ranges,
    )


# ---------------------------------------------------------------------------
# Point-Adjust (PA) evaluation — protocol from OmniAnomaly / TranAD
# ---------------------------------------------------------------------------


@dataclass
class PointAdjustMetrics:
    """Metrics computed after the point-adjust protocol."""

    precision: float
    recall: float
    f1: float


def point_adjust(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> np.ndarray:
    """Apply the point-adjust protocol to binary predictions.

    For each contiguous ground-truth anomaly segment, if **any** predicted
    point within the segment is ``True``, the **entire** segment is marked
    as detected (``True``).  Points outside ground-truth segments are left
    unchanged.

    Parameters
    ----------
    y_true : 1-D array of bool / {0, 1}
        Ground-truth anomaly labels.
    y_pred : 1-D array of bool / {0, 1}
        Raw binary predictions (before adjustment).

    Returns
    -------
    np.ndarray
        Adjusted predictions (copy — input is not modified).
    """
    y_true = np.asarray(y_true, dtype=bool)
    y_pred = np.asarray(y_pred, dtype=bool).copy()

    in_segment = False
    segment_start = 0
    segment_hit = False

    for i in range(len(y_true)):
        if y_true[i]:
            if not in_segment:
                # entering a new GT segment
                in_segment = True
                segment_start = i
                segment_hit = False
            if y_pred[i]:
                segment_hit = True
        else:
            if in_segment:
                # leaving a GT segment — apply adjustment
                if segment_hit:
                    y_pred[segment_start:i] = True
                in_segment = False

    # close final segment if it extends to the end
    if in_segment and segment_hit:
        y_pred[segment_start : len(y_true)] = True

    return y_pred


def compute_point_adjust_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
) -> PointAdjustMetrics:
    """Compute precision / recall / F1 using the point-adjust protocol.

    This is the standard metric used by OmniAnomaly, TranAD, MTAD-GAT and
    many other time-series anomaly detection papers.  Note that it
    significantly inflates recall compared to point-level evaluation.
    """
    y_true_aligned, y_pred_aligned = _align_series(y_true, y_pred)
    adjusted = point_adjust(y_true_aligned.values, y_pred_aligned.values)

    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true_aligned.astype(int).values,
        adjusted.astype(int),
        average="binary",
        pos_label=1,
        zero_division=0,
    )
    return PointAdjustMetrics(
        precision=float(prec),
        recall=float(rec),
        f1=float(f1),
    )


# ---------------------------------------------------------------------------
# Best-F1 threshold search
# ---------------------------------------------------------------------------


@dataclass
class BestF1Result:
    """Result of a best-F1 threshold sweep."""

    f1: float
    precision: float
    recall: float
    threshold: float
    pa_f1: float  # point-adjust F1 at the same threshold


def compute_best_f1(
    y_true: pd.Series,
    scores: pd.Series,
    n_thresholds: int = 100,
) -> BestF1Result:
    """Find the threshold that maximises point-level F1.

    Sweeps *n_thresholds* evenly spaced quantiles between the 90th and
    100th percentile of *scores* (the interesting region for anomaly
    detection).  For each threshold ``t`` the predictions are
    ``scores >= t``; we pick the ``t`` with the highest F1.

    Also reports the PA-F1 at that same threshold for reference.
    """
    y_true_al, scores_al = _align_series(y_true, scores)
    y_true_arr = y_true_al.astype(int).values
    scores_arr = scores_al.values.astype(float)

    if y_true_arr.sum() == 0 or y_true_arr.sum() == len(y_true_arr):
        return BestF1Result(f1=0.0, precision=0.0, recall=0.0, threshold=0.0, pa_f1=0.0)

    lo = float(np.percentile(scores_arr, 90))
    hi = float(np.nanmax(scores_arr))
    if lo >= hi:
        lo = float(np.nanmin(scores_arr))
    thresholds = np.linspace(lo, hi, n_thresholds)

    best_f1 = -1.0
    best_prec = 0.0
    best_rec = 0.0
    best_thr = 0.0

    for thr in thresholds:
        preds = (scores_arr >= thr).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(
            y_true_arr, preds, average="binary", pos_label=1, zero_division=0,
        )
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr)

    # Recompute full metrics at best threshold
    best_preds = (scores_arr >= best_thr).astype(int)
    best_prec, best_rec, best_f1, _ = precision_recall_fscore_support(
        y_true_arr, best_preds, average="binary", pos_label=1, zero_division=0,
    )

    # PA-F1 at same threshold
    adjusted = point_adjust(y_true_arr, best_preds.astype(bool))
    pa_prec, pa_rec, pa_f1, _ = precision_recall_fscore_support(
        y_true_arr, adjusted.astype(int), average="binary", pos_label=1, zero_division=0,
    )

    return BestF1Result(
        f1=float(best_f1),
        precision=float(best_prec),
        recall=float(best_rec),
        threshold=best_thr,
        pa_f1=float(pa_f1),
    )


# ---------------------------------------------------------------------------
# Detection latency
# ---------------------------------------------------------------------------


@dataclass
class LatencyResult:
    """Detection latency statistics across anomaly segments."""

    mean_latency: float
    median_latency: float
    n_segments: int
    n_detected: int
    n_missed: int


def compute_detection_latency(
    y_true: pd.Series,
    y_pred: pd.Series,
) -> LatencyResult:
    """Compute how many timesteps into each GT anomaly segment the first detection occurs.

    For each contiguous ground-truth anomaly segment, the *latency* is the
    offset (in timesteps) from the segment start to the first predicted
    ``True`` within it.  Segments with no detection contribute
    ``segment_length`` to the latency (worst case).

    Returns aggregate statistics across all segments.
    """
    y_true_al, y_pred_al = _align_series(y_true, y_pred)
    gt_bool = y_true_al.astype(bool).values
    pred_bool = y_pred_al.astype(bool).values

    # Find GT segments (start, length)
    segments: list[tuple[int, int]] = []
    i = 0
    while i < len(gt_bool):
        if gt_bool[i]:
            start = i
            while i < len(gt_bool) and gt_bool[i]:
                i += 1
            segments.append((start, i - start))
        else:
            i += 1

    if not segments:
        return LatencyResult(
            mean_latency=0.0, median_latency=0.0,
            n_segments=0, n_detected=0, n_missed=0,
        )

    latencies: list[int] = []
    n_detected = 0
    n_missed = 0

    for seg_start, seg_len in segments:
        # Find first detection within segment
        found = False
        for offset in range(seg_len):
            if pred_bool[seg_start + offset]:
                latencies.append(offset)
                n_detected += 1
                found = True
                break
        if not found:
            latencies.append(seg_len)  # worst case: full segment missed
            n_missed += 1

    arr = np.array(latencies, dtype=float)
    return LatencyResult(
        mean_latency=float(arr.mean()),
        median_latency=float(np.median(arr)),
        n_segments=len(segments),
        n_detected=n_detected,
        n_missed=n_missed,
    )


def summarize_anomalies(
    name: str,
    y_test: pd.Series,  # actual values
    anomalies: pd.Series,  # predicted labels (bool / {0,1})
    scores: pd.Series,  # anomaly scores (higher = more anomalous)
    y_true_labels: pd.Series | None = None,  # ground-truth labels, same index
    top_n: int = 10,
) -> tuple[PointMetrics, RangeMetrics | None] | None:
    """
    Print basic summary, top-N anomalous points, and (optionally)
    point- and range-based evaluation metrics if ground-truth labels
    are provided.

    Returns (PointMetrics, RangeMetrics) when labels are given, None otherwise.
    """
    assert (y_test.index == anomalies.index).all()
    assert (y_test.index == scores.index).all()

    n_total = len(y_test)
    n_anom = anomalies.astype(bool).sum()

    print(f"=== {name} ===")
    print(f"Total test points: {n_total}")
    print(f"Flagged anomalies: {n_anom} ({n_anom / n_total:.1%})")

    if n_anom > 0:
        df = pd.DataFrame(
            {
                "value": y_test,
                "score": scores,
                "is_anomaly": anomalies.astype(bool),
            }
        )
        top = df[df["is_anomaly"]].sort_values("score", ascending=False).head(top_n)
        print("\nTop anomalous points:")
        print(top[["value", "score"]])
    print()

    # If we have ground-truth labels, compute metrics
    if y_true_labels is not None:
        # Point metrics + AUROC / AUPRC
        pm = compute_point_metrics(
            y_true=y_true_labels,
            y_pred=anomalies,
            scores=scores,
        )
        print("Point metrics:")
        print(f"  precision: {pm.precision:.4f}")
        print(f"  recall:    {pm.recall:.4f}")
        print(f"  f1:        {pm.f1:.4f}")
        if pm.auc_roc is not None:
            print(f"  auc_roc:   {pm.auc_roc:.4f}")
        if pm.auc_pr is not None:
            print(f"  auc_pr:    {pm.auc_pr:.4f}")

        # Range-level F1
        rm = compute_range_f1_from_labels(
            y_true=y_true_labels,
            y_pred=anomalies,
        )
        print("\nRange metrics:")
        print(f"  precision: {rm.precision:.4f}")
        print(f"  recall:    {rm.recall:.4f}")
        print(f"  f1:        {rm.f1:.4f}")
        print(f"  gt_ranges: {rm.n_gt_ranges}")
        print(f"  pred_ranges: {rm.n_pred_ranges}")
        print(f"  tp_ranges: {rm.n_tp_ranges}")

    print()

    if y_true_labels is not None:
        return (pm, rm)
    return None
