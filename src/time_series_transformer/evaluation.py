from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
)


@dataclass
class PointMetrics:
    precision: float
    recall: float
    f1: float
    auc_roc: Optional[float] = None
    auc_pr: Optional[float] = None


def _align_series(
    y_true: pd.Series,
    y_pred: pd.Series,
) -> Tuple[pd.Series, pd.Series]:
    """
    Align two Series on a common index (inner join).
    """
    y_true_aligned, y_pred_aligned = y_true.align(y_pred, join="inner")
    return y_true_aligned, y_pred_aligned


def labels_to_ranges(labels: pd.Series) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Convert a boolean / {0,1} Series (indexed by timestamp) into
    a list of contiguous anomaly ranges (start, end).

    Contiguous = consecutive True values in the Series.
    """
    if not isinstance(labels.index, pd.DatetimeIndex):
        raise TypeError("labels must have a DatetimeIndex.")

    labels_bool = labels.astype(bool)

    ranges: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    in_range = False
    start: Optional[pd.Timestamp] = None
    prev_t: Optional[pd.Timestamp] = None

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


def _ranges_overlap(
    r1: Tuple[pd.Timestamp, pd.Timestamp],
    r2: Tuple[pd.Timestamp, pd.Timestamp],
) -> bool:
    s1, e1 = r1
    s2, e2 = r2
    return max(s1, s2) <= min(e1, e2)


def compute_point_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
    scores: Optional[pd.Series] = None,
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

    precision = (
        tp_ranges / (tp_ranges + fp_ranges) if (tp_ranges + fp_ranges) > 0 else 0.0
    )
    recall = tp_ranges / (tp_ranges + fn_ranges) if (tp_ranges + fn_ranges) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return RangeMetrics(
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        n_gt_ranges=n_gt,
        n_pred_ranges=n_pred,
        n_tp_ranges=tp_ranges,
    )


def summarize_anomalies(
    name: str,
    y_test: pd.Series,  # actual values
    anomalies: pd.Series,  # predicted labels (bool / {0,1})
    scores: pd.Series,  # anomaly scores (higher = more anomalous)
    y_true_labels: Optional[pd.Series] = None,  # ground-truth labels, same index
    top_n: int = 10,
) -> None:
    """
    Print basic summary, top-N anomalous points, and (optionally)
    point- and range-based evaluation metrics if ground-truth labels
    are provided.
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
        # Note: y_true_labels must have a DatetimeIndex for labels_to_ranges()
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
