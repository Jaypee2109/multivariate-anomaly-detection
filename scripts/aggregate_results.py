"""Aggregate multivariate results across all SMD machines.

Reads per-machine result CSVs from artifacts/multivariate/,
computes evaluation metrics per machine per model, and outputs:

1. artifacts/multivariate/metrics_per_machine.csv
   - One row per (machine, model) with all metrics
2. artifacts/multivariate/metrics_average.csv
   - One row per model with mean +/- std across machines
3. artifacts/multivariate/all_machines_results.csv
   - Concatenated raw results (scores + predictions) so the
     dashboard can visualize pooled metrics via the machine dropdown
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

# Ensure src is importable
_project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_project_root / "src"))

from time_series_transformer.config import ARTIFACTS_DIR
from time_series_transformer.evaluation import (
    compute_best_f1,
    compute_detection_latency,
    compute_point_adjust_metrics,
)

RESULTS_DIR = ARTIFACTS_DIR / "multivariate"


def discover_machines() -> list[str]:
    """Find all machine result CSVs (excluding the aggregate itself)."""
    if not RESULTS_DIR.exists():
        return []
    machines = []
    for p in sorted(RESULTS_DIR.glob("*_results.csv")):
        name = p.stem.replace("_results", "")
        if name in ("all_machines", "all_machines_avg"):
            continue
        machines.append(name)
    return machines


def discover_models(df: pd.DataFrame) -> list[str]:
    """Extract model names from column pattern {model}_score + {model}_is_anomaly."""
    models = []
    for col in df.columns:
        if col.endswith("_score"):
            name = col[: -len("_score")]
            if f"{name}_is_anomaly" in df.columns:
                models.append(name)
    return sorted(models)


def compute_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
    scores: pd.Series,
) -> dict[str, float]:
    """Compute all evaluation metrics for one model on one machine."""
    y_true_arr = y_true.astype(int).values
    y_pred_arr = y_pred.astype(int).values

    # Point metrics
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true_arr, y_pred_arr, average="binary", zero_division=0,
    )

    # AUC-ROC
    auc_roc = np.nan
    if len(np.unique(y_true_arr)) > 1:
        try:
            auc_roc = roc_auc_score(y_true_arr, scores.values)
        except ValueError:
            pass

    # AUC-PR
    auc_pr = np.nan
    if y_true_arr.sum() > 0:
        try:
            auc_pr = average_precision_score(y_true_arr, scores.values)
        except ValueError:
            pass

    # Point-adjust F1
    pa = compute_point_adjust_metrics(y_true, y_pred)

    # Best-F1 (oracle threshold)
    bf = compute_best_f1(y_true, scores, n_thresholds=50)

    # Detection latency
    dl = compute_detection_latency(y_true, y_pred)

    return {
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "auc_roc": float(auc_roc),
        "auc_pr": float(auc_pr),
        "pa_precision": float(pa.precision),
        "pa_recall": float(pa.recall),
        "pa_f1": float(pa.f1),
        "best_f1": float(bf.f1),
        "best_f1_threshold": float(bf.threshold),
        "best_f1_pa_f1": float(bf.pa_f1),
        "mean_latency": float(dl.mean_latency),
        "median_latency": float(dl.median_latency),
        "n_segments": int(dl.n_segments),
        "n_detected": int(dl.n_detected),
        "n_missed": int(dl.n_missed),
        "detection_rate": float(dl.n_detected / dl.n_segments) if dl.n_segments > 0 else 0.0,
    }


def main() -> None:
    machines = discover_machines()
    if not machines:
        print("No result CSVs found in", RESULTS_DIR)
        return

    print(f"Found {len(machines)} machines")

    # --- 1. Compute per-machine metrics ---
    all_rows: list[dict] = []
    concat_frames: list[pd.DataFrame] = []

    for machine_id in machines:
        path = RESULTS_DIR / f"{machine_id}_results.csv"
        df = pd.read_csv(path)

        if "is_anomaly" not in df.columns:
            print(f"  {machine_id}: no ground truth column, skipping")
            continue

        models = discover_models(df)
        if not models:
            print(f"  {machine_id}: no model results, skipping")
            continue

        y_true = pd.Series(df["is_anomaly"].astype(int).values)

        for model in models:
            score_col = f"{model}_score"
            anom_col = f"{model}_is_anomaly"

            y_pred = pd.Series(df[anom_col].astype(int).values)
            scores = pd.Series(
                pd.to_numeric(df[score_col], errors="coerce").fillna(0).values
            )

            metrics = compute_metrics(y_true, y_pred, scores)
            all_rows.append({"machine": machine_id, "model": model, **metrics})

        concat_frames.append(df)
        print(f"  {machine_id}: {len(models)} models")

    if not all_rows:
        print("No metrics computed.")
        return

    # --- Save per-machine metrics ---
    per_machine_df = pd.DataFrame(all_rows)
    per_machine_path = RESULTS_DIR / "metrics_per_machine.csv"
    per_machine_df.to_csv(per_machine_path, index=False)
    print(f"\nSaved per-machine metrics: {per_machine_path}")
    print(f"  {len(per_machine_df)} rows ({len(machines)} machines x models)")

    # --- 2. Compute average metrics ---
    metric_cols = [
        "precision", "recall", "f1", "auc_roc", "auc_pr",
        "pa_precision", "pa_recall", "pa_f1",
        "best_f1", "best_f1_pa_f1",
        "mean_latency", "median_latency", "detection_rate",
    ]

    avg_rows: list[dict] = []
    for model, group in per_machine_df.groupby("model"):
        row: dict[str, object] = {"model": model, "n_machines": len(group)}
        for col in metric_cols:
            vals = group[col].dropna()
            row[f"{col}_mean"] = float(vals.mean()) if len(vals) > 0 else np.nan
            row[f"{col}_std"] = float(vals.std()) if len(vals) > 1 else 0.0
        # Segment detection: sum across machines
        row["total_segments"] = int(group["n_segments"].sum())
        row["total_detected"] = int(group["n_detected"].sum())
        row["total_missed"] = int(group["n_missed"].sum())
        avg_rows.append(row)

    avg_df = pd.DataFrame(avg_rows)
    avg_path = RESULTS_DIR / "metrics_average.csv"
    avg_df.to_csv(avg_path, index=False)
    print(f"Saved average metrics: {avg_path}")

    # --- Print summary table ---
    print("\n" + "=" * 80)
    print("AVERAGE METRICS ACROSS ALL MACHINES")
    print("=" * 80)
    for _, row in avg_df.iterrows():
        print(f"\n  {row['model']} ({int(row['n_machines'])} machines)")
        print(f"    AUC-ROC:    {row['auc_roc_mean']:.4f} +/- {row['auc_roc_std']:.4f}")
        print(f"    PA-F1:      {row['pa_f1_mean']:.4f} +/- {row['pa_f1_std']:.4f}")
        print(f"    F1:         {row['f1_mean']:.4f} +/- {row['f1_std']:.4f}")
        print(f"    Best-F1:    {row['best_f1_mean']:.4f} +/- {row['best_f1_std']:.4f}")
        print(f"    Precision:  {row['precision_mean']:.4f} +/- {row['precision_std']:.4f}")
        print(f"    Recall:     {row['recall_mean']:.4f} +/- {row['recall_std']:.4f}")
        print(f"    Latency:    {row['mean_latency_mean']:.1f} +/- {row['mean_latency_std']:.1f}")
        det = row["total_detected"]
        seg = row["total_segments"]
        print(f"    Segments:   {det}/{seg} ({det / seg * 100:.1f}%)" if seg > 0 else "    Segments:   N/A")

    # --- 3. Best machine per model (highest AUC-ROC) ---
    best_rows: list[dict] = []
    for model, group in per_machine_df.groupby("model"):
        best_idx = group["auc_roc"].idxmax()
        best = group.loc[best_idx]
        best_rows.append(best.to_dict())

    best_df = pd.DataFrame(best_rows)
    best_path = RESULTS_DIR / "metrics_best_machine.csv"
    best_df.to_csv(best_path, index=False)
    print(f"Saved best-machine metrics: {best_path}")

    print("\n" + "-" * 80)
    print("BEST MACHINE PER MODEL (by AUC-ROC)")
    print("-" * 80)
    for _, row in best_df.iterrows():
        print(
            f"  {row['model']:30s}  {row['machine']:15s}  "
            f"AUC-ROC={row['auc_roc']:.4f}  PA-F1={row['pa_f1']:.4f}  "
            f"Seg={int(row['n_detected'])}/{int(row['n_segments'])}"
        )

    # --- 4. Concatenated results for dashboard ---
    if concat_frames:
        all_df = pd.concat(concat_frames, ignore_index=True)
        all_path = RESULTS_DIR / "all_machines_results.csv"
        all_df.to_csv(all_path, index=False)
        print(f"\nSaved concatenated results for dashboard: {all_path}")
        print(f"  {len(all_df)} total rows")
        print("  -> Select 'all_machines' in the dashboard machine dropdown")


if __name__ == "__main__":
    main()
