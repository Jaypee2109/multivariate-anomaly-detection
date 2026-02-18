"""Benchmark result collection and export utilities."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd


@dataclass
class BenchmarkResult:
    """Metrics for a single model x dataset run."""

    dataset_name: str
    model_name: str
    success: bool
    error_message: str | None = None

    # Timing
    fit_time_seconds: float | None = None
    predict_time_seconds: float | None = None

    # Point metrics
    point_precision: float | None = None
    point_recall: float | None = None
    point_f1: float | None = None
    point_auc_roc: float | None = None
    point_auc_pr: float | None = None

    # Range metrics
    range_precision: float | None = None
    range_recall: float | None = None
    range_f1: float | None = None
    n_gt_ranges: int | None = None
    n_pred_ranges: int | None = None
    n_tp_ranges: int | None = None

    # Basic stats
    test_size: int | None = None
    n_anomalies_flagged: int | None = None
    anomaly_rate: float | None = None


class ResultsCollector:
    """Accumulate :class:`BenchmarkResult` instances and export them."""

    def __init__(self) -> None:
        self.results: list[BenchmarkResult] = []

    def add(self, result: BenchmarkResult) -> None:
        self.results.append(result)

    # ------------------------------------------------------------------
    # Export helpers
    # ------------------------------------------------------------------

    def to_dataframe(self) -> pd.DataFrame:
        if not self.results:
            return pd.DataFrame()
        return pd.DataFrame([asdict(r) for r in self.results])

    def to_csv(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.to_dataframe().to_csv(path, index=False)

    def print_summary(self) -> None:
        """Pretty-print key metrics to the console."""
        df = self.to_dataframe()
        if df.empty:
            print("No benchmark results.")
            return

        key_cols = [
            "dataset_name",
            "model_name",
            "success",
            "point_f1",
            "range_f1",
            "point_auc_roc",
            "fit_time_seconds",
            "anomaly_rate",
        ]
        cols = [c for c in key_cols if c in df.columns]
        display = df[cols].copy()

        # Format floats
        for col in display.columns:
            if display[col].dtype == float:
                display[col] = display[col].map(lambda x: f"{x:.4f}" if pd.notna(x) else "-")

        sep = "=" * 88
        print(f"\n{sep}")
        print("BENCHMARK RESULTS")
        print(sep)
        print(display.to_string(index=False))
        print()
