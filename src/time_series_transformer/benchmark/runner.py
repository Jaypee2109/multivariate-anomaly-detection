"""Core benchmark runner — iterate models x datasets, collect metrics."""

from __future__ import annotations

import logging
import time
from contextlib import nullcontext

import numpy as np
import torch

from time_series_transformer.config import RANDOM_STATE, TRAIN_RATIO
from time_series_transformer.data_pipeline.data_loading import load_timeseries
from time_series_transformer.data_pipeline.labels import (
    load_label_times,
    make_point_labels_from_times,
)
from time_series_transformer.data_pipeline.preprocessing import load_csv_to_df
from time_series_transformer.evaluation import (
    compute_point_metrics,
    compute_range_f1_from_labels,
)
from time_series_transformer.split import train_test_split_series

from .dataset_spec import DatasetSpec, MultivariateDatasetSpec
from .registry import get_factory, is_multivariate, list_models
from .results import BenchmarkResult, ResultsCollector

logger = logging.getLogger(__name__)


def _seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class BenchmarkRunner:
    """Run registered models across multiple datasets and collect results.

    Parameters
    ----------
    datasets : list[DatasetSpec | MultivariateDatasetSpec]
        Datasets to evaluate on.  Univariate models run on
        :class:`DatasetSpec`, multivariate models run on
        :class:`MultivariateDatasetSpec`.
    model_names : list[str] | None
        Subset of registered model names (default: all).
    log_to_mlflow : bool
        If *True*, log each run to MLflow.
    train_ratio : float
        Fraction of data used for training (time-ordered split).
    random_state : int
        RNG seed for reproducibility.
    """

    def __init__(
        self,
        datasets: list[DatasetSpec | MultivariateDatasetSpec],
        model_names: list[str] | None = None,
        log_to_mlflow: bool = False,
        train_ratio: float = TRAIN_RATIO,
        random_state: int = RANDOM_STATE,
    ) -> None:
        self.datasets = datasets
        self.train_ratio = train_ratio
        self.random_state = random_state
        self.log_to_mlflow = log_to_mlflow

        # Resolve model names
        available = set(list_models())
        if model_names is None:
            self.model_names = list_models()
        else:
            self.model_names = [m for m in model_names if m in available]
            missing = set(model_names) - set(self.model_names)
            if missing:
                logger.warning("Skipping unregistered models: %s", missing)

        self.collector = ResultsCollector()

        # MLflow setup (lazy — only if requested)
        self._mlflow = None
        self._mlflow_fns: dict = {}
        if self.log_to_mlflow:
            try:
                import mlflow

                from time_series_transformer.mlflow_utils import (
                    log_anomaly_summary,
                    log_data_hash,
                    log_environment_info,
                    log_params_from_model,
                    log_point_metrics,
                    log_range_metrics,
                    setup_mlflow,
                )

                setup_mlflow()
                self._mlflow = mlflow
                self._mlflow_fns = {
                    "env": log_environment_info,
                    "hash": log_data_hash,
                    "params": log_params_from_model,
                    "point": log_point_metrics,
                    "range": log_range_metrics,
                    "summary": log_anomaly_summary,
                }
            except ImportError:
                logger.warning("mlflow not installed — disabling tracking.")
                self.log_to_mlflow = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> ResultsCollector:
        """Execute the full benchmark and return collected results."""
        _seed_everything(self.random_state)

        # Build list of (dataset, model_name) pairs, matching model type to
        # dataset type (univariate ↔ DatasetSpec, multivariate ↔ MultivariateDatasetSpec).
        pairs: list[tuple] = []
        for ds in self.datasets:
            if isinstance(ds, MultivariateDatasetSpec):
                applicable = [m for m in self.model_names if is_multivariate(m)]
            else:
                applicable = [m for m in self.model_names if not is_multivariate(m)]
            for model_name in applicable:
                pairs.append((ds, model_name))

        total = len(pairs)
        logger.info(
            "Starting benchmark: %d dataset(s), %d run(s) total",
            len(self.datasets),
            total,
        )

        for idx, (ds, model_name) in enumerate(pairs, 1):
            logger.info("[%d/%d] %s on %s", idx, total, model_name, ds.name)
            if isinstance(ds, MultivariateDatasetSpec):
                result = self._run_single_multivariate(ds, model_name)
            else:
                result = self._run_single(ds, model_name)
            self.collector.add(result)

        ok = sum(r.success for r in self.collector.results)
        logger.info("Benchmark complete: %d/%d succeeded", ok, total)
        return self.collector

    # ------------------------------------------------------------------
    # Single run
    # ------------------------------------------------------------------

    def _run_single(self, ds: DatasetSpec, model_name: str) -> BenchmarkResult:
        try:
            # 1. Load data + split
            y = load_timeseries(ds.csv_path)
            y_train, y_test = train_test_split_series(y, train_ratio=self.train_ratio)

            # 2. Load labels (if available)
            y_true_labels = self._load_labels(ds, y)

            # 3. Build model
            model = get_factory(model_name)()

            # 4. MLflow context
            ctx = (
                self._mlflow.start_run(run_name=f"{model_name} — {ds.name}")
                if self._mlflow
                else nullcontext()
            )

            with ctx:
                self._log_mlflow_params(ds, model_name, model, y_train, y_test)

                # Fit
                t0 = time.time()
                model.fit(y_train)
                fit_time = time.time() - t0

                # Predict
                t0 = time.time()
                scores = model.decision_function(y_test)
                anomalies = model.predict(y_test)
                predict_time = time.time() - t0

                # Metrics
                n_test = len(y_test)
                n_anom = int(anomalies.astype(bool).sum())
                anom_rate = n_anom / n_test if n_test > 0 else 0.0

                pm, rm = None, None
                if y_true_labels is not None:
                    pm = compute_point_metrics(
                        y_true=y_true_labels, y_pred=anomalies, scores=scores,
                    )
                    rm = compute_range_f1_from_labels(
                        y_true=y_true_labels, y_pred=anomalies,
                    )

                self._log_mlflow_metrics(fit_time, predict_time, n_test, n_anom, pm, rm)

                return BenchmarkResult(
                    dataset_name=ds.name,
                    model_name=model_name,
                    success=True,
                    fit_time_seconds=fit_time,
                    predict_time_seconds=predict_time,
                    test_size=n_test,
                    n_anomalies_flagged=n_anom,
                    anomaly_rate=anom_rate,
                    point_precision=pm.precision if pm else None,
                    point_recall=pm.recall if pm else None,
                    point_f1=pm.f1 if pm else None,
                    point_auc_roc=pm.auc_roc if pm else None,
                    point_auc_pr=pm.auc_pr if pm else None,
                    range_precision=rm.precision if rm else None,
                    range_recall=rm.recall if rm else None,
                    range_f1=rm.f1 if rm else None,
                    n_gt_ranges=rm.n_gt_ranges if rm else None,
                    n_pred_ranges=rm.n_pred_ranges if rm else None,
                    n_tp_ranges=rm.n_tp_ranges if rm else None,
                )

        except Exception as e:
            logger.error("FAILED %s on %s: %s", model_name, ds.name, e, exc_info=True)
            return BenchmarkResult(
                dataset_name=ds.name,
                model_name=model_name,
                success=False,
                error_message=str(e),
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_labels(self, ds: DatasetSpec, y: "pd.Series") -> "pd.Series | None":
        if not ds.has_labels():
            return None
        import pandas as pd

        df = load_csv_to_df(ds.csv_path, parse_dates=["timestamp"])
        df = df.set_index("timestamp").sort_index()

        label_times = load_label_times(ds.labels_path, ds.labels_key)
        y_true_all = make_point_labels_from_times(
            df.reset_index(), label_times, timestamp_col="timestamp",
        )
        y_true_all.index = df.index

        _, y_true_test = train_test_split_series(y_true_all, train_ratio=self.train_ratio)
        return y_true_test

    def _log_mlflow_params(self, ds, model_name, model, y_train, y_test) -> None:
        if not self._mlflow:
            return
        self._mlflow_fns["env"]()
        self._mlflow_fns["hash"](ds.csv_path)
        self._mlflow.log_params({
            "dataset": ds.name,
            "train_ratio": self.train_ratio,
            "random_state": self.random_state,
            "train_size": len(y_train),
            "test_size": len(y_test),
        })
        self._mlflow_fns["params"](model_name, model)

    def _run_single_multivariate(
        self, ds: MultivariateDatasetSpec, model_name: str,
    ) -> BenchmarkResult:
        try:
            from time_series_transformer.data_pipeline.smd_loading import (
                load_smd_machine,
            )

            machine_data = load_smd_machine(
                ds.machine_id, base_dir=ds.base_dir, normalize=ds.normalize,
            )
            X_train = machine_data.train_df
            X_test = machine_data.test_df
            y_true = machine_data.test_labels

            model = get_factory(model_name)()

            ctx = (
                self._mlflow.start_run(run_name=f"{model_name} — {ds.name}")
                if self._mlflow
                else nullcontext()
            )

            with ctx:
                t0 = time.time()
                model.fit(X_train)
                fit_time = time.time() - t0

                t0 = time.time()
                scores = model.decision_function(X_test)
                anomalies = model.predict(X_test)
                predict_time = time.time() - t0

                n_test = len(X_test)
                n_anom = int(anomalies.astype(bool).sum())
                anom_rate = n_anom / n_test if n_test > 0 else 0.0

                pm = compute_point_metrics(
                    y_true=y_true, y_pred=anomalies, scores=scores,
                )
                rm = compute_range_f1_from_labels(
                    y_true=y_true, y_pred=anomalies,
                )

                self._log_mlflow_metrics(
                    fit_time, predict_time, n_test, n_anom, pm, rm,
                )

                return BenchmarkResult(
                    dataset_name=ds.name,
                    model_name=model_name,
                    success=True,
                    fit_time_seconds=fit_time,
                    predict_time_seconds=predict_time,
                    test_size=n_test,
                    n_anomalies_flagged=n_anom,
                    anomaly_rate=anom_rate,
                    point_precision=pm.precision,
                    point_recall=pm.recall,
                    point_f1=pm.f1,
                    point_auc_roc=pm.auc_roc,
                    point_auc_pr=pm.auc_pr,
                    range_precision=rm.precision,
                    range_recall=rm.recall,
                    range_f1=rm.f1,
                    n_gt_ranges=rm.n_gt_ranges,
                    n_pred_ranges=rm.n_pred_ranges,
                    n_tp_ranges=rm.n_tp_ranges,
                )

        except Exception as e:
            logger.error(
                "FAILED %s on %s: %s", model_name, ds.name, e, exc_info=True,
            )
            return BenchmarkResult(
                dataset_name=ds.name,
                model_name=model_name,
                success=False,
                error_message=str(e),
            )

    def _log_mlflow_metrics(self, fit_time, predict_time, n_test, n_anom, pm, rm) -> None:
        if not self._mlflow:
            return
        self._mlflow.log_metric("fit_time_seconds", fit_time)
        self._mlflow.log_metric("predict_time_seconds", predict_time)
        self._mlflow_fns["summary"](n_test, n_anom)
        if pm is not None:
            self._mlflow_fns["point"](pm)
        if rm is not None:
            self._mlflow_fns["range"](rm)
