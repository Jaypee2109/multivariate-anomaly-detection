"""Benchmark framework — run models across datasets and collect metrics."""

from .dataset_spec import DatasetSpec, MultivariateDatasetSpec
from .registry import get_factory, list_models, register_model
from .results import BenchmarkResult, ResultsCollector
from .runner import BenchmarkRunner

__all__ = [
    "BenchmarkResult",
    "BenchmarkRunner",
    "DatasetSpec",
    "MultivariateDatasetSpec",
    "ResultsCollector",
    "get_factory",
    "list_models",
    "register_model",
]
