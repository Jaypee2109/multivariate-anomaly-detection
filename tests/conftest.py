"""Shared fixtures for time_series_transformer tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture()
def sample_timestamps() -> pd.DatetimeIndex:
    return pd.date_range("2020-01-01", periods=100, freq="h")


@pytest.fixture()
def sample_timeseries(sample_timestamps: pd.DatetimeIndex) -> pd.Series:
    """100 hourly values with a simple sine pattern."""
    rng = np.random.default_rng(42)
    values = np.sin(np.linspace(0, 4 * np.pi, 100)) + rng.normal(0, 0.1, 100)
    return pd.Series(values, index=sample_timestamps, name="value")


@pytest.fixture()
def sample_labels(sample_timestamps: pd.DatetimeIndex) -> pd.Series:
    """Boolean labels — marks indices 20-24 and 60-62 as anomalies."""
    labels = pd.Series(False, index=sample_timestamps, dtype=bool)
    labels.iloc[20:25] = True
    labels.iloc[60:63] = True
    return labels


@pytest.fixture()
def tmp_csv(tmp_path, sample_timeseries: pd.Series) -> pd.core.generic.NDFrame:
    """Write sample timeseries to a CSV and return the path."""
    csv_path = tmp_path / "test_data.csv"
    df = pd.DataFrame({"timestamp": sample_timeseries.index, "value": sample_timeseries.values})
    df.to_csv(csv_path, index=False)
    return csv_path


# ---- multivariate fixtures ------------------------------------------------


@pytest.fixture()
def sample_multivariate_df() -> pd.DataFrame:
    """100 rows x 5 features with sine patterns + noise."""
    rng = np.random.default_rng(42)
    n, k = 100, 5
    data = np.column_stack(
        [np.sin(np.linspace(0, 4 * np.pi, n) + i) + rng.normal(0, 0.1, n) for i in range(k)]
    )
    return pd.DataFrame(data, columns=[f"f{i}" for i in range(k)])


@pytest.fixture()
def sample_integer_labels() -> pd.Series:
    """Boolean labels with integer index — marks rows 20-24 and 60-62."""
    labels = pd.Series(False, index=range(100), dtype=bool)
    labels.iloc[20:25] = True
    labels.iloc[60:63] = True
    return labels
