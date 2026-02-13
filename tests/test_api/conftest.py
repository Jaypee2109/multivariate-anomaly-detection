"""Fixtures for API tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from time_series_transformer.api.inference_server import app, manager
from time_series_transformer.models.baseline.rolling_zscore import RollingZScoreAnomalyDetector


@pytest.fixture()
def sample_request_data() -> dict:
    """100 hourly data points shaped as an API request dict."""
    timestamps = pd.date_range("2020-01-01", periods=100, freq="h")
    rng = np.random.default_rng(42)
    values = np.sin(np.linspace(0, 4 * np.pi, 100)) + rng.normal(0, 0.1, 100)
    return {
        "timestamps": [t.isoformat() for t in timestamps],
        "values": values.tolist(),
    }


@pytest.fixture()
def loaded_client(sample_timeseries) -> TestClient:
    """TestClient with a Rolling Z-Score model pre-loaded (fast, no torch)."""
    model = RollingZScoreAnomalyDetector(window=12, z_thresh=2.0)
    model.fit(sample_timeseries)
    manager._models["rolling_zscore"] = model
    yield TestClient(app)
    manager._models.clear()


@pytest.fixture()
def empty_client() -> TestClient:
    """TestClient with no models loaded."""
    manager._models.clear()
    yield TestClient(app)
