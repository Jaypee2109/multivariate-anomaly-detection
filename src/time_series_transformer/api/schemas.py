"""Pydantic schemas for the anomaly detection inference API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    status: str
    models_loaded: list[str]
    mv_machines: list[str] = []
    timestamp: str


# ---------------------------------------------------------------------------
# Model info
# ---------------------------------------------------------------------------


class ModelDetail(BaseModel):
    name: str
    display_name: str
    model_class: str
    parameters: dict[str, Any]


class ModelsInfoResponse(BaseModel):
    models: list[ModelDetail]
    mv_machines: list[str] = []
    checkpoint_dir: str


# ---------------------------------------------------------------------------
# Detection request / response
# ---------------------------------------------------------------------------


class TimeSeriesRequest(BaseModel):
    """Input time series for anomaly detection."""

    timestamps: list[str] = Field(..., description="ISO-8601 timestamps")
    values: list[float] = Field(..., description="Observed values, same length as timestamps")


class DetectRequest(BaseModel):
    """Request body for POST /detect."""

    data: TimeSeriesRequest
    models: list[str] | None = Field(
        None, description="Which loaded models to run (default: all loaded)"
    )


class AnomalyPoint(BaseModel):
    timestamp: str
    value: float
    score: float
    is_anomaly: bool


class SingleModelResult(BaseModel):
    model: str
    display_name: str
    anomaly_count: int
    anomaly_ratio: float
    points: list[AnomalyPoint]
    latency_ms: float


class DetectResponse(BaseModel):
    results: list[SingleModelResult]
    metadata: dict[str, Any]


# ---------------------------------------------------------------------------
# Dashboard-optimised (parallel arrays for Plotly / Chart.js)
# ---------------------------------------------------------------------------


class DashboardModelSeries(BaseModel):
    scores: list[float | None]
    anomalies: list[bool]
    threshold: float | None = None


class DashboardChartData(BaseModel):
    timestamps: list[str]
    values: list[float]
    models: dict[str, DashboardModelSeries]


class DashboardDetectResponse(BaseModel):
    chart_data: DashboardChartData
    summary: dict[str, Any]
    latency_ms: float
    timestamp: str
