"""Anomaly Detection Inference API.

FastAPI REST API for serving baseline anomaly detection models.

Usage::

    python -m time_series_transformer serve --checkpoint-dir artifacts/checkpoints
"""

from __future__ import annotations

import io
import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from time_series_transformer.api.model_manager import MODEL_DISPLAY_NAMES, ModelManager
from time_series_transformer.api.schemas import (
    AnomalyPoint,
    DashboardChartData,
    DashboardDetectResponse,
    DashboardModelSeries,
    DetectRequest,
    DetectResponse,
    HealthResponse,
    ModelDetail,
    ModelsInfoResponse,
    SingleModelResult,
)
from time_series_transformer.config import ARTIFACTS_DIR
from time_series_transformer.exceptions import TransformerError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App & global state
# ---------------------------------------------------------------------------

manager = ModelManager()


@asynccontextmanager
async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Load model checkpoints on startup."""
    checkpoint_dir = Path(
        os.environ.get("ANOMALY_CHECKPOINT_DIR", str(ARTIFACTS_DIR / "checkpoints"))
    )
    if checkpoint_dir.exists():
        loaded = manager.load_from_directory(checkpoint_dir)
        logger.info("Loaded %d model(s) from %s: %s", len(loaded), checkpoint_dir, loaded)
    else:
        logger.warning("Checkpoint directory not found: %s", checkpoint_dir)
    yield


app = FastAPI(
    title="Anomaly Detection Inference API",
    description="REST API for time series anomaly detection using baseline models.",
    version="0.1.0",
    lifespan=_lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_series(timestamps: list[str], values: list[float]) -> pd.Series:
    """Convert raw lists into a datetime-indexed pd.Series."""
    index = pd.to_datetime(timestamps)
    return pd.Series(values, index=index, name="value")


def _validate_detect_input(
    timestamps: list[str], values: list[float], min_points: int = 10
) -> None:
    if len(timestamps) != len(values):
        raise HTTPException(
            status_code=400,
            detail=f"timestamps ({len(timestamps)}) and values ({len(values)}) must have the same length",
        )
    if len(timestamps) < min_points:
        raise HTTPException(
            status_code=400,
            detail=f"Need at least {min_points} data points, got {len(timestamps)}",
        )


def _resolve_models(requested: list[str] | None) -> list[str]:
    """Return validated list of model slugs to use."""
    if not manager.loaded_model_names:
        raise HTTPException(status_code=503, detail="No models loaded")
    slugs = requested or manager.loaded_model_names
    unknown = set(slugs) - set(manager.loaded_model_names)
    if unknown:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown/unloaded models: {sorted(unknown)}. "
            f"Available: {manager.loaded_model_names}",
        )
    return slugs


def _run_detection(
    y: pd.Series,
    slugs: list[str],
    timestamps: list[str],
    values: list[float],
) -> tuple[list[SingleModelResult], float]:
    """Run detection and build per-model result objects."""
    try:
        raw = manager.detect(y, model_slugs=slugs)
    except TransformerError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Detection failed: {exc}") from exc

    total_latency = 0.0
    n_input = len(timestamps)
    results: list[SingleModelResult] = []
    for slug, (anomalies, scores, latency) in raw.items():
        total_latency += latency
        # Some models (e.g. Rolling Z-Score) prepend history to output.
        # Align to the last n_input points to match the request.
        sc_vals = scores.values[-n_input:]
        an_vals = anomalies.values[-n_input:]
        points = [
            AnomalyPoint(
                timestamp=ts,
                value=val,
                score=float(sc) if pd.notna(sc) else 0.0,
                is_anomaly=bool(flag),
            )
            for ts, val, sc, flag in zip(timestamps, values, sc_vals, an_vals, strict=True)
        ]
        n_anom = int(an_vals.astype(bool).sum())
        results.append(
            SingleModelResult(
                model=slug,
                display_name=MODEL_DISPLAY_NAMES.get(slug, slug),
                anomaly_count=n_anom,
                anomaly_ratio=n_anom / len(y) if len(y) > 0 else 0.0,
                points=points,
                latency_ms=round(latency, 2),
            )
        )
    return results, total_latency


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    return HealthResponse(
        status="healthy" if manager.loaded_model_names else "no_models_loaded",
        models_loaded=manager.loaded_model_names,
        timestamp=datetime.now().isoformat(),
    )


@app.get("/models", response_model=ModelsInfoResponse)
async def get_models() -> ModelsInfoResponse:
    if not manager.loaded_model_names:
        raise HTTPException(status_code=503, detail="No models loaded")
    details = [ModelDetail(**manager.get_model_info(slug)) for slug in manager.loaded_model_names]
    return ModelsInfoResponse(models=details, checkpoint_dir=manager.checkpoint_dir)


@app.get("/models/{model_name}", response_model=ModelDetail)
async def get_model_detail(model_name: str) -> ModelDetail:
    if model_name not in manager.loaded_model_names:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not loaded")
    return ModelDetail(**manager.get_model_info(model_name))


@app.post("/detect", response_model=DetectResponse)
async def detect_anomalies(request: DetectRequest) -> DetectResponse:
    _validate_detect_input(request.data.timestamps, request.data.values)
    slugs = _resolve_models(request.models)
    y = _build_series(request.data.timestamps, request.data.values)

    results, _ = _run_detection(y, slugs, request.data.timestamps, request.data.values)
    metadata: dict[str, Any] = {
        "input_length": len(y),
        "date_range": {
            "start": request.data.timestamps[0],
            "end": request.data.timestamps[-1],
        },
        "models_used": slugs,
    }
    return DetectResponse(results=results, metadata=metadata)


@app.post("/detect/dashboard", response_model=DashboardDetectResponse)
async def detect_for_dashboard(request: DetectRequest) -> DashboardDetectResponse:
    """Dashboard-optimised: returns parallel arrays ready for Plotly traces."""
    _validate_detect_input(request.data.timestamps, request.data.values)
    slugs = _resolve_models(request.models)
    y = _build_series(request.data.timestamps, request.data.values)

    try:
        raw = manager.detect(y, model_slugs=slugs)
    except TransformerError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Detection failed: {exc}") from exc

    total_latency = 0.0
    n_input = len(request.data.timestamps)
    model_series: dict[str, DashboardModelSeries] = {}
    summary: dict[str, Any] = {}
    for slug, (anomalies, scores, latency) in raw.items():
        total_latency += latency
        sc_vals = scores.values[-n_input:]
        an_vals = anomalies.values[-n_input:]
        n_anom = int(an_vals.astype(bool).sum())
        model_series[slug] = DashboardModelSeries(
            scores=[float(s) if pd.notna(s) else None for s in sc_vals],
            anomalies=[bool(a) for a in an_vals],
        )
        summary[slug] = {
            "anomaly_count": n_anom,
            "anomaly_ratio": n_anom / len(y) if len(y) > 0 else 0.0,
        }

    chart_data = DashboardChartData(
        timestamps=request.data.timestamps,
        values=request.data.values,
        models=model_series,
    )
    return DashboardDetectResponse(
        chart_data=chart_data,
        summary=summary,
        latency_ms=round(total_latency, 2),
        timestamp=datetime.now().isoformat(),
    )


@app.post("/detect/csv", response_model=DetectResponse)
async def detect_from_csv(
    file: Annotated[UploadFile, File()],
    models: Annotated[str | None, Form()] = None,
) -> DetectResponse:
    """Upload a CSV file with ``timestamp,value`` columns."""
    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content), parse_dates=["timestamp"])
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Cannot parse CSV: {exc}") from exc

    if "timestamp" not in df.columns or "value" not in df.columns:
        raise HTTPException(
            status_code=400,
            detail=f"CSV must contain 'timestamp' and 'value' columns. Found: {df.columns.tolist()}",
        )

    df = df.sort_values("timestamp")
    timestamps = [t.isoformat() for t in df["timestamp"]]
    values = df["value"].tolist()

    _validate_detect_input(timestamps, values)
    model_slugs = models.split(",") if models else None
    slugs = _resolve_models(model_slugs)
    y = _build_series(timestamps, values)

    results, _ = _run_detection(y, slugs, timestamps, values)
    metadata: dict[str, Any] = {
        "input_length": len(y),
        "date_range": {"start": timestamps[0], "end": timestamps[-1]},
        "models_used": slugs,
        "filename": file.filename,
    }
    return DetectResponse(results=results, metadata=metadata)
