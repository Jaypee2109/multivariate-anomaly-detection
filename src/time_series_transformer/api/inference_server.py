"""Anomaly Detection Inference API.

FastAPI REST API for serving baseline anomaly detection models.

Usage::

    python -m time_series_transformer serve --checkpoint-dir artifacts/checkpoints
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
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
from time_series_transformer.config import ARTIFACTS_DIR, RAW_DATA_DIR
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


# ---------------------------------------------------------------------------
# WebSocket streaming
# ---------------------------------------------------------------------------


def _load_nab_dataset(rel_path: str) -> pd.DataFrame | None:
    """Load a NAB CSV by its relative path under RAW_DATA_DIR/nab."""
    csv_path = RAW_DATA_DIR / "nab" / rel_path
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    if "timestamp" not in df.columns or "value" not in df.columns:
        return None
    return df.sort_values("timestamp")


@app.websocket("/ws/stream")
async def websocket_stream(ws: WebSocket) -> None:
    """Stream anomaly detection results over WebSocket.

    Protocol:
    1. Client connects and sends JSON config
    2. Server loads dataset, runs batch inference, sends "init" message
    3. Server streams "chunk" messages at configured rate
    4. Client can send control messages (pause/resume/reset/speed/close)
    5. Server sends "done" when dataset is exhausted
    """
    await ws.accept()

    try:
        # Step 1: Receive configuration
        raw_config = await asyncio.wait_for(ws.receive_text(), timeout=10.0)
        config = json.loads(raw_config)

        dataset_path = config.get("dataset_path", "")
        model_slugs = config.get("models") or None
        chunk_size = max(1, min(100, config.get("chunk_size", 5)))
        interval_ms = max(100, min(10000, config.get("interval_ms", 1000)))

        # Step 2: Load dataset server-side
        df = _load_nab_dataset(dataset_path)
        if df is None:
            await ws.send_json({"type": "error", "detail": f"Dataset not found: {dataset_path}"})
            await ws.close()
            return

        timestamps = df["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S").tolist()
        values = df["value"].tolist()
        total = len(timestamps)

        if total < 10:
            await ws.send_json({"type": "error", "detail": f"Dataset too small: {total} points"})
            await ws.close()
            return

        # Step 3: Resolve models and run batch inference
        slugs = model_slugs or manager.loaded_model_names
        if not slugs:
            await ws.send_json({"type": "error", "detail": "No models loaded"})
            await ws.close()
            return

        unknown = set(slugs) - set(manager.loaded_model_names)
        if unknown:
            await ws.send_json({
                "type": "error",
                "detail": f"Unknown models: {sorted(unknown)}. Available: {manager.loaded_model_names}",
            })
            await ws.close()
            return

        y = _build_series(timestamps, values)

        try:
            raw_results = manager.detect(y, model_slugs=slugs)
        except Exception as exc:
            await ws.send_json({"type": "error", "detail": f"Inference failed: {exc}"})
            await ws.close()
            return

        # Pre-compute full result arrays (same format as /detect/dashboard)
        total_latency = 0.0
        full_models: dict[str, dict] = {}
        for slug, (anomalies, scores, latency) in raw_results.items():
            total_latency += latency
            sc_vals = scores.values[-total:]
            an_vals = anomalies.values[-total:]
            full_models[slug] = {
                "scores": [float(s) if pd.notna(s) else None for s in sc_vals],
                "anomalies": [bool(a) for a in an_vals],
            }

        # Send init message
        await ws.send_json({
            "type": "init",
            "total": total,
            "dataset_name": dataset_path.split("/")[-1],
            "models_used": list(slugs),
            "latency_ms": round(total_latency, 2),
        })

        # Step 4: Stream chunks
        index = 0
        paused = False

        while index < total:
            # Check for control messages (non-blocking)
            try:
                msg = await asyncio.wait_for(ws.receive_text(), timeout=0.01)
                ctrl = json.loads(msg)
                action = ctrl.get("action", "")

                if action == "pause":
                    paused = True
                    await ws.send_json({"type": "paused", "index": index})
                    continue
                elif action == "resume":
                    paused = False
                    await ws.send_json({"type": "resumed", "index": index})
                elif action == "reset":
                    index = 0
                    paused = False
                    await ws.send_json({"type": "reset", "index": 0})
                    continue
                elif action == "speed":
                    if ctrl.get("chunk_size") is not None:
                        chunk_size = max(1, min(100, ctrl["chunk_size"]))
                    if ctrl.get("interval_ms") is not None:
                        interval_ms = max(100, min(10000, ctrl["interval_ms"]))
                elif action == "close":
                    await ws.close()
                    return
            except asyncio.TimeoutError:
                pass  # No control message — continue streaming

            if paused:
                await asyncio.sleep(0.1)
                continue

            # Build and send chunk
            end = min(index + chunk_size, total)
            chunk_models: dict[str, dict] = {}
            for slug, mdata in full_models.items():
                chunk_models[slug] = {
                    "scores": mdata["scores"][index:end],
                    "anomalies": mdata["anomalies"][index:end],
                }

            await ws.send_json({
                "type": "chunk",
                "index": index,
                "timestamps": timestamps[index:end],
                "values": values[index:end],
                "models": chunk_models,
                "total": total,
                "progress": round(end / total, 4),
            })

            index = end
            await asyncio.sleep(interval_ms / 1000.0)

        # Stream complete
        await ws.send_json({"type": "done", "total": total})

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except asyncio.TimeoutError:
        logger.warning("WebSocket client did not send config within timeout")
        try:
            await ws.close(code=1008, reason="Config timeout")
        except Exception:
            pass
    except Exception:
        logger.exception("WebSocket error")
        try:
            await ws.send_json({"type": "error", "detail": "Internal server error"})
            await ws.close(code=1011)
        except Exception:
            pass
