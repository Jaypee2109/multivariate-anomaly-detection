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
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
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
from time_series_transformer.config import ARTIFACTS_DIR, SMD_BASE_DIR
from time_series_transformer.exceptions import TransformerError
from time_series_transformer.models.multivariate.custom_transformer import (
    CustomTransformerDetector,
)
from time_series_transformer.models.multivariate.isolation_forest import (
    MultivariateIsolationForestDetector,
)
from time_series_transformer.models.multivariate.lstm_autoencoder import (
    LSTMAutoencoderAnomalyDetector,
)
from time_series_transformer.models.multivariate.lstm_forecaster import (
    LSTMForecasterMultivariateDetector,
)
from time_series_transformer.models.multivariate.tranad import TranADAnomalyDetector
from time_series_transformer.models.multivariate.var import VARResidualAnomalyDetector

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App & global state
# ---------------------------------------------------------------------------

manager = ModelManager()

# Multivariate machines discovered at startup
_mv_machines: list[str] = []


@asynccontextmanager
async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Load model checkpoints on startup."""
    # --- Univariate baselines (optional / legacy) ---
    checkpoint_dir = Path(
        os.environ.get("ANOMALY_CHECKPOINT_DIR", str(ARTIFACTS_DIR / "checkpoints"))
    )
    if checkpoint_dir.exists():
        loaded = manager.load_from_directory(checkpoint_dir)
        if loaded:
            logger.info("Loaded %d univariate model(s): %s", len(loaded), loaded)

    # --- Multivariate models — discover available machines ---
    mv_ckpt_dir = ARTIFACTS_DIR / "checkpoints" / "multivariate"
    mv_artifact_dir = ARTIFACTS_DIR / "multivariate"

    machines: set[str] = set()
    if mv_ckpt_dir.exists():
        machines.update(p.name for p in mv_ckpt_dir.iterdir() if p.is_dir())
    if mv_artifact_dir.exists():
        machines.update(
            p.stem.replace("_results", "") for p in mv_artifact_dir.glob("*_results.csv")
        )

    _mv_machines.clear()
    _mv_machines.extend(sorted(machines))

    if _mv_machines:
        logger.info(
            "Discovered %d multivariate machine(s), e.g. %s",
            len(_mv_machines),
            _mv_machines[:5],
        )
    else:
        logger.warning("No multivariate models found in %s or %s", mv_ckpt_dir, mv_artifact_dir)

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
    has_models = bool(manager.loaded_model_names) or bool(_mv_machines)
    return HealthResponse(
        status="healthy" if has_models else "no_models_loaded",
        models_loaded=manager.loaded_model_names,
        mv_machines=_mv_machines,
        timestamp=datetime.now().isoformat(),
    )


@app.get("/models", response_model=ModelsInfoResponse)
async def get_models() -> ModelsInfoResponse:
    if not manager.loaded_model_names and not _mv_machines:
        raise HTTPException(status_code=503, detail="No models loaded")
    details = [ModelDetail(**manager.get_model_info(slug)) for slug in manager.loaded_model_names]
    return ModelsInfoResponse(
        models=details,
        mv_machines=_mv_machines,
        checkpoint_dir=manager.checkpoint_dir,
    )


@app.get("/models/{model_name}", response_model=ModelDetail)
async def get_model_detail(model_name: str) -> ModelDetail:
    if model_name not in manager.loaded_model_names:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not loaded")
    return ModelDetail(**manager.get_model_info(model_name))


@app.get("/artifacts/machines")
async def list_artifact_machines() -> dict[str, Any]:
    """List machines with pre-computed multivariate result artifacts."""
    artifact_dir = ARTIFACTS_DIR / "multivariate"
    if not artifact_dir.exists():
        return {"machines": []}
    machines = sorted(p.stem.replace("_results", "") for p in artifact_dir.glob("*_results.csv"))
    return {"machines": machines}


@app.get("/artifacts/{machine_id}/models")
async def list_artifact_models(machine_id: str) -> dict[str, Any]:
    """List multivariate models available in the artifact CSV for a machine."""
    df = _load_smd_artifact(machine_id)
    if df is None:
        raise HTTPException(status_code=404, detail=f"No artifact for '{machine_id}'")
    models = _discover_artifact_models(df)
    return {"machine_id": machine_id, "models": models}


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
# WebSocket streaming — multivariate model loading
# ---------------------------------------------------------------------------

# Checkpoint filename (without .pt) → loader class
_MV_CHECKPOINT_LOADERS: dict[str, type] = {
    "isolation_forest_mv": MultivariateIsolationForestDetector,
    "lstm_autoencoder": LSTMAutoencoderAnomalyDetector,
    "tranad": TranADAnomalyDetector,
    "custom_transformer_t2v": CustomTransformerDetector,
    "var_residual": VARResidualAnomalyDetector,
    "lstm_forecaster_mv": LSTMForecasterMultivariateDetector,
}


def _load_multivariate_models(machine_id: str) -> dict[str, Any] | None:
    """Load fitted multivariate model checkpoints for a machine.

    Returns ``{slug: model_instance}`` or *None* if no checkpoints found.
    """
    ckpt_dir = ARTIFACTS_DIR / "checkpoints" / "multivariate" / machine_id
    if not ckpt_dir.exists():
        return None

    models: dict[str, Any] = {}
    for slug, cls in _MV_CHECKPOINT_LOADERS.items():
        path = ckpt_dir / f"{slug}.pt"
        if path.exists():
            try:
                models[slug] = cls.load_checkpoint(path)
                logger.info("Loaded MV checkpoint: %s", path)
            except Exception:
                logger.warning("Failed to load checkpoint: %s", path, exc_info=True)
    return models if models else None


def _load_smd_test_data(machine_id: str) -> pd.DataFrame | None:
    """Load raw SMD test data for a machine (unnormalised, matching training)."""
    try:
        from time_series_transformer.data_pipeline.smd_loading import load_smd_machine

        data = load_smd_machine(machine_id, base_dir=SMD_BASE_DIR, normalize=False)
        return data.test_df
    except Exception:
        logger.warning("Failed to load SMD test data for %s", machine_id, exc_info=True)
        return None


def _run_live_inference(
    models: dict[str, Any],
    test_df: pd.DataFrame,
) -> tuple[dict[str, dict], float]:
    """Run decision_function + predict on all loaded models.

    Returns ``({slug: {"scores": [...], "anomalies": [...]}}, total_latency_ms)``.
    """
    results: dict[str, dict] = {}
    total_latency = 0.0

    for slug, model in models.items():
        try:
            t0 = time.time()
            scores = model.decision_function(test_df)
            anomalies = model.predict(test_df)
            latency = (time.time() - t0) * 1000
            total_latency += latency

            results[slug] = {
                "scores": [float(s) if pd.notna(s) else None for s in scores.values],
                "anomalies": anomalies.astype(bool).tolist(),
            }
            logger.info("  %s: %d anomalies (%.0fms)", slug, int(anomalies.sum()), latency)
        except Exception:
            logger.warning("Live inference failed for %s", slug, exc_info=True)

    return results, total_latency


# ---------------------------------------------------------------------------
# WebSocket streaming — artifact fallback
# ---------------------------------------------------------------------------


def _load_smd_artifact(machine_id: str) -> pd.DataFrame | None:
    """Load pre-computed multivariate results from the artifact CSV.

    The artifact CSV (``artifacts/multivariate/{machine_id}_results.csv``)
    contains **test data only** with all feature values, ground-truth labels,
    and per-model anomaly scores and predictions.
    """
    if not machine_id:
        logger.warning("Empty machine_id received — client may be sending stale JS")
        return None

    path = ARTIFACTS_DIR / "multivariate" / f"{machine_id}_results.csv"
    if not path.exists():
        logger.warning("Artifact CSV not found: %s", path)
        return None

    try:
        return pd.read_csv(path)
    except Exception:
        logger.warning("Failed to read artifact CSV: %s", path, exc_info=True)
        return None


def _discover_artifact_models(df: pd.DataFrame) -> list[str]:
    """Extract model slugs from artifact CSV column names.

    Looks for ``{model}_score`` columns that also have ``{model}_is_anomaly``.
    """
    models = []
    for col in df.columns:
        if col.endswith("_score"):
            name = col[: -len("_score")]
            if f"{name}_is_anomaly" in df.columns:
                models.append(name)
    return sorted(models)


@app.websocket("/ws/stream")
async def websocket_stream(ws: WebSocket) -> None:
    """Stream anomaly detection results over WebSocket.

    Tries **live inference** first (load model checkpoints + raw test data,
    run ``decision_function``).  Falls back to **artifact CSV** when no
    checkpoints are available.

    Protocol:
    1. Client connects and sends JSON config (machine_id, feature, models, ...)
    2. Server loads models / artifact and sends "init" message
    3. Server streams "chunk" messages at configured rate
    4. Client can send control messages (pause/resume/reset/speed/close)
    5. Server sends "done" when dataset is exhausted
    """
    await ws.accept()

    try:
        # Step 1: Receive configuration
        raw_config = await asyncio.wait_for(ws.receive_text(), timeout=10.0)
        config = json.loads(raw_config)

        machine_id = config.get("machine_id", "")
        feature = config.get("feature", "cpu_r")
        model_slugs = config.get("models") or None
        chunk_size = max(1, min(100, config.get("chunk_size", 5)))
        interval_ms = max(100, min(10000, config.get("interval_ms", 1000)))

        # Step 2: Prepare data — try live inference, fall back to artifact
        mode = "artifact"
        full_models: dict[str, dict] = {}
        values: list[float] = []
        total = 0
        latency_ms = 0.0
        slugs: list[str] = []

        # --- Try live inference (checkpoints + raw test data) ---------------
        mv_models = _load_multivariate_models(machine_id)
        if mv_models:
            test_df = _load_smd_test_data(machine_id)
            if test_df is not None and feature in test_df.columns:
                # Filter to requested models (or use all available)
                active = (
                    {s: m for s, m in mv_models.items() if s in model_slugs}
                    if model_slugs
                    else mv_models
                )
                if active:
                    logger.info(
                        "Running live inference for %s (%d models)…",
                        machine_id,
                        len(active),
                    )
                    full_models, latency_ms = await asyncio.to_thread(
                        _run_live_inference,
                        active,
                        test_df,
                    )
                if full_models:
                    values = test_df[feature].tolist()
                    total = len(test_df)
                    slugs = list(full_models.keys())
                    mode = "live"
                    logger.info(
                        "Live inference complete: %d models, %.0fms",
                        len(slugs),
                        latency_ms,
                    )

        # --- Fall back to artifact CSV --------------------------------------
        if mode == "artifact":
            df = _load_smd_artifact(machine_id)
            if df is None:
                await ws.send_json(
                    {
                        "type": "error",
                        "detail": f"No data for machine '{machine_id}'. "
                        "Run: python -m time_series_transformer train-mv "
                        f"--machine {machine_id} --save-checkpoints",
                    }
                )
                await ws.close()
                return

            if feature not in df.columns:
                await ws.send_json(
                    {
                        "type": "error",
                        "detail": f"Feature '{feature}' not found for {machine_id}",
                    }
                )
                await ws.close()
                return

            values = df[feature].tolist()
            total = len(df)

            available_models = _discover_artifact_models(df)
            if not available_models:
                await ws.send_json(
                    {
                        "type": "error",
                        "detail": "No model results in artifact",
                    }
                )
                await ws.close()
                return

            slugs = model_slugs or available_models
            unknown = set(slugs) - set(available_models)
            if unknown:
                await ws.send_json(
                    {
                        "type": "error",
                        "detail": f"Unknown models: {sorted(unknown)}. "
                        f"Available: {available_models}",
                    }
                )
                await ws.close()
                return

            for slug in slugs:
                score_col = f"{slug}_score"
                anom_col = f"{slug}_is_anomaly"
                scores = df[score_col].tolist() if score_col in df.columns else [0.0] * total
                anomalies = (
                    df[anom_col].astype(bool).tolist()
                    if anom_col in df.columns
                    else [False] * total
                )
                full_models[slug] = {
                    "scores": [float(s) if pd.notna(s) else None for s in scores],
                    "anomalies": anomalies,
                }

            logger.info("Using artifact CSV for %s (%d models)", machine_id, len(slugs))

        # Step 3: Validate & generate timestamps
        if total < 10:
            await ws.send_json(
                {
                    "type": "error",
                    "detail": f"Dataset too small: {total} points",
                }
            )
            await ws.close()
            return

        ts_index = pd.date_range("2020-01-01", periods=total, freq="min")
        timestamps = ts_index.strftime("%Y-%m-%dT%H:%M:%S").tolist()

        # Send init message
        await ws.send_json(
            {
                "type": "init",
                "total": total,
                "dataset_name": f"{machine_id}/{feature}",
                "models_used": list(slugs),
                "latency_ms": round(latency_ms, 1),
                "mode": mode,
            }
        )

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
            except TimeoutError:
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

            await ws.send_json(
                {
                    "type": "chunk",
                    "index": index,
                    "timestamps": timestamps[index:end],
                    "values": values[index:end],
                    "models": chunk_models,
                    "total": total,
                    "progress": round(end / total, 4),
                }
            )

            index = end
            await asyncio.sleep(interval_ms / 1000.0)

        # Stream complete
        await ws.send_json({"type": "done", "total": total})

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except TimeoutError:
        logger.warning("WebSocket client did not send config within timeout")
        with suppress(Exception):
            await ws.close(code=1008, reason="Config timeout")
    except Exception:
        logger.exception("WebSocket error")
        try:
            await ws.send_json({"type": "error", "detail": "Internal server error"})
            await ws.close(code=1011)
        except Exception:
            pass
