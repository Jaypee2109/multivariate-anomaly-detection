# =============================================================================
# Multi-stage Dockerfile for time-series-transformer
#
# Targets:
#   api       — FastAPI inference server (port 8000)
#   dashboard — Plotly Dash dashboard    (port 8050)
#   mlflow    — MLflow tracking UI       (port 5000)
#
# Build examples:
#   docker build --target api       -t tst-api .
#   docker build --target dashboard -t tst-dashboard .
#   docker build --target mlflow    -t tst-mlflow .
# =============================================================================

# ---------------------------------------------------------------------------
# Base stage: shared Python environment with all dependencies
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS base

WORKDIR /app

# System packages needed at build time
RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch first (saves ~1.5 GB vs full CUDA build)
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu

# Copy dependency specs and install the package
COPY pyproject.toml requirements.txt ./
COPY src/ src/
COPY dashboard/ dashboard/

RUN pip install --no-cache-dir .

# Create data/artifact directories (will be overridden by volume mounts)
RUN mkdir -p data/raw data/processed artifacts/checkpoints mlflow-db

# ---------------------------------------------------------------------------
# API target: FastAPI inference server
# ---------------------------------------------------------------------------
FROM base AS api

EXPOSE 8000

CMD ["python", "-m", "time_series_transformer", "serve", "--host", "0.0.0.0"]

# ---------------------------------------------------------------------------
# Dashboard target: Plotly Dash application
# ---------------------------------------------------------------------------
FROM base AS dashboard

EXPOSE 8050

CMD ["python", "-m", "time_series_transformer", "dashboard", "--host", "0.0.0.0"]

# ---------------------------------------------------------------------------
# MLflow target: experiment tracking UI
# ---------------------------------------------------------------------------
FROM base AS mlflow

EXPOSE 5000

CMD ["mlflow", "server", \
     "--host", "0.0.0.0", \
     "--port", "5000", \
     "--backend-store-uri", "sqlite:////app/mlflow-db/mlflow.db"]
