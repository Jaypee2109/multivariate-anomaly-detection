# Time Series Transformer

Anomaly detection on time series data. The project combines statistical/ML baselines (ARIMA, Isolation Forest, LSTM) with a transformer-based detector currently being developed by a project partner.

## Requirements

- Python >= 3.11
- `pip` (dependencies listed in `requirements.txt`, pulled automatically by `pyproject.toml`)
- Kaggle credentials for dataset download (see [kagglehub docs](https://github.com/Kaggle/kagglehub))

## Installation

```bash
git clone <REPO-URL>
cd Transformer

python -m venv .venv

# Activate:
#   Linux/macOS: source .venv/bin/activate
#   Windows PS:  .\.venv\Scripts\Activate.ps1

pip install -e .
```

The editable install registers the `time_series_transformer` package and installs all dependencies.

## Quick Start

```bash
# Download and preprocess datasets
python -m time_series_transformer data

# Train baseline anomaly detectors with MLflow tracking
python -m time_series_transformer train --save-checkpoints --mlflow

# Train with ground-truth evaluation
python -m time_series_transformer train \
    --csv data/raw/nab/realKnownCause/realKnownCause/nyc_taxi.csv \
    --labels data/labels/nab/realKnownCause.json \
    --labels-key realKnownCause/nyc_taxi.csv \
    --save-checkpoints --mlflow

# Benchmark all models on multiple datasets
python -m time_series_transformer benchmark --config configs/benchmark_nab.yaml

# Start inference API server
python -m time_series_transformer serve

# Start interactive dashboard (requires running API server)
python -m time_series_transformer dashboard

# Exploratory data analysis
python -m time_series_transformer eda --csv data/raw/nab/realKnownCause/realKnownCause/nyc_taxi.csv
python -m time_series_transformer eda --anomalies

# Inspect a dataset or MLflow run
python -m time_series_transformer info --data data/raw/nab/realKnownCause/realKnownCause/nyc_taxi.csv
python -m time_series_transformer info --run-id <MLFLOW_RUN_ID> -v

# Launch MLflow UI
python -m time_series_transformer mlflow
```

## CLI Reference

All commands are run via `python -m time_series_transformer <command>`.

| Command     | Description                              | Key flags                                                             |
| ----------- | ---------------------------------------- | --------------------------------------------------------------------- |
| `data`      | Download and preprocess Kaggle datasets  | `--dataset {nab,smd_onmiad,nasa_smap_msl}`                            |
| `train`     | Train baseline anomaly detectors         | `--csv`, `--labels`, `--labels-key`, `--save-checkpoints`, `--mlflow` |
| `benchmark` | Evaluate models across multiple datasets | `--config YAML`, `--csv`, `--model`, `--mlflow`                       |
| `serve`     | Start FastAPI inference server           | `--host`, `--port`, `--reload`                                        |
| `dashboard` | Start interactive Plotly Dash dashboard  | `--host`, `--port`, `--debug`                                         |
| `eda`       | Exploratory data analysis / anomaly viz  | `--csv` or `--anomalies`, `--no-save-html`                            |
| `info`      | Inspect dataset CSV or MLflow run        | `--data` or `--run-id`, `-v`                                          |
| `mlflow`    | Launch MLflow UI                         | `--port`, `--host`                                                    |

Run `python -m time_series_transformer <command> --help` for full usage.

## Configuration

All hyperparameters and settings live in `src/time_series_transformer/config.py` with sensible defaults.
Override any value via environment variables:

```bash
cp .env.example .env
# Edit .env, then run commands as usual
```

See [.env.example](.env.example) for the full list of tunables (train ratio, ARIMA order, LSTM epochs, etc.).

## Datasets

Three Kaggle datasets are preconfigured:

| Name            | Kaggle slug                                             |
| --------------- | ------------------------------------------------------- |
| `nab`           | `boltzmannbrain/nab` (Numenta Anomaly Benchmark)        |
| `smd_onmiad`    | `mgusat/smd-onmiad`                                     |
| `nasa_smap_msl` | `patrickfleith/nasa-anomaly-detection-dataset-smap-msl` |

Data is downloaded to `data/raw/`, preprocessed to `data/processed/`. Both directories are git-ignored; folder structure is preserved via `.gitkeep`.

## Models

### Baselines (implemented)

All baseline models implement the `BaseAnomalyDetector` interface (`fit`, `predict`, `decision_function`):

| Model            | Approach                                         |
| ---------------- | ------------------------------------------------ |
| ARIMA Residual   | ARIMA fit, residual z-score threshold            |
| Isolation Forest | Scikit-learn tree ensemble                       |
| LSTM Forecast    | PyTorch LSTM, forecast-error quantile            |
| Rolling Z-Score  | Rolling mean/std threshold (disabled by default) |

### Transformer (work in progress)

A transformer-based anomaly detector is being developed by a project partner. Prototype code lives in `src/scratch_transformer/` and `src/scratch_time_series_transformer/` but is **not yet integrated** into the main package or CLI. Once ready it will implement `BaseAnomalyDetector` and plug into the existing pipeline.

## Evaluation Metrics

- **Point-level**: precision, recall, F1, AUC-ROC, AUC-PR
- **Range-level**: precision, recall, F1 over contiguous anomaly ranges

Both are printed to stdout and logged to MLflow when `--mlflow` is used.

## Benchmark Framework

The `benchmark` command runs registered models across multiple datasets and collects comparison metrics. Datasets are defined in YAML config files:

```yaml
# configs/benchmark_nab.yaml
datasets:
  - name: NAB_nyc_taxi
    csv: data/raw/nab/realKnownCause/realKnownCause/nyc_taxi.csv
    labels: data/labels/nab/realKnownCause.json
    labels_key: realKnownCause/nyc_taxi.csv
```

```bash
# Run benchmark
python -m time_series_transformer benchmark --config configs/benchmark_nab.yaml

# Filter to specific models
python -m time_series_transformer benchmark --config configs/benchmark_nab.yaml --model arima --model lstm

# List available models
python -m time_series_transformer benchmark --list-models
```

Results are saved to `artifacts/benchmark/results.csv` and printed as a comparison table. New models can be added to the registry without modifying core code.

## Inference Server

A FastAPI REST API for real-time anomaly detection:

```bash
python -m time_series_transformer serve          # default: localhost:8000
python -m time_series_transformer serve --port 9000
```

Key endpoints:

| Endpoint            | Method    | Description                              |
| ------------------- | --------- | ---------------------------------------- |
| `/health`           | GET       | Health check                             |
| `/models`           | GET       | List loaded models and their status      |
| `/models/{name}`    | GET       | Get model details                        |
| `/detect`           | POST      | Run anomaly detection (JSON)             |
| `/detect/dashboard` | POST      | Dashboard-optimised parallel arrays      |
| `/detect/csv`       | POST      | Upload CSV for detection                 |
| `/ws/stream`        | WebSocket | Stream chunked anomaly detection results |

## Dashboard

An interactive Plotly Dash dashboard for exploring data, models, and live monitoring:

```bash
# Start the API server first
python -m time_series_transformer serve

# Then start the dashboard
python -m time_series_transformer dashboard      # default: localhost:8050
```

Pages:

- **Home** — System overview, loaded models, quick dataset stats
- **Data Analysis** — Interactive time series exploration with zoom/pan
- **Model Analysis** — Compare model configs and MLflow experiment results
- **Live Monitoring** — Stream data through models and watch anomaly detection in real time

## Docker

A multi-stage `Dockerfile` and `docker-compose.yml` run all three services (API, dashboard, MLflow) in containers. The image uses `python:3.11-slim` with **CPU-only PyTorch** (~1.5 GB smaller than the full CUDA build).

> **Important:** Docker containers serve data and models from host directories via volume mounts — they do **not** download datasets or train models themselves. You must complete steps 1-3 below (install, download data, train) on the host **before** starting Docker. If you skip these steps, the services will start but the API will have no models loaded and the dashboard will show no data.

This section walks through every step needed to go from a fresh clone to a fully running system.

### Step 0: Install prerequisites

1. **Python >= 3.11** — [python.org/downloads](https://www.python.org/downloads/)
2. **Docker Desktop** — [docker.com/get-started](https://www.docker.com/get-started/) (includes `docker compose`)
3. **Kaggle credentials** — needed to download datasets. Create an API token at [kaggle.com/settings](https://www.kaggle.com/settings) and place `kaggle.json` in:
   - Linux/macOS: `~/.kaggle/kaggle.json`
   - Windows: `C:\Users\<you>\.kaggle\kaggle.json`

### Step 1: Clone and install the Python package

```bash
git clone <REPO-URL>
cd Transformer

python -m venv .venv

# Activate:
#   Linux/macOS: source .venv/bin/activate
#   Windows PS:  .\.venv\Scripts\Activate.ps1

pip install -e .
```

### Step 2: Download datasets

```bash
python -m time_series_transformer data
```

This downloads the NAB, SMD, and NASA SMAP/MSL datasets from Kaggle into `data/raw/` and preprocesses them into `data/processed/`.

### Step 3: Train baseline models

```bash
python -m time_series_transformer train \
    --csv data/raw/nab/realKnownCause/realKnownCause/nyc_taxi.csv \
    --labels data/labels/nab/realKnownCause.json \
    --labels-key realKnownCause/nyc_taxi.csv \
    --save-checkpoints \
    --mlflow
```

The `--save-checkpoints` flag is required — without it, trained models are not persisted and the inference server will have nothing to load. `--mlflow` enables experiment tracking (optional but recommended).

After training, verify the checkpoints exist:

```
artifacts/checkpoints/
├── arima_residual.joblib
├── isolation_forest.joblib
├── lstm_checkpoint.pt
└── rolling_zscore.joblib
```

> **Note:** Steps 1-3 populate the `data/` and `artifacts/` directories on your host machine. Docker mounts these directories as volumes — it does not copy or generate them. If `data/` or `artifacts/checkpoints/` are empty, the containers will start but produce no useful results.

### Step 4: Build and start Docker containers

```bash
docker compose build            # Build all three images (~5-10 min first time)
docker compose up -d            # Start services in the background
docker compose ps               # Check they're running
```

Once running, open:

| Service     | URL                   | Description              |
| ----------- | --------------------- | ------------------------ |
| `api`       | http://localhost:8000 | FastAPI inference server |
| `dashboard` | http://localhost:8050 | Plotly Dash dashboard    |
| `mlflow`    | http://localhost:5000 | MLflow tracking UI       |

Verify the API loaded the models:

```bash
curl http://localhost:8000/health
# {"status":"healthy","models_loaded":["arima_residual","isolation_forest","lstm","rolling_zscore"],...}
```

### How it works

The `Dockerfile` defines a shared **base** stage (installs Python deps + the package) and three lightweight **target** stages:

```
base  (python:3.11-slim + pip install .)
 ├── api        → serve --host 0.0.0.0        :8000
 ├── dashboard  → dashboard --host 0.0.0.0    :8050
 └── mlflow     → mlflow ui --host 0.0.0.0    :5000
```

`docker-compose.yml` wires them together:

- **api** mounts `./data` and `./artifacts`, sets `ANOMALY_CHECKPOINT_DIR=/app/artifacts/checkpoints`
- **dashboard** mounts `./data`, sets `ANOMALY_API_URL=http://api:8000` and `ANOMALY_WS_HOST=api:8000` so it reaches the API via Docker's internal DNS (not `localhost`)
- **mlflow** mounts `./mlruns` and uses a named volume for its SQLite database
- `dashboard` has `depends_on: api` to ensure correct startup order

Data, checkpoints, and MLflow runs live on the host and are **mounted as volumes** — nothing is baked into the image:

| Host path      | Container path      | Used by        |
| -------------- | ------------------- | -------------- |
| `./data`       | `/app/data`         | api, dashboard |
| `./artifacts`  | `/app/artifacts`    | api            |
| `./mlruns`     | `/app/mlruns`       | mlflow         |
| (named volume) | `/app/db/mlflow.db` | mlflow         |

### Common commands

```bash
docker compose up -d            # Start all services (detached)
docker compose ps               # Check running containers
docker compose logs -f          # Stream all logs
docker compose logs -f api      # Stream API logs only
docker compose down             # Stop and remove containers
docker compose up -d --build    # Rebuild images after code changes
```

### Building individual images

You can also build a single target without compose:

```bash
docker build --target api       -t tst-api .
docker build --target dashboard -t tst-dashboard .
docker build --target mlflow    -t tst-mlflow .
```

## Experiment Tracking

MLflow stores runs in a local SQLite database (`mlflow.db`). Each model gets its own top-level run with:

- Environment info (Python, torch, platform, CUDA, git SHA)
- Data hash (SHA-256 of input CSV)
- Model hyperparameters
- Point and range metrics
- Fit time

View results: `python -m time_series_transformer mlflow`

## Project Structure

```
src/
├── time_series_transformer/        # Main package (pip install -e .)
│   ├── cli/                        # Subcommand-based CLI
│   ├── data_pipeline/              # Download, load, preprocess, save
│   ├── models/baseline/            # Anomaly detector implementations
│   ├── api/                        # FastAPI inference server
│   ├── benchmark/                  # Model registry + benchmark runner
│   ├── analysis/                   # EDA and visualization
│   ├── utils/                      # I/O, startup checks, data validation
│   ├── config.py                   # Central configuration
│   ├── baseline_pipeline.py        # Training orchestrator
│   ├── evaluation.py               # Point + range metrics
│   ├── mlflow_utils.py             # MLflow logging helpers
│   └── split.py                    # Time-ordered train/test split
├── scratch_transformer/            # Partner WIP: NLP transformer experiments
└── scratch_time_series_transformer/ # Partner WIP: time-series transformer prototype

dashboard/                          # Plotly Dash interactive dashboard
├── app.py                          # Main Dash app with multi-page routing
├── api_client.py                   # HTTP client for the inference API
├── datasets.py                     # Dataset registry for the dashboard
├── mlflow_loader.py                # MLflow data loader
├── assets/                         # Static assets (CSS, JS)
│   └── websocket.js                # Clientside WebSocket manager for live monitoring
└── pages/                          # Dashboard pages (home, data, model, live)

configs/                            # YAML benchmark configs
└── benchmark_nab.yaml              # NAB realKnownCause datasets

Dockerfile                          # Multi-stage build (api / dashboard / mlflow)
docker-compose.yml                  # Orchestrates all three services
```

See [docs/architecture.md](docs/architecture.md) for detailed architecture and data flow.
See [docs/decisions.md](docs/decisions.md) for design decision log.

## Adding Dependencies

1. Add the package to `requirements.txt`
2. Run `pip install -e .` (or `pip install -r requirements.txt`)
3. Commit the updated `requirements.txt`

## License

MIT
