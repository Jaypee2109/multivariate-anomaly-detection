# Architecture

## Project Layout

```
Transformer/
├── pyproject.toml          # Package metadata, editable-install config
├── requirements.txt        # Pinned dependencies (read by pyproject.toml)
├── .env.example            # Tunable env vars with defaults
├── data/
│   ├── raw/                # Downloaded Kaggle datasets (git-ignored)
│   ├── processed/          # Preprocessed CSVs (git-ignored)
│   └── labels/             # Ground-truth label files (e.g. NAB JSON)
├── artifacts/              # Pipeline outputs: anomaly CSVs, benchmark results (git-ignored)
├── configs/                # YAML benchmark configs
│   └── benchmark_nab.yaml  # NAB realKnownCause datasets
├── dashboard/              # Plotly Dash interactive dashboard
│   ├── app.py              # Main Dash app with multi-page routing
│   ├── api_client.py       # HTTP client for the inference API
│   ├── datasets.py         # Dataset registry for the dashboard
│   ├── mlflow_loader.py    # MLflow data loader
│   ├── assets/             # Static assets (CSS, JS)
│   │   └── websocket.js    # Clientside WebSocket manager for live monitoring
│   └── pages/              # Dashboard pages (home, data, model, live)
├── Dockerfile              # Multi-stage build (api / dashboard / mlflow targets)
├── docker-compose.yml      # Orchestrates all three services
├── mlflow.db               # SQLite backend for MLflow (git-ignored)
├── docs/                   # Project documentation
└── src/
    ├── time_series_transformer/   # Main package
    │   ├── __main__.py            # python -m entry point
    │   ├── config.py              # Central config (env-var overridable)
    │   ├── baseline_pipeline.py   # Orchestrates baseline model training
    │   ├── evaluation.py          # Point + range anomaly metrics
    │   ├── mlflow_utils.py        # MLflow setup, logging helpers
    │   ├── split.py               # Time-ordered train/test split
    │   ├── cli/                   # Subcommand-based CLI
    │   ├── data_pipeline/         # Download, load, preprocess, save
    │   ├── models/baseline/       # Anomaly detector implementations
    │   ├── api/                   # FastAPI inference server
    │   ├── benchmark/             # Model registry + benchmark runner
    │   ├── analysis/              # EDA and visualization
    │   ├── utils/                 # I/O, startup checks, data validation
    │   └── scripts/               # Legacy standalone entry points
    ├── scratch_transformer/       # Partner WIP: NLP transformer experiments
    └── scratch_time_series_transformer/  # Partner WIP: time-series transformer prototype
```

## Package Boundaries

### `time_series_transformer` (main package)

Installable via `pip install -e .`. All imports use the `time_series_transformer.*` namespace.

### `scratch_transformer` / `scratch_time_series_transformer` (partner WIP)

Transformer model code being developed by a project partner. Contains an NLP word-level transformer and an early time-series transformer prototype. **Not yet integrated** into the main package or CLI. Once the transformer detector is ready it will implement `BaseAnomalyDetector` and be added to `models/`.

## Key Modules

### `config.py` — Central Configuration

Single source of truth for paths, hyperparameters, and dataset definitions.
Every value can be overridden via environment variables (see `.env.example`).
Optional `python-dotenv` support loads a `.env` file at import time.

Derived paths are relative to the package location, never hardcoded:

```
PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent.parent
DATA_DIR     = PROJECT_ROOT / "data"
```

### `cli/` — Subcommand Dispatcher

Pattern: each module exposes `register(subparsers)` + `run(args)`.
`cli/main.py` wires them into a single `argparse` parser.

| Command     | Module          | Purpose                                    |
|-------------|-----------------|------------------------------------------  |
| `data`      | `cli/data`      | Download and preprocess Kaggle datasets    |
| `train`     | `cli/train`     | Train baselines, optional MLflow + eval    |
| `train-mv`  | `cli/train_multivariate` | Train multivariate models on SMD    |
| `benchmark` | `cli/benchmark` | Evaluate models across multiple datasets   |
| `serve`     | `cli/serve`     | Start FastAPI inference server             |
| `dashboard` | `cli/dashboard` | Start Plotly Dash interactive dashboard    |
| `eda`       | `cli/eda`       | EDA on raw CSV or anomaly artifacts        |
| `info`      | `cli/info`      | Inspect dataset or MLflow run              |
| `mlflow`    | `cli/main`      | Launch MLflow UI (inline, small)           |

### `models/baseline/` — Anomaly Detectors

All inherit from `BaseAnomalyDetector` (ABC):

```python
class BaseAnomalyDetector(ABC):
    def fit(self, y: pd.Series) -> "BaseAnomalyDetector": ...
    def predict(self, y: pd.Series) -> pd.Series: ...          # bool flags
    def decision_function(self, y: pd.Series) -> pd.Series: ... # scores
```

Convention: higher scores = more anomalous.

| Model                          | Approach                                     |
|--------------------------------|----------------------------------------------|
| `RollingZScoreAnomalyDetector` | Rolling mean/std, z-score threshold           |
| `ARIMAResidualAnomalyDetector` | ARIMA fit, residual z-score threshold          |
| `IsolationForestAnomalyDetector` | Scikit-learn ensemble, contamination param  |
| `LSTMForecastAnomalyDetector`  | PyTorch LSTM, forecast-error quantile          |

### `models/multivariate/` — Multivariate Anomaly Detectors

All follow the same `fit() / predict() / decision_function()` interface (via `BaseMultivariateAnomalyDetector` or `@dataclass`). Designed for the SMD dataset (38 features, pre-normalised to [0, 1]).

| Model                                  | Approach                                       |
|----------------------------------------|------------------------------------------------|
| `VARResidualAnomalyDetector`           | VAR(p) forecast residual z-scores, max/mean aggregation |
| `MultivariateIsolationForestDetector`  | Scikit-learn ensemble on raw multivariate features |
| `LSTMAutoencoderAnomalyDetector`       | LSTM autoencoder reconstruction error, quantile threshold |
| `LSTMForecasterMultivariateDetector`   | LSTM next-step forecast error, quantile threshold |

LSTM models use MSE for both training loss and anomaly scoring. Thresholds are computed as quantiles of training-set scores (overlap-averaged for the autoencoder). See decisions D14–D17 for rationale.

### `evaluation.py` — Metrics

Two metric levels:

- **PointMetrics**: precision, recall, F1, AUC-ROC, AUC-PR (point-wise)
- **RangeMetrics**: precision, recall, F1 at the contiguous-anomaly-range level

`summarize_anomalies()` prints results and returns `(PointMetrics, RangeMetrics)` when ground-truth labels are provided.

### `data_pipeline/` — ETL

```
download (kagglehub) -> load (CSV walk) -> preprocess (datetime + z-score) -> save
```

- `data_download.py`: Kaggle download via `kagglehub`, copies to `data/raw/`
- `data_loading.py`: Recursive CSV discovery, skips macOS metadata
- `preprocessing.py`: `PreprocessingConfig` dataclass, datetime index, standard scaling
- `data_save.py`: Write processed DataFrames back to `data/processed/`
- `labels.py`: Load NAB-format JSON labels, convert to point-wise binary series
- `pipeline.py`: Orchestrates the full ETL, accepts optional dataset filter

### `mlflow_utils.py` — Experiment Tracking

- SQLite backend at `PROJECT_ROOT/mlflow.db`
- One flat top-level run per model (not nested)
- Logs: environment info (Python, torch, OS, CUDA, git SHA), data hash (SHA-256), model params (filtered), point/range metrics, anomaly summary, fit time

### `baseline_pipeline.py` — Training Orchestrator

1. Seeds numpy/torch/CUDA
2. Loads time series via `load_timeseries()`
3. Splits with `train_test_split_series()`
4. Iterates models: fit, predict, score, evaluate, optionally log to MLflow
5. Saves combined anomaly artifacts CSV

### `api/` — Inference Server

FastAPI-based REST + WebSocket API for real-time anomaly detection.

- `inference_server.py`: FastAPI app with REST endpoints (`/health`, `/models`, `/detect`, `/detect/dashboard`, `/detect/csv`) and a WebSocket endpoint (`/ws/stream`) for chunked streaming
- `model_manager.py`: Loads and manages model instances, handles checkpoint discovery
- `schemas.py`: Pydantic request/response schemas

The server loads trained model checkpoints on startup and exposes them for inference. The `/ws/stream` WebSocket endpoint supports server-side dataset loading, batch inference, and chunked delivery with pause/resume/reset/speed controls. Launched via `python -m time_series_transformer serve`.

### `benchmark/` — Benchmark Framework

Systematic evaluation of models across multiple datasets.

- `dataset_spec.py`: `DatasetSpec` dataclass (name, csv_path, optional labels)
- `registry.py`: Model factory registry — `register_model(name, factory)`, `get_factory(name)`, `list_models()`. Auto-registers built-in models (arima, isolation_forest, lstm, rolling_zscore)
- `runner.py`: `BenchmarkRunner` — iterates models x datasets, collects metrics with error-tolerant execution
- `results.py`: `BenchmarkResult` dataclass + `ResultsCollector` (DataFrame export, CSV, console table)

New models are added via `register_model("name", factory_fn)` — no core code changes needed. Dataset lists are defined in YAML config files under `configs/`.

### `dashboard/` — Interactive Dashboard

Plotly Dash multi-page application for data exploration and live monitoring. Communicates with the inference server via HTTP (`api_client.py`) and WebSocket (`assets/websocket.js`).

- `app.py`: Main Dash app with navbar navigation
- `api_client.py`: HTTP client wrapping inference API calls
- `datasets.py`: Dataset registry (paths + metadata for the UI)
- `mlflow_loader.py`: Loads MLflow experiment data for comparison
- `assets/websocket.js`: Clientside JavaScript managing WebSocket lifecycle and JS-to-Dash buffer drain for Live Monitoring
- `pages/home.py`: System overview, model status, dataset stats
- `pages/data_analysis.py`: Interactive time series exploration
- `pages/model_analysis.py`: Model config comparison + MLflow results
- `pages/live_monitoring.py`: Real-time streaming anomaly detection via WebSocket

## Data Flow

```
Kaggle ──download──> data/raw/
                       │
                    load_dataset()
                       │
                  preprocess_dataset_dict()
                       │
                       v
                   data/processed/
                       │
              load_timeseries(csv_path)
                       │
               train_test_split_series()
                       │
                 ┌─────┴─────┐
              y_train      y_test
                 │            │
            model.fit()  model.predict()
                          model.decision_function()
                              │
                     summarize_anomalies()
                              │
                       ┌──────┴──────┐
                       v             v
           artifacts/anomalies/   MLflow (metrics,
                *.csv             params, tags)

─── Multivariate (SMD) path ───

  data/raw/smd_onmiad/ServerMachineDataset/
                       │
              load_smd_machine(normalize=False)
                       │              ↑ SMD is pre-normalised to [0,1]
                 ┌─────┴─────┐
              X_train      X_test + y_true
                 │            │
            model.fit()  model.predict()
                          model.decision_function()
                              │
                     compute_point_metrics()
                              │
                              v
              artifacts/multivariate/{machine}_results.csv
                              │
                       Dashboard (Model Analysis page)

─── Benchmark path ───

  configs/*.yaml ──> BenchmarkRunner
                        │
                   datasets × models
                        │
                  _run_single() per pair
                        │
                        v
              artifacts/benchmark/results.csv

─── Inference path ───

  Model checkpoints ──> ModelManager (serve)
                             │
                    FastAPI REST + WebSocket
                             │
                     ┌───────┴────────┐
                     │                │
              REST endpoints     /ws/stream
           /detect, /models    (chunked streaming)
                     │                │
                Dashboard (Dash) ─────┘
                ├── api_client.py (HTTP)
                └── websocket.js  (WS)

─── Docker path ───

  docker compose up
        │
  ┌─────┼──────────┐
  api   dashboard   mlflow
  :8000 :8050       :5000
  │       │
  └── Docker DNS ──┘
      (dashboard → api)
```

## Configuration Precedence

1. Environment variable (highest priority)
2. `.env` file (loaded by python-dotenv if installed)
3. Default value in `config.py` (lowest priority)
