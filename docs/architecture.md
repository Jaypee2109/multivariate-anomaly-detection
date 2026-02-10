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
├── artifacts/              # Pipeline outputs: anomaly CSVs (git-ignored)
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
    │   ├── analysis/              # EDA and visualization
    │   ├── utils/                 # I/O helpers (anomaly_io)
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

| Command  | Module       | Purpose                                    |
|----------|--------------|--------------------------------------------|
| `data`   | `cli/data`   | Download and preprocess Kaggle datasets    |
| `train`  | `cli/train`  | Train baselines, optional MLflow + eval    |
| `eda`    | `cli/eda`    | EDA on raw CSV or anomaly artifacts        |
| `info`   | `cli/info`   | Inspect dataset or MLflow run              |
| `mlflow` | `cli/main`   | Launch MLflow UI (inline, small)           |

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
                              v
                  artifacts/anomalies/*.csv
                  MLflow (metrics, params, tags)
```

## Configuration Precedence

1. Environment variable (highest priority)
2. `.env` file (loaded by python-dotenv if installed)
3. Default value in `config.py` (lowest priority)
