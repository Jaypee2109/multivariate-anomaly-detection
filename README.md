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
python -m time_series_transformer train --mlflow

# Train with ground-truth evaluation
python -m time_series_transformer train \
    --csv data/raw/nab/realKnownCause/realKnownCause/nyc_taxi.csv \
    --labels data/labels/nab/realKnownCause.json \
    --labels-key realKnownCause/nyc_taxi.csv \
    --mlflow

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

| Command  | Description                                | Key flags                                  |
|----------|--------------------------------------------|--------------------------------------------|
| `data`   | Download and preprocess Kaggle datasets    | `--dataset {nab,smd_onmiad,nasa_smap_msl}` |
| `train`  | Train baseline anomaly detectors           | `--csv`, `--labels`, `--labels-key`, `--mlflow` |
| `eda`    | Exploratory data analysis / anomaly viz    | `--csv` or `--anomalies`, `--no-save-html` |
| `info`   | Inspect dataset CSV or MLflow run          | `--data` or `--run-id`, `-v`               |
| `mlflow` | Launch MLflow UI                           | `--port`, `--host`                         |

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

| Name             | Kaggle slug                                                  |
|------------------|--------------------------------------------------------------|
| `nab`            | `boltzmannbrain/nab` (Numenta Anomaly Benchmark)             |
| `smd_onmiad`     | `mgusat/smd-onmiad`                                         |
| `nasa_smap_msl`  | `patrickfleith/nasa-anomaly-detection-dataset-smap-msl`      |

Data is downloaded to `data/raw/`, preprocessed to `data/processed/`. Both directories are git-ignored; folder structure is preserved via `.gitkeep`.

## Models

### Baselines (implemented)

All baseline models implement the `BaseAnomalyDetector` interface (`fit`, `predict`, `decision_function`):

| Model               | Approach                             |
|----------------------|--------------------------------------|
| ARIMA Residual       | ARIMA fit, residual z-score threshold |
| Isolation Forest     | Scikit-learn tree ensemble            |
| LSTM Forecast        | PyTorch LSTM, forecast-error quantile |
| Rolling Z-Score      | Rolling mean/std threshold (disabled by default) |

### Transformer (work in progress)

A transformer-based anomaly detector is being developed by a project partner. Prototype code lives in `src/scratch_transformer/` and `src/scratch_time_series_transformer/` but is **not yet integrated** into the main package or CLI. Once ready it will implement `BaseAnomalyDetector` and plug into the existing pipeline.

## Evaluation Metrics

- **Point-level**: precision, recall, F1, AUC-ROC, AUC-PR
- **Range-level**: precision, recall, F1 over contiguous anomaly ranges

Both are printed to stdout and logged to MLflow when `--mlflow` is used.

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
│   ├── analysis/                   # EDA and visualization
│   ├── utils/                      # I/O helpers
│   ├── config.py                   # Central configuration
│   ├── baseline_pipeline.py        # Training orchestrator
│   ├── evaluation.py               # Point + range metrics
│   ├── mlflow_utils.py             # MLflow logging helpers
│   └── split.py                    # Time-ordered train/test split
├── scratch_transformer/            # Partner WIP: NLP transformer experiments
└── scratch_time_series_transformer/ # Partner WIP: time-series transformer prototype
```

See [docs/architecture.md](docs/architecture.md) for detailed architecture and data flow.
See [docs/decisions.md](docs/decisions.md) for design decision log.

## Adding Dependencies

1. Add the package to `requirements.txt`
2. Run `pip install -e .` (or `pip install -r requirements.txt`)
3. Commit the updated `requirements.txt`

## License

MIT
