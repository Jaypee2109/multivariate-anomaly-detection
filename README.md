# Time Series Transformer

Multivariate time series anomaly detection on the [Server Machine Dataset (SMD)](https://www.kaggle.com/datasets/mgusat/smd-onmiad). Four models — Isolation Forest, LSTM Autoencoder, TranAD, and a custom Transformer with Time2Vec embeddings — are trained, evaluated, and served through an interactive dashboard and REST/WebSocket API.

## Requirements

- Python >= 3.11
- [Docker Desktop](https://www.docker.com/get-started/) (for containerized deployment)
- A free [Kaggle](https://www.kaggle.com/) account (for dataset download)

## Quick Start

**Step 1 — Clone and install**

```bash
git clone https://github.com/Jaypee2109/multivariate-anomaly-detection.git
cd multivariate-anomaly-detection
python -m venv .venv
source .venv/bin/activate            # Windows: .\.venv\Scripts\Activate.ps1
pip install -e .
```

**Step 2 — Set up Kaggle credentials**

The SMD dataset is hosted on Kaggle. To allow automatic download you need an API token:

1. Go to [kaggle.com/settings](https://www.kaggle.com/settings) and scroll to the **API** section
2. Click **Generate New Token** — this shows your username and a key
3. In the project folder, copy the example config: `cp .env.example .env`
4. Open `.env` and fill in the two lines at the top:
   ```
   KAGGLE_USERNAME=your_username
   KAGGLE_KEY=your_api_key
   ```

**Step 3 — Download data and train models**

```bash
python -m time_series_transformer data --dataset smd
docker compose up -d mlflow          # Start MLflow server first (port 5000)
python -m time_series_transformer train-mv --machine all --save-checkpoints --mlflow
```

**Step 4 — Launch**

```bash
docker compose up -d                 # Start API + Dashboard (MLflow already running)
```

Or without Docker:

```bash
python -m time_series_transformer mlflow &   # MLflow on :5000
python -m time_series_transformer serve &    # API on :8000
python -m time_series_transformer dashboard  # Dashboard on :8050
```

| Service   | URL                   |
| --------- | --------------------- |
| Dashboard | http://localhost:8050 |
| API       | http://localhost:8000 |
| MLflow    | http://localhost:5000 |

## Models

All multivariate models target SMD (28 server machines, 38 features, window size 30) and share a unified evaluation pipeline.

| Model              | Strategy       | Approach                                                              |
| ------------------ | -------------- | --------------------------------------------------------------------- |
| Isolation Forest   | Point-wise     | Scikit-learn ensemble on raw multivariate features                    |
| LSTM Autoencoder   | Reconstruction | Encoder-decoder LSTM, MSE reconstruction error                        |
| TranAD             | Reconstruction | Transformer encoder + dual decoder with adversarial self-conditioning |
| Custom Transformer | Forecasting    | Learnable Time2Vec + cross-attention decoder, MSE prediction error    |

Optional models: `--model var` (VAR Residual), `--model lstm_forecaster` (LSTM Forecaster).

## Evaluation

Five complementary protocols, computed across all 28 machines:

- **Point-level** — Precision, Recall, F1, AUC-ROC, AUC-PR
- **Point-adjust** — PA-F1 (segment-level credit, OmniAnomaly protocol)
- **Best-F1** — Oracle threshold search (upper bound on point-level F1)
- **Detection latency** — Timesteps from anomaly onset to first detection
- **Range-level** — Precision, Recall, F1 over contiguous segments

## Dashboard

Four pages for interactive exploration:

- **Home** — System overview, loaded models, dataset stats
- **Data Analysis** — Time series exploration with autocorrelation, distributions
- **Model Analysis** — Side-by-side metric comparison, detection visualization, score distributions
- **Live Monitoring** — Real-time streaming via WebSocket with pause/resume controls

## CLI Reference

All commands: `python -m time_series_transformer <command>`. Run with `--help` for full usage.

| Command     | Description                             | Key flags                                                |
| ----------- | --------------------------------------- | -------------------------------------------------------- |
| `data`      | Download and preprocess Kaggle datasets | `--dataset {nab,smd,nasa_smap_msl}`               |
| `train-mv`  | Train multivariate detectors on SMD     | `--machine`, `--model`, `--save-checkpoints`, `--mlflow` |
| `train`     | Train univariate baseline detectors     | `--csv`, `--labels`, `--save-checkpoints`                |
| `benchmark` | Evaluate models across datasets         | `--config YAML`, `--model`, `--mlflow`                   |
| `serve`     | Start FastAPI inference server          | `--host`, `--port`                                       |
| `dashboard` | Start Plotly Dash dashboard             | `--host`, `--port`                                       |
| `mlflow`    | Launch MLflow UI                        | `--port`                                                 |
| `eda`       | Exploratory data analysis               | `--csv` or `--anomalies`                                 |
| `info`      | Inspect dataset or MLflow run           | `--data` or `--run-id`                                   |

## Configuration

All hyperparameters live in `src/time_series_transformer/config.py` with sensible defaults. Override via environment variables in `.env` (see [.env.example](.env.example)).

## API Endpoints

| Endpoint            | Method    | Description                         |
| ------------------- | --------- | ----------------------------------- |
| `/health`           | GET       | Health check                        |
| `/models`           | GET       | List loaded models                  |
| `/detect`           | POST      | Run anomaly detection (JSON)        |
| `/detect/dashboard` | POST      | Dashboard-optimized response format |
| `/detect/csv`       | POST      | Upload CSV for detection            |
| `/ws/stream`        | WebSocket | Stream chunked detection results    |

## Docker

```bash
docker compose up -d            # Start all services
docker compose ps               # Check status
docker compose logs -f          # Stream logs
docker compose down             # Stop
docker compose up -d --build    # Rebuild after code changes
```

> Data and trained models must exist on the host before starting Docker (steps 1–3 from [Quick Start](#quick-start)). Docker mounts `./data` and `./artifacts` as volumes.

## Project Structure

```
src/time_series_transformer/        # Main package
├── cli/                            # Subcommand-based CLI
├── data_pipeline/                  # Download, load, preprocess
├── models/
│   ├── baseline/                   # Univariate (ARIMA, IF, LSTM, Z-Score)
│   └── multivariate/               # Multivariate (IF, LSTM-AE, TranAD, Custom TF)
├── api/                            # FastAPI inference server
├── benchmark/                      # Model registry + benchmark runner
├── config.py                       # Central configuration
├── evaluation.py                   # All evaluation metrics
├── multivariate_pipeline.py        # SMD training orchestrator
└── mlflow_utils.py                 # MLflow logging helpers

dashboard/                          # Plotly Dash application
├── app.py                          # Main app with multi-page routing
├── pages/                          # home, data_analysis, model_analysis, live_monitoring
└── assets/                         # CSS, JS (WebSocket manager, theme toggle)

Dockerfile                          # Multi-stage build (api / dashboard / mlflow)
docker-compose.yml                  # Orchestrates all three services
```

See [docs/architecture.md](docs/architecture.md) for detailed architecture and data flow.

## License

MIT
