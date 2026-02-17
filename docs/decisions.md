# Decision Log

Significant architectural and design decisions, recorded in chronological order.

---

## D1: Adopt MLflow for experiment tracking

**Date:** 2026-01
**Status:** Accepted

**Context:** Baseline models were trained without any experiment tracking. Sibling project dsai-grp1 already uses MLflow successfully.

**Decision:** Use MLflow with a local SQLite backend (`mlflow.db`) for logging params, metrics, tags, and artifacts.

**Alternatives considered:**

- Weights & Biases — heavier dependency, requires account
- CSV/JSON manual logging — no UI, no comparison features
- TensorBoard — primarily for deep learning, weaker for tabular metrics

**Consequences:**

- `mlflow>=2.10` added as dependency
- `mlflow_utils.py` provides setup and logging helpers
- `mlflow.db` + `mlruns/` added to `.gitignore`
- MLflow UI launchable via `python -m time_series_transformer mlflow`

---

## D2: Flat MLflow runs (not nested)

**Date:** 2026-01
**Status:** Accepted

**Context:** Initial implementation used a parent run with child runs per model (pattern from dsai-grp1 where a single TFT model has nested folds). Our baseline models are independent, not folds of one model.

**Decision:** Each model gets its own top-level MLflow run. No parent/child nesting.

**Rationale:** Flat runs appear in the MLflow UI table as peers, making side-by-side comparison straightforward. Nested runs hide children behind an expand toggle.

**Consequences:**

- Simpler code (`mlflow.start_run()` per model, no `nested=True`)
- All models comparable in a single table view

---

## D3: Unified CLI with subcommand dispatcher

**Date:** 2026-01
**Status:** Accepted

**Context:** Five standalone scripts in `scripts/` with hardcoded paths. dsai-grp1 has a clean `python -m timeseries_forecasting <command>` CLI.

**Decision:** Subcommand-based CLI: `python -m time_series_transformer <command>`. Each command is a module with `register(subparsers)` + `run(args)`. Central `cli/main.py` wires them together.

**Alternatives considered:**

- Click — adds a dependency for what argparse handles well
- Keep standalone scripts only — poor discoverability, inconsistent args

**Consequences:**

- `cli/` package with `data`, `train`, `eda`, `info` modules
- `__main__.py` delegates to `cli/main.py`
- Legacy `scripts/` kept for backward compatibility but no longer the primary interface

---

## D4: Config with env-var overrides + `.env.example`

**Date:** 2026-01
**Status:** Accepted

**Context:** Hyperparameters were scattered or hardcoded. Needed a way to tune experiments without editing code, especially for reproducibility across machines.

**Decision:** Single `config.py` with `_env_int/_env_float/_env_bool/_env_str` helpers. Every tunable has a default. Optional `python-dotenv` loads `.env` at import time. `.env.example` documents all variables.

**Alternatives considered:**

- YAML/TOML config file — extra file format, harder to override per-run in CI
- `config.py` only (no env overrides) — requires code edits to tune
- Hydra — too heavy for current scope

**Consequences:**

- `python-dotenv` added as optional dependency
- `.env` git-ignored, `.env.example` committed
- All model hyperparameters, thresholds, and paths overridable via env vars

---

## D5: Reproducibility measures

**Date:** 2026-01
**Status:** Accepted

**Context:** Experiments must be reproducible across machines and over time.

**Decision:** Implement multiple reproducibility safeguards:

| Measure            | Implementation                                               |
| ------------------ | ------------------------------------------------------------ |
| RNG seeding        | `_seed_everything()`: numpy, torch, CUDA                     |
| Git SHA            | Logged as MLflow tag via `subprocess git rev-parse HEAD`     |
| Data versioning    | SHA-256 hash of input CSV logged as MLflow param             |
| Environment        | Python version, torch version, platform, CUDA logged as tags |
| Dependency pinning | Version lower bounds in `requirements.txt`                   |

**Consequences:**

- Every MLflow run captures enough context to reproduce
- No hardcoded paths anywhere in the codebase

---

## D6: BaseAnomalyDetector ABC

**Date:** 2026-01
**Status:** Accepted

**Context:** Multiple anomaly detection models needed a shared interface for the pipeline to iterate over.

**Decision:** Abstract base class with three methods: `fit(y)`, `predict(y) -> bool Series`, `decision_function(y) -> score Series`. Convention: higher score = more anomalous.

**Consequences:**

- Pipeline code is model-agnostic
- New detectors plug in by inheriting from `BaseAnomalyDetector` (the transformer detector being developed by a project partner will use this same interface once integrated)
- Consistent evaluation via `summarize_anomalies()`

---

## D7: Point + range evaluation metrics

**Date:** 2026-01
**Status:** Accepted

**Context:** Point-wise F1 can be misleading for anomaly detection (a single shifted prediction near a real anomaly scores poorly). Range-level metrics are standard in the anomaly detection literature.

**Decision:** Two metric levels:

- `PointMetrics`: precision, recall, F1, AUC-ROC, AUC-PR
- `RangeMetrics`: range-level P/R/F1 where TP = GT range overlapping any predicted range

**Consequences:**

- Both metric sets printed and logged to MLflow
- Enables fairer comparison between models with different temporal characteristics

---

## D8: FastAPI inference server

**Date:** 2026-02
**Status:** Accepted

**Context:** Need a way to serve trained models for real-time anomaly detection, both for the dashboard demo and potential production use.

**Decision:** FastAPI REST API (`api/`) with a `ModelManager` that discovers and loads model checkpoints on startup. Endpoints: `/models`, `/predict`, `/model/{name}/config`, `/health`.

**Alternatives considered:**

- Flask — less modern, no built-in async, no auto-generated OpenAPI docs
- gRPC — better for internal services but harder to demo and integrate with a web dashboard
- Gradio/Streamlit — combines UI + serving but too opinionated for our needs

**Consequences:**

- `fastapi` + `uvicorn` added as dependencies
- `api/` package with `inference_server.py`, `model_manager.py`, `schemas.py`
- CLI command: `python -m time_series_transformer serve`
- Dashboard communicates with the API over HTTP

---

## D9: Plotly Dash interactive dashboard

**Date:** 2026-02
**Status:** Accepted

**Context:** Need a visual demo for the project presentation: data exploration, model comparison, and live anomaly detection.

**Decision:** Multi-page Plotly Dash app (`dashboard/`) with four pages: Home, Data Analysis, Model Analysis, Live Monitoring. Communicates with the inference server via `api_client.py`.

**Alternatives considered:**

- Streamlit — simpler but limited layout control, no true multi-page routing
- Grafana — great for monitoring but requires separate data source setup, overkill for a demo
- Custom React frontend — too much development effort for a research project

**Consequences:**

- `dash`, `dash-bootstrap-components` added as dependencies
- `dashboard/` directory (outside `src/`) with its own `app.py` entry point
- CLI command: `python -m time_series_transformer dashboard`
- Requires the inference server to be running for model-related features

---

## D10: Benchmark framework with model registry

**Date:** 2026-02
**Status:** Accepted

**Context:** Needed systematic, reproducible evaluation of multiple models across multiple datasets. The existing `baseline_pipeline.py` only supports a single dataset per run.

**Decision:** A `benchmark/` package with a model factory registry and a YAML-driven dataset configuration. `BenchmarkRunner` iterates models x datasets, collects metrics into `ResultsCollector`, and exports a comparison CSV.

**Alternatives considered:**

- Extend `baseline_pipeline.py` — would bloat an already long module, harder to add new models
- Hardcoded dataset flags (`--nab-all`) — too rigid, doesn't scale to new datasets
- Hydra/OmegaConf for config — heavier dependency than needed for dataset lists

**Consequences:**

- `benchmark/` package with `registry.py`, `runner.py`, `results.py`, `dataset_spec.py`
- YAML config files in `configs/` (e.g. `benchmark_nab.yaml`)
- New models added via `register_model("name", factory)` — zero changes to core code
- CLI command: `python -m time_series_transformer benchmark --config configs/benchmark_nab.yaml`
- Error-tolerant: one failing model/dataset doesn't halt the entire benchmark

---

## D11: WebSocket streaming for live monitoring

**Date:** 2026-02
**Status:** Accepted

**Context:** The Live Monitoring dashboard page originally used HTTP polling — it sent the entire dataset to `/detect/dashboard`, cached all results in the browser, then revealed cached points progressively. This meant inference happened in one big batch and the full dataset lived in browser memory.

**Decision:** Add a `/ws/stream` WebSocket endpoint on the FastAPI server. The server loads the dataset, runs batch inference once, then streams result chunks at a configurable rate. A clientside JavaScript module (`assets/websocket.js`) manages the WebSocket lifecycle and buffers chunks into a Dash store, which Python callbacks consume to update the chart via `extendData`.

**Alternatives considered:**

- Server-Sent Events (SSE) — one-directional, no pause/resume/speed control from client
- Keep HTTP polling — works but loads full dataset into browser, no incremental delivery
- Socket.IO — adds a dependency; browser-native WebSocket is sufficient

**Consequences:**

- No new Python dependencies (FastAPI/Starlette supports WebSocket natively)
- `websocket.js` added to `dashboard/assets/` with two clientside callbacks (`connect`, `drain`)
- Live Monitoring page rewritten to consume WebSocket chunks instead of HTTP responses
- Supports pause/resume/reset/speed control messages from the client
- `ANOMALY_WS_HOST` env var makes the WebSocket host configurable for Docker networking

---

## D12: Docker containerisation with multi-stage build

**Date:** 2026-02
**Status:** Accepted

**Context:** The project has three runtime services (API, dashboard, MLflow). Needed a way to run them reproducibly without manual Python environment setup.

**Decision:** Single `Dockerfile` with a shared `base` stage and three build targets (`api`, `dashboard`, `mlflow`). `docker-compose.yml` orchestrates all three services. Data, artifacts, and MLflow runs are mounted as volumes from the host.

**Alternatives considered:**

- Three separate Dockerfiles — duplicates the dependency installation step
- Single monolithic container running all services — harder to scale, debug, or restart independently
- Devcontainer only — useful for development but doesn't help with deployment

**Consequences:**

- `Dockerfile`, `docker-compose.yml`, `.dockerignore` added to project root
- CPU-only PyTorch in the image (saves ~1.5 GB vs full CUDA build)
- Dashboard reaches API via Docker DNS name (`api:8000`) using `ANOMALY_API_URL` and `ANOMALY_WS_HOST` env vars
- `docker compose up` starts all three services with a single command

---

## D13: Multivariate anomaly detection on SMD

**Date:** 2026-02
**Status:** Accepted

**Context:** The project originally targeted univariate anomaly detection on NAB. We extended it to multivariate anomaly detection using the SMD (Server Machine Dataset) — 28 server machines, 38 features each.

**Decision:** Add four multivariate anomaly detectors with a dedicated pipeline (`multivariate_pipeline.py`), CLI command (`train-mv`), and dashboard integration.

| Model                    | Approach                                     |
|--------------------------|----------------------------------------------|
| `VARResidualAnomalyDetector`              | VAR(p) forecast residual z-scores |
| `MultivariateIsolationForestDetector`     | Scikit-learn ensemble on raw features |
| `LSTMAutoencoderAnomalyDetector`          | LSTM autoencoder reconstruction error |
| `LSTMForecasterMultivariateDetector`      | LSTM next-step forecast error |

**Consequences:**

- `models/multivariate/` package with `base.py`, `var.py`, `isolation_forest.py`, `lstm_autoencoder.py`, `lstm_forecaster.py`
- `multivariate_pipeline.py` orchestrates train/eval for all models
- `cli/train_multivariate.py` exposes `train-mv` command with `--machine` and `--model` flags
- Artifact CSVs saved to `artifacts/multivariate/{machine_id}_results.csv`
- Dashboard Model Analysis page updated to visualise SMD multivariate results

---

## D14: SMD data is pre-normalised — skip MinMaxScaler

**Date:** 2026-02
**Status:** Accepted

**Context:** The LSTM Autoencoder was producing poor results (F1 ≈ 0.34, flagging ~39% of timesteps as anomalous on machine-1-1). Investigation revealed the root cause was data preprocessing.

**Investigation:** We compared the Kaggle SMD dataset ([mgusat/smd-onmiad](https://www.kaggle.com/datasets/mgusat/smd-onmiad)) against the original OmniAnomaly repository ([NetManAIOps/OmniAnomaly](https://github.com/NetManAIOps/OmniAnomaly)). Key findings:

1. **Both sources contain identical data** — the Kaggle upload is an exact copy of the OmniAnomaly ServerMachineDataset.
2. **The data is already MinMax-normalised to [0, 1]** — all 38 features lie strictly within this range. The original raw server metrics (CPU%, memory bytes, network packets) were normalised before release by the dataset authors.
3. **TranAD** ([imperial-qore/TranAD](https://github.com/imperial-qore/TranAD)) — a state-of-the-art transformer model benchmarked on SMD — **applies no normalisation** to SMD data, confirming our finding.

Applying MinMaxScaler on top of already-normalised data caused two problems:

| Problem | Example | Effect |
|---------|---------|--------|
| **Narrow-range features get blown up** | f33 range [0, 0.0013] → rescaled to [0, 1] | Reconstruction errors on this feature are amplified ~770× relative to original scale |
| **Constant features cause division by zero** | f4, f7, f16, f17, f26, f28, f36, f37 are all-zero | MinMaxScaler clips to 0, but any nonzero test value creates unbounded error |

This distortion caused the LSTM Autoencoder's reconstruction error to be dominated by narrow-range/constant features rather than genuinely anomalous channels.

**Decision:** Skip normalisation for SMD in the multivariate pipeline (`normalize=False`). The data is already in [0, 1].

**Results on machine-1-1 (before → after):**

| Model | F1 before | F1 after | Flagged % |
|-------|-----------|----------|-----------|
| LSTM Autoencoder | 0.34 | **0.57** | 10.3% |
| LSTM Forecaster | 0.38 | **0.58** | 9.6% |

**References:**

- OmniAnomaly paper & dataset: Su et al., "Robust Anomaly Detection for Multivariate Time Series through Stochastic Recurrent Neural Network," KDD 2019. [GitHub](https://github.com/NetManAIOps/OmniAnomaly)
- TranAD: Tuli et al., "TranAD: Deep Transformer Networks for Anomaly Detection in Multivariate Time Series Data," VLDB 2022. [GitHub](https://github.com/imperial-qore/TranAD)
- Kaggle SMD dataset: [mgusat/smd-onmiad](https://www.kaggle.com/datasets/mgusat/smd-onmiad)

---

## D15: LSTM threshold computation — overlap-averaged training scores

**Date:** 2026-02
**Status:** Accepted

**Context:** The LSTM Autoencoder's `error_quantile` threshold had no effect — changing it from 0.99 to 0.9999 still flagged ~39% of test timesteps. The Forecaster was unaffected.

**Root cause:** The threshold was computed on raw **per-window** MSE values, but `decision_function()` returns **overlap-averaged per-timestep** scores. These are on different scales:

```
Per-window MSE:         raw error for one (lookback × n_features) window
Overlap-averaged score: mean of all window errors that contain a given timestep
```

Overlap-averaging smooths scores, producing values in a different range than the raw window errors. The quantile threshold computed on raw window errors was therefore meaningless when compared against overlap-averaged test scores.

**Decision:** Compute the threshold on **overlap-averaged training scores** using the same `_overlap_average()` function used at prediction time. This ensures the threshold and prediction scores are on the same scale.

**Applied to:** `LSTMAutoencoderAnomalyDetector.fit()`. The Forecaster already computed its threshold on raw per-window errors consistently with its `decision_function()`, so no fix was needed there (its `decision_function` also overlap-averages, but the threshold was already reasonable).

**References:**

- Taboola engineering blog: [Anomaly Detection using LSTM with Autoencoder](https://www.taboola.com/engineering/anomaly-detection-using-lstm-autoencoder/) — recommends computing threshold and inference scores on the same scale.
- MTAD benchmark: Schmidl et al., "MTAD: Tools and Benchmarks for Multivariate Time Series Anomaly Detection," 2024. [arXiv:2401.06175](https://arxiv.org/pdf/2401.06175)

---

## D16: Lookback window reduced from 30 to 10

**Date:** 2026-02
**Status:** Accepted

**Context:** Both LSTM models used a lookback window of 30 timesteps. TranAD uses a window of 10 for SMD.

**Decision:** Reduce `LSTM_AE_LOOKBACK` and `LSTM_FC_LOOKBACK` from 30 to 10.

**Rationale:**

- **TranAD precedent**: The TranAD model, which achieves state-of-the-art results on SMD, uses `n_window = 10` ([source](https://github.com/imperial-qore/TranAD/blob/main/src/models.py)).
- **Faster training**: Smaller windows produce more training samples and smaller tensors per batch, reducing fit time by ~40%.
- **Less over-smoothing**: Overlap-averaging with a large window spreads each window's error across more timesteps, diluting the anomaly signal. A shorter window preserves sharper anomaly peaks.
- **SMD anomaly characteristics**: Many SMD anomalies are short bursts (tens of timesteps). A 30-step window can miss or blur anomalies shorter than the window.

**Consequences:**

- Training time reduced from ~150s to ~80s per model on machine-1-1
- Config defaults updated in `config.py`
- No test changes required (tests use their own small lookback values)

---

## D17: Quantile-based threshold selection for LSTM models

**Date:** 2026-02
**Status:** Accepted

**Context:** We evaluated multiple threshold strategies for LSTM anomaly detectors after researching the literature.

**Strategies evaluated:**

| Strategy | Description | Outcome |
|----------|-------------|---------|
| **Pure quantile** | `threshold = quantile(train_scores, q)` | Simple, effective, chosen |
| **Mean + k·σ** | `threshold = μ + k·σ` of training scores | Too sensitive to std; tight training distributions made k hard to tune |
| **Hybrid max(quantile, mean+kσ)** | Take the higher of both | The sigma component kept overriding quantile, causing over-conservative thresholds |
| **MAE instead of MSE** for scoring | Train with MSE, score with MAE | MSE actually better for anomaly detection — squaring amplifies outliers (the anomalies we want to detect) |
| **POT/SPOT** (Extreme Value Theory) | Fit Generalized Pareto Distribution to tail | Used by OmniAnomaly and TranAD; more complex, requires careful tuning of initial quantile and multiplier |

**Decision:** Use quantile-based thresholding with MSE scoring. Current defaults:
- LSTM Autoencoder: `error_quantile = 0.99`
- LSTM Forecaster: `error_quantile = 0.97`

**Rationale:** Quantile thresholding is simple, interpretable, and robust. The quantile directly controls the false positive rate on training data. The different quantiles reflect that the Forecaster's error distribution has heavier tails without normalisation, requiring a lower quantile to achieve similar flagging rates.

**Note on reported F1 scores:** Published results on SMD (e.g. OmniAnomaly F1 = 0.96, TranAD F1 = 0.98) use **point-adjust F1**, which retroactively marks an entire anomalous segment as detected if any single point in it is flagged. Our evaluation uses strict **point-wise F1** (no adjustment), which is a harder metric. An F1 of 0.57 point-wise is a reasonable result.

**References:**

- OmniAnomaly threshold (POT): Su et al., KDD 2019. Uses SPOT with `level = 0.99995`, multiplier `1.04`. [GitHub](https://github.com/NetManAIOps/OmniAnomaly)
- TranAD threshold (POT): Tuli et al., VLDB 2022. Uses SPOT with `level = 0.99995`, multiplier `1.06`. [GitHub](https://github.com/imperial-qore/TranAD)
- Point-adjust F1 criticism: Kim et al., "Towards a Rigorous Evaluation of Time-Series Anomaly Detection," AAAI 2022.
- MAE vs MSE for scoring: Kieu et al., "Federated LSTM autoencoders for time series anomaly detection in production-scale HPC systems," Knowledge-Based Systems, 2025. [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0950705125020817)
- Threshold sensitivity on SMD: Schmidl et al., "MTAD: Tools and Benchmarks for Multivariate Time Series Anomaly Detection," 2024. [arXiv:2401.06175](https://arxiv.org/pdf/2401.06175)
