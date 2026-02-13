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
