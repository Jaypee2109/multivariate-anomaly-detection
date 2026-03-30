# TODO — Julian

## Datasets

- [x] Consider dropping NAB — too few labels for proper evaluation.
- [x] Verify SMAP/MSL and SMD label coverage is sufficient for all metrics

## Evaluation Framework

- [x] F1@point (precision, recall, F1) — `evaluation.py`
- [x] F1@range (range-level precision, recall, F1) — `evaluation.py`
- [x] AUROC / AUPRC — `evaluation.py`
- [x] **Dynamic evaluation framework** — `benchmark/` package + `benchmark` CLI command
      (`python -m time_series_transformer benchmark --dataset nab --mlflow`) - Model registry with `register_model()` for adding new models - Multi-dataset support with auto NAB discovery - Console table + CSV export
- [x] Add TranAD to multivariate pipeline and registry (D18)
- [x] Custom transformer detector (partner WIP)

## Inference / MLOps

- [x] FastAPI inference server (REST)
- [x] WebSocket streaming endpoint (`/ws/stream`) — server-side dataset loading + batch inference, chunked delivery with pause/resume/reset/speed controls
- [x] Dockerfile + docker-compose — multi-stage build (api, dashboard, mlflow targets) + `docker compose up`

## Dashboard / Demo

- [x] Home page
- [x] Data Analysis page
- [x] Model Analysis page
- [x] Live Monitoring (WebSocket streaming from inference server)

## Report

- [ ] Final report draft
