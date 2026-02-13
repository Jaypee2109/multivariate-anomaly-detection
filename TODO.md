# TODO — Julian

## Datasets

- [ ] Consider dropping NAB — too few labels for proper evaluation.
      Keep for now until decision is final.
- [ ] Verify SMAP/MSL and SMD label coverage is sufficient for all metrics

## Evaluation Framework

- [x] F1@point (precision, recall, F1) — `evaluation.py`
- [x] F1@range (range-level precision, recall, F1) — `evaluation.py`
- [x] AUROC / AUPRC — `evaluation.py`
- [x] **Dynamic evaluation harness** — `benchmark/` package + `benchmark` CLI command
      (`python -m time_series_transformer benchmark --dataset nab --mlflow`) - Model registry with `register_model()` for adding new models - Multi-dataset support with auto NAB discovery - Console table + CSV export
- [ ] Add new models to registry (TranAD, custom transformer, ...)

## Inference / MLOps

- [x] FastAPI inference server (REST)
- [ ] WebSocket / SSE streaming endpoint
- [ ] Dockerfile + docker-compose

## Dashboard / Demo

- [x] Home page
- [x] Data Analysis page
- [x] Model Analysis page
- [x] Live Monitoring (simulated streaming via polling + cache)

## Report

- [ ] Final report draft
