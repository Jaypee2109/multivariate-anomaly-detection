# Project Overview: Time Series Anomaly Detection with Transformer Models

> **Authors:** Julian, Lars
> **University:** University of Leipzig, Master's Project
> **Python:** >= 3.11 | **Framework:** PyTorch >= 2.0
> **Package:** `time-series-transformer` v0.1.0

---

## 1. Project Structure

```
Transformer/
├── pyproject.toml                 # Build configuration, editable install via setuptools
├── requirements.txt               # Pinned dependencies (PyTorch, scikit-learn, MLflow, FastAPI, Dash, ...)
├── .env.example                   # All configurable env vars with defaults
├── Dockerfile                     # Multi-stage build (api / dashboard / mlflow targets)
├── docker-compose.yml             # Orchestration of all three services
│
├── configs/                       # YAML benchmark configurations
│   ├── benchmark_nab.yaml         # 7 NAB realKnownCause datasets
│   └── benchmark_smd.yaml         # 28 SMD machines (3 groups)
│
├── data/
│   ├── raw/                       # Raw data (Kaggle downloads, git-ignored)
│   │   ├── nab/                   #   NAB time series (univariate)
│   │   ├── nasa_smap_msl/         #   NASA SMAP/MSL telemetry data
│   │   └── smd/                   #   Server Machine Dataset (38 features)
│   ├── processed/                 # Preprocessed CSVs (standardized, datetime index)
│   │   ├── nab/                   #   Processed NAB files
│   │   ├── smd/                   #   SMD with named columns (train/test/test_label)
│   │   └── nasa_smap_msl/         #   NASA labels
│   └── labels/
│       └── nab/realKnownCause.json  # Ground-truth anomaly timestamps (NAB format)
│
├── artifacts/                     # Pipeline outputs (git-ignored)
│   ├── anomalies/                 #   baseline_anomalies.csv (univariate)
│   ├── checkpoints/               #   Saved model checkpoints (.pt / .joblib)
│   │   └── multivariate/          #   Per machine: {model}.pt
│   └── multivariate/              #   Per-machine and aggregated result CSVs
│       ├── machine-*_results.csv  #   Scores + predictions per machine
│       ├── metrics_per_machine.csv  # All metrics per (machine, model)
│       ├── metrics_average.csv    #   Mean +/- std across all machines
│       └── metrics_best_machine.csv # Best machine per model (by AUC-ROC)
│
├── dashboard/                     # Plotly Dash multi-page application
│   ├── app.py                     # Main Dash app with navbar navigation
│   ├── api_client.py              # HTTP client for the inference API
│   ├── datasets.py                # Dataset registry for the dashboard
│   ├── mlflow_loader.py           # Load MLflow data for comparisons
│   ├── assets/
│   │   ├── styles.css             # Custom CSS
│   │   └── websocket.js           # Client-side WebSocket manager (live monitoring)
│   └── pages/
│       ├── home.py                # System overview, model status
│       ├── data_analysis.py       # Interactive time series exploration
│       ├── model_analysis.py      # Model comparison + MLflow results
│       └── live_monitoring.py     # Real-time streaming via WebSocket
│
├── scripts/
│   └── aggregate_results.py       # Aggregates multivariate results across all machines
│
├── src/
│   ├── time_series_transformer/   # === Main package ===
│   │   ├── __init__.py
│   │   ├── __main__.py            # Entry point: python -m time_series_transformer
│   │   ├── config.py              # Central configuration (env-var overridable)
│   │   ├── baseline_pipeline.py   # Univariate training pipeline
│   │   ├── multivariate_pipeline.py # Multivariate training pipeline (SMD)
│   │   ├── evaluation.py          # Point, range, point-adjust metrics, Best-F1, latency
│   │   ├── split.py               # Time-ordered train/test split
│   │   ├── mlflow_utils.py        # MLflow setup + logging helpers
│   │   ├── exceptions.py          # Custom exceptions
│   │   ├── logging_config.py      # Structured logging setup
│   │   │
│   │   ├── cli/                   # Subcommand-based CLI
│   │   │   ├── main.py            # Argparse dispatcher (register/run pattern)
│   │   │   ├── data.py            # Download + preprocessing
│   │   │   ├── train.py           # Univariate training
│   │   │   ├── train_multivariate.py # Multivariate training (SMD)
│   │   │   ├── benchmark.py       # Benchmark runner CLI
│   │   │   ├── eda.py             # Exploratory data analysis
│   │   │   ├── info.py            # Dataset/run inspection
│   │   │   ├── serve.py           # Start FastAPI server
│   │   │   └── dashboard.py       # Start dashboard
│   │   │
│   │   ├── data_pipeline/         # ETL: Download -> Load -> Preprocess -> Save
│   │   │   ├── data_download.py   # Kaggle download via kagglehub
│   │   │   ├── data_loading.py    # Recursive CSV loading
│   │   │   ├── preprocessing.py   # Standard scaling, datetime index
│   │   │   ├── data_save.py       # Save processed DataFrames
│   │   │   ├── labels.py          # NAB label JSON -> point labels
│   │   │   ├── smd_loading.py     # SMD-specific: .txt -> .csv, MinMaxScaler
│   │   │   └── pipeline.py        # Orchestration of the full pipeline
│   │   │
│   │   ├── models/
│   │   │   ├── baseline/          # Univariate detectors
│   │   │   │   ├── base.py        # BaseAnomalyDetector (ABC)
│   │   │   │   ├── rolling_zscore.py # Rolling Z-Score
│   │   │   │   ├── arima.py       # ARIMA Residual
│   │   │   │   ├── isolation_forest.py # Isolation Forest
│   │   │   │   └── lstm.py        # LSTM Forecast (univariate)
│   │   │   └── multivariate/      # Multivariate detectors
│   │   │       ├── base.py        # BaseMultivariateAnomalyDetector (ABC)
│   │   │       ├── isolation_forest.py # Multivariate Isolation Forest
│   │   │       ├── var.py         # VAR Residual
│   │   │       ├── lstm_forecaster.py  # LSTM Forecaster (multivariate)
│   │   │       ├── lstm_autoencoder.py # LSTM Autoencoder
│   │   │       ├── tranad.py      # TranAD (VLDB 2022)
│   │   │       └── custom_transformer.py # Custom Transformer (Time2Vec + Cross-Attention)
│   │   │
│   │   ├── api/                   # FastAPI inference server
│   │   │   ├── inference_server.py # REST + WebSocket endpoints
│   │   │   ├── model_manager.py   # Checkpoint loading + model management
│   │   │   └── schemas.py         # Pydantic request/response schemas
│   │   │
│   │   ├── benchmark/             # Benchmark framework
│   │   │   ├── registry.py        # Model factory registry
│   │   │   ├── dataset_spec.py    # DatasetSpec / MultivariateDatasetSpec
│   │   │   ├── runner.py          # BenchmarkRunner (models x datasets)
│   │   │   └── results.py         # BenchmarkResult + ResultsCollector
│   │   │
│   │   ├── analysis/
│   │   │   └── eda.py             # EDA visualizations
│   │   │
│   │   └── utils/
│   │       ├── anomaly_io.py      # Save anomaly artifacts
│   │       ├── data_validation.py # CSV validation
│   │       └── startup_checks.py  # Pre-flight checks for CLI commands
│   │
│   ├── scratch_transformer/       # Experimental: NLP Transformer (Lars)
│   │   ├── transformer.py         # Word-level Transformer from scratch
│   │   ├── positional_encoding.py # Sinusoidal PE
│   │   ├── rotary_encoding.py     # RoPE implementation
│   │   └── inference.py / inference_rotary.py
│   │
│   └── scratch_time_series_transformer/  # Experimental: Early TS Transformer prototypes
│       ├── transformer.py
│       ├── positional_encoding.py
│       └── main.py
│
├── Transformer/                   # Older standalone experiments (pre-package refactoring)
│   ├── InitialTransformer/        # NLP Transformer (copy, standalone)
│   └── TimeSeriesTransformer/     # Early TS Transformer with Time2Vec, DDP
│       ├── transformer.py         # First custom transformer architecture
│       ├── time2vec.py            # Time2Vec embedding
│       ├── learnableTime2Vec.py   # Learnable Time2Vec (Lars)
│       ├── main_electricity_data_ddp.py # Distributed Data Parallel training
│       └── SC_main.py             # Miscellaneous experiments
│
├── tests/                         # Pytest test suite
│   ├── conftest.py                # Shared fixtures
│   ├── test_config.py             # Config tests
│   ├── test_evaluation.py         # Evaluation metrics tests
│   ├── test_custom_transformer.py # Custom Transformer unit tests
│   ├── test_tranad.py             # TranAD unit tests
│   ├── test_multivariate_models.py # Multivariate model tests
│   ├── test_lstm_forecaster.py    # LSTM Forecaster tests
│   ├── test_smd_loading.py        # SMD loading tests
│   ├── test_labels.py             # Label loading tests
│   ├── test_api/                  # API tests (health, predict, model_info)
│   └── ...
│
└── docs/
    └── architecture.md            # Architecture documentation
```

---

## 2. Pipeline Architecture

### 2.1 High-Level Overview: From Raw Data to Anomaly Detection

The pipeline follows these main steps:

```
[1] Data Download       ->  [2] Preprocessing        ->  [3] Train/Test Split
      (Kaggle)                (Scaling, Indexing)          (time-ordered)
                                     |
                                     v
[4] Model Training      ->  [5] Anomaly Scoring      ->  [6] Evaluation
  (fit on training data)      (decision_function)         (metric computation)
                                     |
                                     v
                           [7] Artifact Export        ->  [8] Serving / Dashboard
                             (CSVs, checkpoints)          (FastAPI + Dash)
```

### 2.2 Step-by-Step with Responsible Modules

| Step | Module | Description |
|------|--------|-------------|
| **1. Download** | [data_download.py](src/time_series_transformer/data_pipeline/data_download.py) | Downloads from Kaggle via `kagglehub`, copies to `data/raw/` |
| **2a. Load (univariate)** | [data_loading.py](src/time_series_transformer/data_pipeline/data_loading.py) | Recursive CSV loading under `data/raw/<dataset>/` |
| **2b. Load (SMD)** | [smd_loading.py](src/time_series_transformer/data_pipeline/smd_loading.py) | SMD-specific: .txt -> DataFrame with 38 named columns |
| **3. Preprocessing** | [preprocessing.py](src/time_series_transformer/data_pipeline/preprocessing.py) | Standard scaling + datetime index conversion |
| **4. Split** | [split.py](src/time_series_transformer/split.py) | Time-ordered split (default 70/30), SMD uses pre-split |
| **5. Training** | [baseline_pipeline.py](src/time_series_transformer/baseline_pipeline.py), [multivariate_pipeline.py](src/time_series_transformer/multivariate_pipeline.py) | Orchestration: fit(), predict(), decision_function() |
| **6. Evaluation** | [evaluation.py](src/time_series_transformer/evaluation.py) | Point/range/PA metrics, Best-F1, latency |
| **7. Export** | [anomaly_io.py](src/time_series_transformer/utils/anomaly_io.py) | Artifact CSVs + checkpoints |
| **8. Serving** | [inference_server.py](src/time_series_transformer/api/inference_server.py) | REST + WebSocket API |

### 2.3 Two Pipeline Paths

**Univariate Path** (`train` command):
- CSV with `timestamp,value` columns -> train/test split (70/30) -> baselines (ARIMA, Isolation Forest, LSTM)
- Controlled by: [baseline_pipeline.py:89-255](src/time_series_transformer/baseline_pipeline.py#L89-L255)

**Multivariate Path** (`train-mv` command):
- SMD data (38 features, pre-split) -> multivariate models (IF, LSTM-AE, TranAD, Custom Transformer)
- Controlled by: [multivariate_pipeline.py:202-318](src/time_series_transformer/multivariate_pipeline.py#L202-L318)

---

## 3. Data Preprocessing

### 3.1 Loading Raw Data

**Univariate Data** (NAB): Recursive CSV loading with `os.walk()`, filtering macOS metadata (`__MACOSX`, `._` files).
- Reference: [data_loading.py:16-60](src/time_series_transformer/data_pipeline/data_loading.py#L16-L60)

**SMD Data** (multivariate): Conversion from headerless `.txt` files (comma-separated) into CSVs with 38 named columns:

```python
# smd_loading.py:22-61
SMD_COLUMN_NAMES = [
    "cpu_r", "load_1", "load_5", "load_15", "mem_shmem", "mem_u", "mem_u_e",
    "total_mem", "disk_q", "disk_r", "disk_rb", "disk_svc", "disk_u", "disk_w",
    "disk_wa", "disk_wb", "si", "so", "eth1_fi", "eth1_fo", "eth1_pi", "eth1_po",
    "tcp_tw", "tcp_use", "active_opens", "curr_estab", "in_errs", "in_segs",
    "listen_overflows", "out_rsts", "out_segs", "passive_opens", "retransegs",
    "tcp_timeouts", "udp_in_dg", "udp_out_dg", "udp_rcv_buf_errs", "udp_snd_buf_errs",
]
```

- Reference: [smd_loading.py:87-146](src/time_series_transformer/data_pipeline/smd_loading.py#L87-L146)

### 3.2 Normalization

**Univariate Data**: Standard scaling (z-score normalization) across all numeric columns:
- Reference: [preprocessing.py:41-58](src/time_series_transformer/data_pipeline/preprocessing.py#L41-L58)

```python
def standard_scale(df, exclude=None):
    for col in cols_to_scale:
        mean = df[col].mean()
        std = df[col].std()
        if std == 0 or pd.isna(std):
            df[col] = 0.0
        else:
            df[col] = (df[col] - mean) / std
```

**SMD Data**: MinMaxScaler (fit on train, transform on both splits). SMD data is already pre-normalized to [0,1]; the pipeline uses `normalize=False` by default:
- Reference: [smd_loading.py:181-227](src/time_series_transformer/data_pipeline/smd_loading.py#L181-L227)
- [multivariate_pipeline.py:216](src/time_series_transformer/multivariate_pipeline.py#L216): `normalize=False`

### 3.3 Sliding Windows

All time-series-based models use sliding windows. Window creation uses `np.lib.stride_tricks.sliding_window_view` for memory-efficient windowing:

| Model | Window Length (Lookback) | Configurable |
|-------|-------------------------|--------------|
| LSTM (univariate) | 72 | `LSTM_LOOKBACK` |
| LSTM Autoencoder | 30 | `LSTM_AE_LOOKBACK` |
| LSTM Forecaster (MV) | 30 | `LSTM_FC_LOOKBACK` |
| TranAD | 30 | `TRANAD_LOOKBACK` |
| Custom Transformer | 30 | `CUSTOM_TF_LOOKBACK` |

Example implementation (Custom Transformer):
- Reference: [custom_transformer.py:234-257](src/time_series_transformer/models/multivariate/custom_transformer.py#L234-L257)

### 3.4 Configurable Parameters

All parameters are overridable via environment variables (see [.env.example](.env.example)). Central configuration: [config.py](src/time_series_transformer/config.py)

| Parameter | Default | Env Variable |
|-----------|---------|--------------|
| Train/Test Ratio | 0.7 | `TRAIN_RATIO` |
| Random Seed | 42 | `RANDOM_STATE` |
| Data Directory | `PROJECT_ROOT/data` | `DATA_DIR` |
| Artifact Directory | `PROJECT_ROOT/artifacts` | `ARTIFACTS_DIR` |
| MLflow Experiment | `"Anomaly_Detection"` | `MLFLOW_EXPERIMENT_NAME` |

---

## 4. Model Architecture

### 4.1 Model Variant Overview

The project implements **10 anomaly detection models** in two categories:

#### Univariate Baselines (operate on `pd.Series`)

| Model | Class | File | Approach |
|-------|-------|------|----------|
| **Rolling Z-Score** | `RollingZScoreAnomalyDetector` | [rolling_zscore.py](src/time_series_transformer/models/baseline/rolling_zscore.py) | Rolling mean/std, z-score threshold |
| **ARIMA Residual** | `ARIMAResidualAnomalyDetector` | [arima.py](src/time_series_transformer/models/baseline/arima.py) | ARIMA(2,1,2) fit, residual z-scores |
| **Isolation Forest** | `IsolationForestAnomalyDetector` | [isolation_forest.py](src/time_series_transformer/models/baseline/isolation_forest.py) | sklearn ensemble (univariate) |
| **LSTM Forecast** | `LSTMForecastAnomalyDetector` | [lstm.py](src/time_series_transformer/models/baseline/lstm.py) | LSTM next-step forecast, MAE error |

#### Multivariate Models (operate on `pd.DataFrame` with 38 features)

| Model | Class | File | Approach |
|-------|-------|------|----------|
| **Isolation Forest (MV)** | `MultivariateIsolationForestDetector` | [multivariate/isolation_forest.py](src/time_series_transformer/models/multivariate/isolation_forest.py) | sklearn on raw feature matrix |
| **VAR Residual** | `VARResidualAnomalyDetector` | [multivariate/var.py](src/time_series_transformer/models/multivariate/var.py) | VAR(p) forecast residual z-scores |
| **LSTM Forecaster (MV)** | `LSTMForecasterMultivariateDetector` | [multivariate/lstm_forecaster.py](src/time_series_transformer/models/multivariate/lstm_forecaster.py) | LSTM next-step, MSE score |
| **LSTM Autoencoder** | `LSTMAutoencoderAnomalyDetector` | [multivariate/lstm_autoencoder.py](src/time_series_transformer/models/multivariate/lstm_autoencoder.py) | Encoder-decoder reconstruction error |
| **TranAD** | `TranADAnomalyDetector` | [multivariate/tranad.py](src/time_series_transformer/models/multivariate/tranad.py) | Transformer encoder + dual decoder (VLDB 2022) |
| **Custom Transformer** | `CustomTransformerDetector` | [multivariate/custom_transformer.py](src/time_series_transformer/models/multivariate/custom_transformer.py) | Time2Vec + cross-attention forecaster |

### 4.2 Detailed Model Architectures

#### 4.2.1 LSTM Autoencoder ([lstm_autoencoder.py:20-73](src/time_series_transformer/models/multivariate/lstm_autoencoder.py#L20-L73))

```
Input (B, seq_len, 38)
  -> Encoder LSTM (38 -> hidden=64, layers=1)
  -> Last Hidden State (B, 64)
  -> Linear (64 -> latent=32)
  -> Latent (B, 32)
  -> Repeat T times (B, T, 32)
  -> Decoder LSTM (32 -> 64)
  -> Linear (64 -> 38)
  -> Reconstruction (B, seq_len, 38)
```

**Hyperparameters:**

| Parameter | Default | Env Variable |
|-----------|---------|--------------|
| Lookback | 30 | `LSTM_AE_LOOKBACK` |
| Hidden Size | 64 | `LSTM_AE_HIDDEN_SIZE` |
| Latent Dim | 32 | `LSTM_AE_LATENT_DIM` |
| Num Layers | 1 | `LSTM_AE_NUM_LAYERS` |
| Dropout | 0.0 | `LSTM_AE_DROPOUT` |
| Epochs | 30 | `LSTM_AE_EPOCHS` |
| Batch Size | 64 | `LSTM_AE_BATCH_SIZE` |
| Learning Rate | 1e-3 | `LSTM_AE_LR` |
| Error Quantile | 0.99 | `LSTM_AE_ERROR_QUANTILE` |
| Score Metric | mse | `LSTM_AE_SCORE_METRIC` |

#### 4.2.2 TranAD ([tranad.py:158-247](src/time_series_transformer/models/multivariate/tranad.py#L158-L247))

Reproduction of the TranAD architecture (Tuli et al., VLDB 2022):

```
Input (seq_len, B, F=38)
  Phase 1:
    src = cat(input, zeros) -> (seq_len, B, 2F)
    src = src * sqrt(F) + PositionalEncoding
    memory = TransformerEncoder(src)    # Custom layer: no LayerNorm, LeakyReLU
    tgt = last_timestep.repeat(1,1,2)
    x1 = FCN(TransformerDecoder1(tgt, memory))  # Sigmoid output

  Phase 2 (Self-Conditioning):
    c = (x1 - input)^2                 # Squared error as conditioning signal
    src2 = cat(input, c)
    memory2 = TransformerEncoder(src2)
    x2 = FCN(TransformerDecoder2(tgt, memory2))

  Output: (x1, x2), each (1, B, F)
```

Key implementation details:
- **Custom Transformer layers** without LayerNorm, using LeakyReLU instead of ReLU (faithful to the original)
  - Reference: [tranad.py:70-151](src/time_series_transformer/models/multivariate/tranad.py#L70-L151)
- **d_model = 2F** (76 for SMD with 38 features)
- **Default n_heads = n_features = 38** (each head sees 2 dimensions)
- **Positional Encoding**: Sin + Cos added (not interleaved), faithful to the original

**Hyperparameters:**

| Parameter | Default | Env Variable |
|-----------|---------|--------------|
| Lookback | 30 | `TRANAD_LOOKBACK` |
| N Heads | 0 (=auto=38) | `TRANAD_N_HEADS` |
| Dim Feedforward | 16 | `TRANAD_DIM_FF` |
| Num Layers | 1 | `TRANAD_NUM_LAYERS` |
| Dropout | 0.1 | `TRANAD_DROPOUT` |
| Epochs | 15 | `TRANAD_EPOCHS` |
| Batch Size | 128 | `TRANAD_BATCH_SIZE` |
| Learning Rate | 1e-4 | `TRANAD_LR` |
| Error Quantile | 0.99 | `TRANAD_ERROR_QUANTILE` |

#### 4.2.3 Custom Transformer with Time2Vec ([custom_transformer.py:72-186](src/time_series_transformer/models/multivariate/custom_transformer.py#L72-L186))

Custom architecture, adapted from Lars' Time2Vec work:

```
Input: (B, seq_len=30, 38 features) + offsets (B,)

1. Time2Vec Temporal Encoding:
   abs_pos = offset + arange(seq_len)
   minute_of_hour = (pos % 60) / 59.0      # [0, 1]
   hour_of_day   = ((pos // 60) % 24) / 23.0  # [0, 1]
   t2v = LearnableTime2Vec(in_dim=2, out_dim=16)  # (B, seq_len, 16)

2. Input Projection:
   x = cat(features, t2v)  # (B, seq_len, 38+16=54)
   x = Linear(54 -> 64)    # (B, seq_len, model_dim=64)

3. Transformer Encoder (causal mask):
   encoder_layer = TransformerEncoderLayer(d_model=64, nhead=4, ff=256, dropout=0.1)
   x = TransformerEncoder(x, num_layers=2, mask=causal_mask)

4. Cross-Attention Decoder:
   next_pos = offset + seq_len
   query = Time2Vec(next_pos) -> Linear(16 -> 64)    # (B, 1, 64)
   attended = MultiheadAttention(query, key=x, value=x)  # (B, 1, 64)

5. MLP Decoder:
   pred = Linear(64) -> ReLU -> Linear(38)  # (B, 38)
```

**LearnableTime2Vec** ([custom_transformer.py:38-59](src/time_series_transformer/models/multivariate/custom_transformer.py#L38-L59)):

```python
class LearnableTime2Vec(nn.Module):
    def __init__(self, in_dim=1, out_dim=16):
        self.w0 = nn.Parameter(torch.randn(in_dim))    # Linear component
        self.b0 = nn.Parameter(torch.randn(1))
        self.W = nn.Parameter(torch.randn(out_dim-1, in_dim))  # Sinusoidal
        self.B = nn.Parameter(torch.randn(out_dim-1))

    def forward(self, t):
        v0 = t @ w0 + b0             # Linear trend
        vp = sin(t @ W^T + B)        # Learnable periodic components
        return cat([v0, vp], dim=-1)  # (..., out_dim)
```

**Hyperparameters:**

| Parameter | Default | Env Variable |
|-----------|---------|--------------|
| Lookback | 30 | `CUSTOM_TF_LOOKBACK` |
| T2V Dim | 16 | `CUSTOM_TF_T2V_DIM` |
| Model Dim | 64 | `CUSTOM_TF_MODEL_DIM` |
| Num Heads | 4 | `CUSTOM_TF_NUM_HEADS` |
| Num Layers | 2 | `CUSTOM_TF_NUM_LAYERS` |
| Dim Feedforward | 256 | `CUSTOM_TF_DIM_FF` |
| Dropout | 0.1 | `CUSTOM_TF_DROPOUT` |
| Epochs | 15 | `CUSTOM_TF_EPOCHS` |
| Batch Size | 64 | `CUSTOM_TF_BATCH_SIZE` |
| Learning Rate | 1e-3 | `CUSTOM_TF_LR` |
| Error Quantile | 0.99 | `CUSTOM_TF_ERROR_QUANTILE` |

---

## 5. Training

### 5.1 General Training Flow

1. **Seeding** (`_seed_everything`): `np.random.seed`, `torch.manual_seed`, `torch.cuda.manual_seed_all`
   - Reference: [multivariate_pipeline.py:85-89](src/time_series_transformer/multivariate_pipeline.py#L85-L89)

2. **Load data**: Pre-split SMD data via `load_smd_machine()` (no custom split needed)
   - Reference: [multivariate_pipeline.py:215-218](src/time_series_transformer/multivariate_pipeline.py#L215-L218)

3. **Instantiate models**: Factory pattern via `_build_model(key)`
   - Reference: [multivariate_pipeline.py:92-188](src/time_series_transformer/multivariate_pipeline.py#L92-L188)

4. **Training**: `model.fit(X_train)` -> `model.decision_function(X_test)` -> `model.predict(X_test)`

5. **Evaluation**: Compute all metrics (see Section 7)

6. **Export artifacts**: Results CSV + optional checkpoints

### 5.2 Loss Functions

| Model | Loss Function | Reference |
|-------|--------------|-----------|
| LSTM Autoencoder | `nn.MSELoss()` (reconstruction) | [lstm_autoencoder.py:163](src/time_series_transformer/models/multivariate/lstm_autoencoder.py#L163) |
| LSTM Forecaster | `nn.MSELoss()` (next-step) | [lstm_forecaster.py:135](src/time_series_transformer/models/multivariate/lstm_forecaster.py#L135) |
| TranAD | `(1/epoch)*MSE(x1) + (1-1/epoch)*MSE(x2)` — weighted shift from Decoder1 to Decoder2 | [tranad.py:376-379](src/time_series_transformer/models/multivariate/tranad.py#L376-L379) |
| Custom Transformer | `nn.MSELoss()` (forecast) | [custom_transformer.py:309](src/time_series_transformer/models/multivariate/custom_transformer.py#L309) |
| LSTM (univariate) | `nn.MSELoss()` | [lstm.py:146](src/time_series_transformer/models/baseline/lstm.py#L146) |

### 5.3 Optimizers and Learning Rate Schedules

| Model | Optimizer | LR Schedule | Gradient Clipping |
|-------|-----------|-------------|-------------------|
| LSTM Autoencoder | Adam (lr=1e-3) | None | No |
| LSTM Forecaster (MV) | Adam (lr=1e-3) | None | No |
| TranAD | AdamW (lr=1e-4, wd=1e-5) | StepLR (step=5, gamma=0.9) | No |
| Custom Transformer | Adam (lr=1e-3) | **OneCycleLR** (max_lr=1e-3, pct_start=0.1, cosine) | `clip_grad_norm_(1.0)` |
| LSTM (univariate) | Adam (lr=1e-3) | None | No |

**Custom Transformer Learning Rate Schedule** ([custom_transformer.py:301-308](src/time_series_transformer/models/multivariate/custom_transformer.py#L301-L308)):
```python
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    opt,
    max_lr=self.lr,
    steps_per_epoch=n_batches,
    epochs=self.epochs,
    pct_start=0.1,          # 10% warmup
    anneal_strategy="cos",  # Cosine annealing
)
```

### 5.4 Threshold Computation

All models compute the anomaly threshold as a **quantile of training scores**:

```python
# Example: LSTM Autoencoder (lstm_autoencoder.py:196)
self.threshold_ = float(np.quantile(train_scores, self.error_quantile))
```

Scores are mapped to timestep level via **overlap averaging**:
- Reference: [lstm_autoencoder.py:129-137](src/time_series_transformer/models/multivariate/lstm_autoencoder.py#L129-L137)

```python
def _overlap_average(self, window_errors, n):
    """Each window contributes its score to all contained timesteps."""
    for i, err in enumerate(window_errors):
        accum[i : i + T] += err
        counts[i : i + T] += 1
    return accum / counts
```

### 5.5 Distributed Training / SLURM

The project contains an early **Distributed Data Parallel (DDP)** prototype for multi-GPU training:
- Reference: [Transformer/TimeSeriesTransformer/main_electricity_data_ddp.py](Transformer/TimeSeriesTransformer/main_electricity_data_ddp.py)

This uses `torch.distributed` with the `nccl` backend. However, it is part of the experimental/older codebase and is **not** integrated into the main package. There is **no SLURM configuration** in the repository.

The main package uses single-GPU/CPU training with automatic device detection:
```python
self._device = "cuda" if torch.cuda.is_available() else "cpu"
```

---

## 6. Anomaly Detection

### 6.1 Scoring Approaches

| Approach | Models | Score Computation |
|----------|--------|-------------------|
| **Forecast Error** | LSTM (UV), LSTM Forecaster (MV), Custom Transformer | MSE/MAE between prediction and ground truth |
| **Reconstruction Error** | LSTM Autoencoder, TranAD | MSE between input and reconstruction |
| **Residual Z-Score** | ARIMA, VAR, Rolling Z-Score | Normalized deviation from model forecast |
| **Isolation Score** | Isolation Forest (UV/MV) | Inverted sklearn decision score |

### 6.2 Threshold Determination

**Quantile-based Threshold**: The standard method for all deep learning models. The threshold is determined as the upper quantile of training anomaly scores:

- LSTM Autoencoder: 99th percentile (`LSTM_AE_ERROR_QUANTILE=0.99`)
- TranAD: 99th percentile (`TRANAD_ERROR_QUANTILE=0.99`)
- Custom Transformer: 99th percentile (`CUSTOM_TF_ERROR_QUANTILE=0.99`)
- LSTM Forecaster (MV): 97th percentile (`LSTM_FC_ERROR_QUANTILE=0.97`)
- LSTM (univariate): 99.7th percentile (`LSTM_ERROR_QUANTILE=0.997`)

**Z-Score Threshold**: For statistical models (ARIMA: 8.5, VAR: 3.0, Rolling Z-Score: 1.8)

**Contamination Parameter**: For Isolation Forest (UV: 0.004, MV: 0.01)

**Best-F1 Threshold Search** ([evaluation.py:317-388](src/time_series_transformer/evaluation.py#L317-L388)):
Sweep over 100 quantiles between P90 and P100 of scores, selecting the threshold with maximum point-F1.

### 6.3 Score Aggregation (Window -> Timestep)

Since window-based models produce one score per window, **overlap averaging** is used: each timestep receives the average of all window scores that contain it. This smooths the scores and avoids edge artifacts.

### 6.4 TranAD-Specific Self-Conditioning

TranAD uses a two-phase anomaly detection approach:
1. **Phase 1**: Reconstruction without conditioning (c=0)
2. **Phase 2**: Reconstruction with `c = (x1 - src)^2` as conditioning signal
3. **Anomaly Score**: MSE of the Phase 2 reconstruction

The loss weighting shifts over epochs from Phase 1 to Phase 2:
```python
loss = (1/epoch) * MSE(x1, target) + (1 - 1/epoch) * MSE(x2, target)
```
- Reference: [tranad.py:376-379](src/time_series_transformer/models/multivariate/tranad.py#L376-L379)

---

## 7. Evaluation

### 7.1 Metrics

The project computes a comprehensive metric set, implemented in [evaluation.py](src/time_series_transformer/evaluation.py):

#### Point-Level Metrics ([evaluation.py:75-128](src/time_series_transformer/evaluation.py#L75-L128))

| Metric | Description |
|--------|-------------|
| **Precision** | TP / (TP + FP) — proportion of correctly flagged anomalies |
| **Recall** | TP / (TP + FN) — proportion of found anomalies |
| **F1** | Harmonic mean of precision and recall |
| **AUC-ROC** | Area under ROC curve (requires scores) |
| **AUC-PR** | Average precision (requires scores) |

#### Point-Adjust (PA) Metrics ([evaluation.py:222-298](src/time_series_transformer/evaluation.py#L222-L298))

Standard evaluation protocol from OmniAnomaly / TranAD / MTAD-GAT:
- If **any** point within a GT anomaly segment is correctly detected, the **entire** segment is marked as detected
- Significantly inflates recall compared to point-level evaluation
- Reference: [evaluation.py:222-271](src/time_series_transformer/evaluation.py#L222-L271)

#### Range-Level Metrics ([evaluation.py:141-204](src/time_series_transformer/evaluation.py#L141-L204))

| Metric | Description |
|--------|-------------|
| **Range Precision** | TP_ranges / (TP_ranges + FP_ranges) |
| **Range Recall** | TP_ranges / (TP_ranges + FN_ranges) |
| **Range F1** | Harmonic mean |
| **n_gt_ranges** | Number of ground-truth anomaly segments |
| **n_pred_ranges** | Number of predicted segments |
| **n_tp_ranges** | Correctly detected segments |

A GT range counts as detected (TP) if it overlaps with at least one predicted range.

#### Best-F1 with Oracle Threshold ([evaluation.py:317-388](src/time_series_transformer/evaluation.py#L317-L388))

Sweep over 100 thresholds in the range [P90, max(scores)], reporting:
- Optimal F1, corresponding threshold
- PA-F1 at the same threshold

#### Detection Latency ([evaluation.py:407-469](src/time_series_transformer/evaluation.py#L407-L469))

For each GT anomaly segment: offset (in timesteps) until first detection.
- `mean_latency`, `median_latency`
- `n_detected`, `n_missed`, `n_segments`
- Undetected segments count with full segment length (worst case)

### 7.2 Evaluation Protocol in the Pipeline

The multivariate pipeline ([multivariate_pipeline.py:265-303](src/time_series_transformer/multivariate_pipeline.py#L265-L303)) computes for each model:
1. Point metrics (P, R, F1, AUC-ROC, AUC-PR)
2. Point-adjust metrics (PA-P, PA-R, PA-F1)
3. Best-F1 threshold search
4. Detection latency

### 7.3 Aggregation Across Machines

The script [aggregate_results.py](scripts/aggregate_results.py) computes:
1. **Per-Machine Metrics**: All above metrics per (machine, model) combination
2. **Average Metrics**: Mean +/- standard deviation across all 28 machines
3. **Best Machine**: Best machine per model (by AUC-ROC)

---

## 8. Datasets

### 8.1 Supported Datasets

| Dataset | Type | Source | Features | Size |
|---------|------|--------|----------|------|
| **SMD** (Server Machine Dataset) | Multivariate | Kaggle: `mgusat/smd-onmiad` | 38 server metrics | 28 machines, ~28k train + ~28k test each |
| **NAB** (Numenta Anomaly Benchmark) | Univariate | Kaggle: `boltzmannbrain/nab` | 1 (value) | ~50+ time series (various categories) |
| **NASA SMAP/MSL** | Multivariate | Kaggle: `patrickfleith/nasa-anomaly-detection-dataset-smap-msl` | variable | Spacecraft telemetry data |

### 8.2 Dataset Configuration

Kaggle datasets are registered in [config.py:80-84](src/time_series_transformer/config.py#L80-L84):
```python
KAGGLE_DATASETS = {
    "smd": "mgusat/smd-onmiad",
    "nasa_smap_msl": "patrickfleith/nasa-anomaly-detection-dataset-smap-msl",
    "nab": "boltzmannbrain/nab",
}
```

Benchmark configurations as YAML:
- **NAB** ([benchmark_nab.yaml](configs/benchmark_nab.yaml)): 7 time series from `realKnownCause` with ground-truth labels
- **SMD** ([benchmark_smd.yaml](configs/benchmark_smd.yaml)): All 28 machines (3 groups: machine-1-*, machine-2-*, machine-3-*)

### 8.3 SMD Structure in Detail

| Directory | Content |
|-----------|---------|
| `train/machine-X-Y.{txt,csv}` | Training data (38 features, no anomalies) |
| `test/machine-X-Y.{txt,csv}` | Test data (38 features, with anomalies) |
| `test_label/machine-X-Y.{txt,csv}` | Binary labels (0/1 per timestep) |
| `interpretation_label/machine-X-Y.txt` | Which features are anomalous in which segment |

SMD feature names (38 server monitoring metrics):
CPU, load (1/5/15 min), memory, disk (queue/read/write/utilization), network (eth1 in/out), TCP (time-wait/use/established), UDP, etc.
- Reference: [smd_loading.py:22-61](src/time_series_transformer/data_pipeline/smd_loading.py#L22-L61)

---

## 9. Key Design Decisions

### D1: Unified Detector API

All models follow the same interface (`fit()` / `predict()` / `decision_function()`), inspired by scikit-learn. This enables interchangeable models in pipelines, benchmarks, and the API.
- Univariate: [baseline/base.py](src/time_series_transformer/models/baseline/base.py) — `BaseAnomalyDetector`
- Multivariate: [multivariate/base.py](src/time_series_transformer/models/multivariate/base.py) — `BaseMultivariateAnomalyDetector`

### D2: Environment Variable Configuration

All hyperparameters are controllable via env vars ([config.py](src/time_series_transformer/config.py)), with optional `.env` file support via `python-dotenv`. This enables:
- Simple configuration in Docker containers
- No code changes for hyperparameter tuning
- Reproducible experiments

### D3: Overlap Averaging for Window Scores

Instead of only scoring the last timestep of a window, all windows containing a timestep are averaged. This produces smoother, more stable anomaly scores.

### D4: Faithful TranAD Reimplementation

The TranAD implementation uses **custom Transformer layers without LayerNorm** and with **LeakyReLU** instead of ReLU ([tranad.py:70-151](src/time_series_transformer/models/multivariate/tranad.py#L70-L151)), to ensure fair comparability with the original paper.

### D5: Time2Vec Instead of Sinusoidal PE in Custom Transformer

The Custom Transformer uses **learnable Time2Vec** with derived time features (`minute_of_hour`, `hour_of_day`), computed from absolute position indices. This allows the model to learn periodic patterns at different time scales, and is more flexible than fixed sinusoidal positional encodings.
- Reference: [custom_transformer.py:134-146](src/time_series_transformer/models/multivariate/custom_transformer.py#L134-L146)

### D6: Cross-Attention Decoder in Custom Transformer

Instead of an autoregressive decoder, the Custom Transformer uses a **cross-attention mechanism**: the query comes from the Time2Vec representation of the next position, key/value from the encoder output. This is more efficient than a full decoder and enables direct next-step prediction.
- Reference: [custom_transformer.py:122-125](src/time_series_transformer/models/multivariate/custom_transformer.py#L122-L125)

### D7: SMD Without Re-Normalization

SMD data is already pre-normalized to [0,1]. The pipeline sets `normalize=False` to avoid double normalization.
- Reference: [multivariate_pipeline.py:216](src/time_series_transformer/multivariate_pipeline.py#L216)

### D8: Multi-Metric Evaluation

Instead of a single F1 value, 5 evaluation protocols are computed in parallel:
1. Point-level (standard)
2. Point-adjust (comparability with literature)
3. Range-level (practically relevant)
4. Best-F1 (oracle threshold, upper bound)
5. Detection latency (operationally relevant)

### D9: Factory Registry for Benchmarks

New models can be registered via `register_model("name", factory_fn)` without modifying core code. The benchmark runner automatically iterates over all registered models.
- Reference: [benchmark/registry.py](src/time_series_transformer/benchmark/registry.py)

### D10: Docker Multi-Stage Build

Three separate targets (api, dashboard, mlflow) share a common base stage with CPU-only PyTorch, reducing image size.
- Reference: [Dockerfile](Dockerfile), [docker-compose.yml](docker-compose.yml)

### D11: WebSocket Streaming for Live Monitoring

The API supports real-time streaming of anomaly detection results via WebSocket with pause/resume/reset/speed controls. The client (dashboard) uses a JavaScript-based buffer drain mechanism.
- Reference: [inference_server.py:446-687](src/time_series_transformer/api/inference_server.py#L446-L687)

---

## 10. Results

### 10.1 Aggregated Results (28 SMD Machines)

From [metrics_average.csv](artifacts/multivariate/metrics_average.csv):

| Model | AUC-ROC | F1 (Point) | PA-F1 | Best-F1 | Precision | Recall | Detection Rate | Det. Latency | Inference Time |
|-------|---------|------------|-------|---------|-----------|--------|---------------|-------------|----------------|
| **TranAD** | **0.9198** +/- 0.062 | **0.3845** +/- 0.195 | **0.8683** +/- 0.112 | **0.6521** +/- 0.178 | **0.3712** +/- 0.248 | **0.7234** +/- 0.218 | **86.9%** (284/327) | **8.5** ts | ~3.1 ms |
| Custom Transformer (T2V) | 0.9087 +/- 0.071 | 0.3628 +/- 0.183 | 0.8391 +/- 0.128 | 0.6284 +/- 0.185 | 0.3547 +/- 0.261 | 0.6892 +/- 0.247 | 85.0% (278/327) | 10.2 ts | ~2.4 ms |
| LSTM Autoencoder | 0.8912 +/- 0.085 | 0.3415 +/- 0.198 | 0.7986 +/- 0.152 | 0.5847 +/- 0.196 | 0.3289 +/- 0.231 | 0.6648 +/- 0.262 | 81.9% (268/327) | 8.9 ts | ~1.2 ms |
| Isolation Forest (MV) | 0.7903 +/- 0.098 | 0.2715 +/- 0.148 | 0.6724 +/- 0.195 | 0.3864 +/- 0.175 | 0.2856 +/- 0.168 | 0.4127 +/- 0.234 | 67.9% (222/327) | 17.8 ts | ~0.3 ms |

**Detection Latency** measures the number of timesteps from the onset of a ground-truth anomaly segment until the model first flags it. Undetected segments are penalized with full segment length (worst case). This metric is dominated by two factors: (1) per-detection speed — how quickly the anomaly score exceeds the threshold once anomalous data enters the window, and (2) detection rate — missed segments contribute large penalties. LSTM Autoencoder achieves the second-lowest detection latency despite its lower detection rate because its simple reconstruction error spikes almost immediately upon anomaly onset. **Inference Time** is the computational time per sample (window of 30 timesteps, 38 features) on GPU; Isolation Forest runs on CPU.

### 10.2 Interpretation of Results

1. **TranAD** achieves the highest average scores as an established benchmark model for time-series anomaly detection:
   - Highest AUC-ROC (0.92), PA-F1 (0.87), and detection rate (86.9%)
   - Lowest detection latency (8.5 timesteps) — driven primarily by the highest detection rate, which minimizes missed-segment penalties
   - The self-conditioning architecture with two decoders leverages reconstruction error iteratively for refinement

2. **Custom Transformer (T2V)** achieves results close to TranAD using prediction-error-based scoring:
   - AUC-ROC 0.91 and PA-F1 0.84 — comparable to TranAD's reconstruction-based approach
   - Time2Vec encoding + cross-attention decoder for next-step forecasting as an alternative anomaly mechanism
   - 85% segment detection demonstrates reliable detection of anomalous regions

3. **LSTM Autoencoder** as a recurrent baseline:
   - Solid AUC-ROC (0.89) and PA-F1 (0.80)
   - Fastest per-detection response due to its simpler architecture — reconstruction error spikes immediately upon anomaly onset, yielding detection latency (8.9 ts) close to TranAD despite 5% lower detection rate
   - Lowest inference time among deep learning models (1.2 ms vs. 2.4-3.1 ms for Transformers)

4. **Isolation Forest (MV)** as a point-based baseline:
   - Lowest AUC-ROC (0.79) and PA-F1 (0.67) — lack of temporal modeling limits performance
   - Fastest inference time (0.3 ms, CPU-based) but highest detection latency (17.8 ts): without temporal context, gradual anomaly onset phases are missed, while temporal models detect pattern changes early
   - 68% segment detection highlights the advantage of sequential models

5. **General Observations**:
   - The 99th percentile threshold on training errors enables deployment without labeled anomalies (unsupervised)
   - Point-level F1 is consistently low (0.27-0.38), typical for anomaly detection with imbalanced data
   - PA-F1 is significantly higher (0.67-0.87), confirming the literature on segment-based evaluation
   - Best-F1 (oracle) shows the potential of the models (0.39-0.65) — threshold tuning is critical
   - 327 GT segments across 28 machines; TranAD detects 284 of them
   - Transformer-based models (TranAD, Custom Transformer) significantly outperform non-sequential approaches

### 10.3 Best Machine per Model (Highest AUC-ROC)

From [metrics_best_machine.csv](artifacts/multivariate/metrics_best_machine.csv):
- **machine-2-8** appears for multiple models — a particularly "readable" anomaly case with high AUC-ROC (>0.99 for all models)

---

## Appendix: CLI Commands

```bash
# Download and preprocess data
python -m time_series_transformer data

# Univariate training (baselines)
python -m time_series_transformer train --csv path/to/data.csv --mlflow

# Multivariate training on SMD
python -m time_series_transformer train-mv --machine machine-1-1
python -m time_series_transformer train-mv --machine all --save-checkpoints

# Benchmark
python -m time_series_transformer benchmark --config configs/benchmark_smd.yaml

# Aggregate results
python scripts/aggregate_results.py

# Start inference server
python -m time_series_transformer serve

# Start dashboard
python -m time_series_transformer dashboard

# MLflow UI
python -m time_series_transformer mlflow

# Docker (all services)
docker compose up
```
