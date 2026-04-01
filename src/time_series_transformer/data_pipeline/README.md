# Data Pipeline

This document describes the data pipeline executed by `python -m time_series_transformer data`.

## Overview

```
Download (kagglehub) → Load (CSV walk) → Preprocess (datetime + z-score) → Save
```

## 1. Download

`data_download.py` downloads datasets from Kaggle via `kagglehub`:

| Name            | Kaggle slug                                             |
| --------------- | ------------------------------------------------------- |
| `nab`           | `boltzmannbrain/nab`                                    |
| `smd`    | `mgusat/smd-onmiad`                                    |
| `nasa_smap_msl` | `patrickfleith/nasa-anomaly-detection-dataset-smap-msl` |

Data is downloaded to `data/raw/<dataset_name>/`. On subsequent runs, the Kaggle cache is used.

## 2. Loading

`data_loading.py` walks `data/raw/<dataset_name>/` recursively:

- Skips `__MACOSX` directories and `._` files (macOS resource forks)
- Reads all `.csv` files with `pd.read_csv()`
- Returns `{ "relative/path/to/file.csv": DataFrame, ... }`

## 3. Preprocessing

`preprocessing.py` applies transformations via `PreprocessingConfig`:

- **Datetime index** (`use_datetime_index=True`): Detects time columns ("timestamp", "time", "datetime", "date"), converts to `DatetimeIndex`, sorts chronologically
- **Z-score scaling** (`scale_numeric=True`): Standardises numeric columns to mean≈0, std≈1. Constant columns (std=0) are set to 0

**Note:** SMD data is pre-normalised to [0, 1] and uses `normalize=False` — see decision D14.

## 4. Saving

`data_save.py` writes preprocessed DataFrames to `data/processed/<dataset_name>/`, preserving the original subfolder structure.

## SMD Loading (Multivariate)

`smd_loading.py` provides `load_smd_machine(machine_id)` for the multivariate pipeline:

- Loads train/test splits and labels from `data/raw/smd/ServerMachineDataset/`
- Returns `MachineData(train_df, test_df, test_labels)`
- No normalisation applied (data is already in [0, 1])
