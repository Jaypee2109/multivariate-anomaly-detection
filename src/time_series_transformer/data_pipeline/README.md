# PIPELINE DESCRIPTION

This document describes what currently happens in data pipeline
when you run `python build_preprocessed_data.py`.

0. PREPARATION

---

- Set up a Python environment
- Install dependencies from requirements.txt
  - kagglehub
  - pandas
  - numpy

1. START OF THE PIPELINE

---

Script: build_preprocessed_data.py

1.1 Ensure directories exist - `ensure_directories()` is called. - If they do not exist yet, the following folders are created: - `data/raw/` - `data/processed/`

1.2 Download / synchronize datasets - `download_all_datasets()` is called. - For each dataset in `KAGGLE_DATASETS`: - Name: "smd_onmiad" -> Slug: "mgusat/smd-onmiad" - Name: "nasa_smap_msl" -> Slug: "patrickfleith/nasa-anomaly-detection-dataset-smap-msl" - Name: "nab" -> Slug: "boltzmannbrain/nab"

    For each dataset:
      - `kagglehub.dataset_download(slug)` is executed.
        - On first run: downloads from Kaggle + extracts into the Kaggle cache.
        - On later runs: uses the cache (no need to download again).
      - The cache path is copied into a project-specific directory:
        - `data/raw/<dataset_name>/`
          - e.g. `data/raw/smd_onmiad/`
          - e.g. `data/raw/nasa_smap_msl/`
          - e.g. `data/raw/nab/`

2. LOADING THE DATA

---

Script: data_loading.py  
Function: `load_dataset(name)`

2.1 Determine root folder - For a dataset `name` the path is: - `root = data/raw/<name>/` - If the folder does not exist: - Error is raised: "Raw data for dataset '<name>' not found".

2.2 Walk through all subfolders and files recursively - `os.walk(root)` iterates through all directories and files.

2.3 Ignore macOS metadata - Directories named `__MACOSX` are completely skipped. - Files starting with `._` (e.g. `._occupancy_6005.csv`) are skipped. - Reason: these are not real CSV files, but macOS resource forks.

2.4 Detect CSV files - Only files with extension `.csv` (case-insensitive) are considered.

2.5 Read CSV files - For each valid CSV: - Full path: `fpath` - Path relative to the dataset root: `rel_path` - Example: `realKnownCause/realKnownCause/nyc_taxi.csv` - Read with: `pd.read_csv(fpath)` - If a `UnicodeDecodeError` occurs: - The error is logged. - The file is skipped.

2.6 Return format - A dictionary is created: - `{ "relative/path/to/file.csv": DataFrame, ... }` - Example: - For dataset "nab": - Key: `realTraffic/realTraffic/speed_6005.csv` - Value: pandas DataFrame with the corresponding data.

2.7 Load multiple datasets - `load_all_datasets(dataset_names)`: - Calls `load_dataset(name)` for each name. - Returns: - `{ dataset_name: { rel_path: df, ... }, ... }`

3. PREPROCESSING

---

Script: preprocessing.py

3.1 Configuration object - `PreprocessingConfig` with, among others: - `scale_numeric` (bool): whether to scale numeric columns. - `use_datetime_index` (bool): whether to use a time column as index. - `exclude_from_scaling` (tuple): columns that should not be scaled.

3.2 Preprocessing a single DataFrame  
 Function: `preprocess_dataframe(df, config)`

    3.2.1 Detect time column (if enabled)
        - If `use_datetime_index == True`:
          - The code tries to find a time column based on common names:
            - "timestamp", "time", "datetime", "date" (case-insensitive).
          - If one of these columns exists:
            - Convert to `datetime` with `pd.to_datetime(..., errors="coerce")`.
            - Drop rows with `NaT` in this column.
            - Sort by this time column.
            - Set the time column as index (`DatetimeIndex`).

    3.2.2 Scale numeric columns (z-score), if enabled
        - If `scale_numeric == True`:
          - All numeric columns are detected.
          - Columns listed in `exclude_from_scaling` are skipped.
          - For each column to be scaled:
            - Mean and standard deviation are computed.
            - If `std == 0` or `NaN`:
              - Column is set to 0 (constant).
            - Otherwise:
              - `(value - mean) / std` is computed.
          - Result: numeric columns have mean ~0 and std deviation ~1.

3.3 Preprocessing an entire dataset (multiple DataFrames)  
 Function: `preprocess_dataset_dict(dataset_name, data, config)`

    - For each `(rel_path, df)` in the input dict:
      - `preprocess_dataframe(df, config)` is called.
      - The result is written into a new dict.
    - Return:
      - `{ rel_path: preprocessed_df, ... }`

3.4 Configuration in the main script - In `build_preprocessed_data.py`: - For each dataset, a default configuration is currently used: - `scale_numeric = True` - `use_datetime_index = True` - `exclude_from_scaling = ()` (empty) - This configuration can later be adjusted per dataset
(e.g. to avoid scaling label columns).

4. SAVING THE PREPROCESSED DATA

---

Script: build_preprocessed_data.py  
Function: `save_processed_dataset(dataset_name, processed_dict)`

4.1 Determine target folder - `base = data/processed/<dataset_name>/` - Example: `data/processed/nab/`

4.2 Rebuild folder structure - For each `rel_path` of a DataFrame: - Example: `realKnownCause/realKnownCause/nyc_taxi.csv` - Output path: `data/processed/<dataset_name>/realKnownCause/realKnownCause/nyc_taxi.csv` - If the subfolder does not exist: - `out_path.parent.mkdir(parents=True, exist_ok=True)` is called.

4.3 Save DataFrames - If the index is a `DatetimeIndex`: - `df.to_csv(out_path, index=True)` - Timestamps then appear in the first column of the CSV. - Otherwise: - `df.to_csv(out_path, index=False)`

4.4 Result - For each original CSV in the raw dataset there is a preprocessed CSV in
the corresponding subfolder of `data/processed/<dataset_name>/`.

5. OVERALL FLOW WHEN RUNNING

---

Command: `python build_preprocessed_data.py`

Order of operations:

1. Create directories `data/raw` and `data/processed` if needed.
2. Download all datasets from Kaggle (or copy from cache) into:
   - `data/raw/smd_onmiad/`
   - `data/raw/nasa_smap_msl/`
   - `data/raw/nab/`
3. For each dataset:
   a) Load raw data from `data/raw/<dataset_name>/` as CSV files.
   - `__MACOSX` folders and `._` files are ignored.
     b) Apply preprocessing to all DataFrames:
   - Optionally: detect a time column and set it as index.
   - Optionally: scale numeric columns using z-score.
     c) Save the preprocessed DataFrames as CSV into
     `data/processed/<dataset_name>/...`.

# END OF PIPELINE DESCRIPTION
