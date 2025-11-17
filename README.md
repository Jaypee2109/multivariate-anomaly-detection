# Transformer

This repository contains a Python project for working with time series data using transformer-based models.

## Requirements

- Python 3.11 (or compatible 3.x)
- `pip`

Python dependencies are defined in:

- `requirements.txt`
- referenced from `pyproject.toml` for the editable install.

---

## Installation

### 1. Clone the repository

    git clone <REPO-URL>
    cd Transformer

Replace `<REPO-URL`> with the HTTPS or SSH URL of this repository.

---

### 2. Create and activate a virtual environment

**On macOS / Linux**

    python3 -m venv .venv
    source .venv/bin/activate

**On Windows (PowerShell)**

    python -m venv .venv
    .\.venv\Scripts\Activate.ps1

You should now see `(.venv)` at the beginning of your terminal prompt.

---

### 3. Install the package and dependencies

**Recommended: editable install via `pyproject.toml`**

    python -m pip install -e .

This:

- installs the `time_series_transformer` package in editable mode
- reads dependencies from `requirements.txt` via `pyproject.toml`

**Alternative: only from `requirements.txt`**

    python -m pip install -r requirements.txt

Use this if you only want the dependencies and don’t care about the editable package.

---

## Data folders

The project uses two main data folders:

- `data/raw/` – original datasets (as downloaded, unmodified)
- `data/processed/` – final, ready-to-use datasets for training/evaluation

Git ignores actual files in these folders (see `.gitignore`), but the folder structure is kept using `.gitkeep` placeholder files.

You are expected to:

- download or generate data into these folders locally
- **not** commit large data files to the repository

---

## Usage

### Running the preprocessing pipeline

A typical entry point is:

- `src/time_series_transformer/data_pipeline/build_preprocessed_data.py`

**From the command line** (project root, venv activated):

    python -m time_series_transformer.data_pipeline.build_preprocessed_data

Using `-m` ensures Python runs the module within the installed package and respects imports like:

    from time_series_transformer.config import ...
    from time_series_transformer.data_pipeline.data_download import download_all_datasets

**In VS Code**

1. Open the `Transformer` folder in VS Code.
2. Select the interpreter from the `.venv` virtual environment  
   (`Python: Select Interpreter` → choose `.venv`).
3. Open i.e. `build_preprocessed_data.py`.
4. Use the Run/Debug button.

As long as the `.venv` interpreter is selected and `python -m pip install -e .` has been run in that environment, imports such as `from time_series_transformer...` will work.

---

## Adding new dependencies

1.  Add the package to `requirements.txt`, e.g.:

    numpy
    pandas
    torch
    scikit-learn
    kagglehub

2.  Install it in your current virtual environment:

        python -m pip install -r requirements.txt

    or, if using the editable install:

        python -m pip install -e .

3.  Commit the updated `requirements.txt`, so collaborators can install the same dependencies.

---

## Collaborator setup (macOS)

On macOS:

    git clone <REPO-URL>
    cd Transformer

    python3 -m venv .venv
    source .venv/bin/activate

    python3 -m pip install -e .
    # or:
    # python3 -m pip install -r requirements.txt

Then in VS Code:

1. Open the `Transformer` folder.
2. Run `Python: Select Interpreter` and choose `.venv/bin/python`.
3. Run scripts via the Run button or from the terminal using `python -m ...`.

---

## License

MIT
