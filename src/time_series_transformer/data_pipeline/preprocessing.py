# preprocessing.py
from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# ---------- Helper functions ----------


def _guess_time_column(columns: Iterable[str]) -> str | None:

    candidates = ["timestamp", "time", "datetime", "date"]
    lower_map = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand in lower_map:
            return lower_map[cand]
    return None


def to_datetime_index(df: pd.DataFrame) -> pd.DataFrame:

    time_col = _guess_time_column(df.columns)
    if time_col is None:
        return df

    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col]).sort_values(time_col)
    df = df.set_index(time_col)
    return df


def standard_scale(df: pd.DataFrame, exclude: Iterable[str] | None = None) -> pd.DataFrame:

    if exclude is None:
        exclude = []

    df = df.copy()
    numeric_cols = df.select_dtypes(include="number").columns
    cols_to_scale = [c for c in numeric_cols if c not in exclude]

    for col in cols_to_scale:
        mean = df[col].mean()
        std = df[col].std()
        logger.debug("col=%s mean=%.4f std=%.4f", col, mean, std)
        if std == 0 or pd.isna(std):
            # Konstante Spalte -> einfach auf 0 setzen
            df[col] = 0.0
        else:
            df[col] = (df[col] - mean) / std

    return df


def load_csv_to_df(
    path: str | Path,
    parse_dates: Iterable[str] | None = None,
    index_col: str | None = None,
    **read_csv_kwargs,
) -> pd.DataFrame:

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if path.suffix.lower() != ".csv":
        raise ValueError(f"Expected a .csv file, got: {path.suffix} ({path})")

    df = pd.read_csv(path, parse_dates=parse_dates, **read_csv_kwargs)

    if index_col is not None:
        if index_col not in df.columns:
            raise KeyError(
                f"Index column '{index_col}' not found in columns: {df.columns.tolist()}"
            )
        df = df.set_index(index_col)

    return df


# ---------- Pipeline ----------


@dataclass
class PreprocessingConfig:
    scale_numeric: bool = True
    use_datetime_index: bool = True
    # Spaltennamen, die nicht skaliert werden sollen (Labels etc.)
    exclude_from_scaling: tuple[str, ...] = ()


def preprocess_dataframe(df: pd.DataFrame, config: PreprocessingConfig) -> pd.DataFrame:

    if config.use_datetime_index:
        df = to_datetime_index(df)

    if config.scale_numeric:
        df = standard_scale(df, exclude=config.exclude_from_scaling)

    return df


def preprocess_dataset_dict(
    dataset_name: str,
    data: dict[str, pd.DataFrame],
    config: PreprocessingConfig | None = None,
) -> dict[str, pd.DataFrame]:

    if config is None:
        # Default-Konfig – kannst du pro Dataset anpassen
        config = PreprocessingConfig()

    logger.info("Preprocessing dataset '%s' ...", dataset_name)
    out: dict[str, pd.DataFrame] = {}

    for rel_path, df in data.items():
        logger.info("  -> %s", rel_path)
        out[rel_path] = preprocess_dataframe(df, config)

    return out


if __name__ == "__main__":
    # Test

    df_test = pd.DataFrame(
        {
            "timestamp": pd.date_range("2020-01-01", periods=5, freq="H"),
            "value": [1, 2, 3, 4, 5],
        }
    )
    cfg = PreprocessingConfig()
    print(preprocess_dataframe(df_test, cfg))
