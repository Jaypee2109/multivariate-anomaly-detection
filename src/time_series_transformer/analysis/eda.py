# src/time_series_transformer/analysis/eda_timeseries.py

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Tuple
from time_series_transformer.data_pipeline.preprocessing import load_csv_to_df

import pandas as pd
import holoviews as hv
from holoviews import opts

# Für VS Code mit Jupyter / Interactive Window:
# - Wenn du dieses Skript als Notebook/interactive ausführst, werden Plots angezeigt.
hv.extension("bokeh")


def basic_overview(df: pd.DataFrame) -> None:

    print("=== HEAD ===")
    print(df.head())

    print("\n=== DESCRIBE ===")
    print(df.describe(include="all"))

    print("\n=== MISSING VALUES PER COLUMN ===")
    print(df.isnull().sum())


def time_range_info(df: pd.DataFrame, timestamp_col: str = "timestamp") -> None:
    """
    Print start time, end time and time difference for a timestamp column.
    """
    if timestamp_col not in df.columns:
        raise KeyError(
            f"Timestamp column '{timestamp_col}' not found in columns: {df.columns.tolist()}"
        )

    start = df[timestamp_col].min()
    end = df[timestamp_col].max()
    diff = end - start

    print("\n=== TIME RANGE ===")
    print("Start time: ", start)
    print("End time:   ", end)
    print("Time diff:  ", diff)


def _infer_value_column(df: pd.DataFrame, value_col: Optional[str]) -> str:
    """
    If value_col is None, pick the first numeric column as value column.
    """
    if value_col is not None:
        if value_col not in df.columns:
            raise KeyError(
                f"Value column '{value_col}' not found in columns: {df.columns.tolist()}"
            )
        return value_col

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if not numeric_cols:
        raise ValueError(
            "No numeric columns found in DataFrame. Please specify `value_col` explicitly."
        )

    inferred = numeric_cols[0]
    print(f"[EDA] No value_col specified, using first numeric column: '{inferred}'")
    return inferred


def make_resampled_curves(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    value_col: Optional[str] = None,
    freqs: Iterable[str] = ("h", "d", "W"),
) -> hv.Layout:

    if timestamp_col not in df.columns:
        raise KeyError(
            f"Timestamp column '{timestamp_col}' not found in columns: {df.columns.tolist()}"
        )

    value_col = _infer_value_column(df, value_col)

    ts_df = df.set_index(timestamp_col)

    freq_names = {
        "h": "Hourly",
        "d": "Daily",
        "W": "Weekly",
    }

    curves = []
    for freq in freqs:
        series = ts_df[value_col].resample(freq).mean()

        title_suffix = freq_names.get(freq, freq)
        curve = hv.Curve(series).opts(
            opts.Curve(
                title=f"{value_col} ({title_suffix})",
                xlabel="Date",
                ylabel=value_col,
                width=700,
                height=300,
                tools=["hover"],
                show_grid=True,
            )
        )
        curves.append(curve)

    if not curves:
        raise ValueError("No frequencies provided to `make_resampled_curves`.")

    layout = hv.Layout(curves).opts(shared_axes=False).cols(1)
    return layout


def run_eda_pipeline(
    csv_path: str | Path,
    timestamp_col: str = "timestamp",
    value_col: Optional[str] = None,
    freqs: Iterable[str] = ("h", "d", "W"),
    save_html: bool = True,
) -> None:

    df = load_csv_to_df(csv_path, parse_dates=[timestamp_col])

    basic_overview(df)
    time_range_info(df, timestamp_col=timestamp_col)

    layout = make_resampled_curves(
        df,
        timestamp_col=timestamp_col,
        value_col=value_col,
        freqs=freqs,
    )

    if save_html:
        csv_path = Path(csv_path)
        save_path = Path("reports") / f"{csv_path.stem}.html"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        hv.save(layout, save_path, fmt="html")
        print(f"\n[EDA] Saved HTML visualization to: {save_path}")

    return df, layout
