from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Tuple, Dict

import pandas as pd
import holoviews as hv
from holoviews import opts

from time_series_transformer.data_pipeline.preprocessing import load_csv_to_df
from time_series_transformer.utils.anomaly_io import load_anomaly_flags_from_artifacts

hv.extension("bokeh")


# ----------------------------------------------------------------------
# Basic utilities
# ----------------------------------------------------------------------


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


# ----------------------------------------------------------------------
# Resampled curves (basic EDA)
# ----------------------------------------------------------------------


def make_resampled_curves(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    value_col: Optional[str] = None,
    freqs: Iterable[str] = ("h", "d", "W"),
) -> hv.Layout:
    """
    Create hourly/daily/weekly (etc.) resampled curves as a Layout.
    """
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
        curve = hv.Curve(series, label=title_suffix).opts(
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


# ----------------------------------------------------------------------
# High-level EDA entrypoints (two separate methods)
# ----------------------------------------------------------------------


def run_basic_eda_from_csv(
    csv_path: str | Path,
    timestamp_col: str = "timestamp",
    value_col: Optional[str] = None,
    freqs: Iterable[str] = ("h", "d", "W"),
    save_html: bool = True,
) -> Tuple[pd.DataFrame, hv.Layout]:
    """
    Basic EDA:
    - load CSV
    - print head/describe/missing
    - print time range
    - create resampled curves (hourly/daily/weekly)
    """
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
        save_path = Path("reports") / f"{csv_path.stem}_basic_eda.html"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        hv.save(layout, save_path, fmt="html")
        print(f"\n[EDA] Saved basic EDA HTML visualization to: {save_path}")

    return df, layout


def run_anomaly_eda_from_artifacts(
    artifacts_path: str | Path,
    timestamp_col: str = "timestamp",
    value_col: str = "value",
    save_html: bool = True,
    html_name: str = "timeseries_anomalies.html",
):
    """
    Simple anomaly EDA that uses only the artifacts CSV.

    Expects columns like:
        timestamp, value,
        <detector>_score, <detector>_is_anomaly

    Example:
        Rolling Z-Score_is_anomaly
        ARIMA Residual_is_anomaly
        Isolation Forest_is_anomaly
    """
    artifacts_path = Path(artifacts_path)

    # 1) Load artifacts
    df = pd.read_csv(artifacts_path, parse_dates=[timestamp_col])
    df = df.set_index(timestamp_col).sort_index()

    # 2) Base curve of the value
    base_curve = hv.Curve(df[value_col], label="Value").opts(
        opts.Curve(
            title=f"{value_col} with anomalies (test period)",
            xlabel="Time",
            ylabel=value_col,
            width=900,
            height=350,
            tools=["hover"],
            show_grid=True,
            muted_alpha=0.1,
        )
    )

    overlays = [base_curve]

    # 3) Find all *_is_anomaly columns
    suffix = "_is_anomaly"
    anomaly_cols = [c for c in df.columns if c.endswith(suffix)]

    detector_colors = [
        "red",
        "orange",
        "purple",
        "green",
        "blue",
    ]

    for i, col in enumerate(anomaly_cols):
        detector_name = col[: -len(suffix)]  # strip "_is_anomaly"
        mask = df[col].astype(bool)

        if not mask.any():
            continue  # skip detectors that found no anomalies

        anom_df = df.loc[mask, [value_col]]

        xs = anom_df.index
        ys = anom_df[value_col]

        color = detector_colors[i % len(detector_colors)]

        points = hv.Scatter(
            (xs, ys),
            kdims=[timestamp_col],
            vdims=[value_col],
            label=detector_name,
        ).opts(
            opts.Scatter(
                size=7,
                marker="circle",
                color=color,
                tools=["hover"],
                muted_alpha=0.05,
            )
        )

        overlays.append(points)

    overlay = hv.Overlay(overlays).opts(
        opts.Overlay(
            legend_position="top_left",
            legend_opts={"click_policy": "hide"},  # click legend to hide/show
        )
    )

    if save_html:
        save_path = Path("reports") / html_name
        save_path.parent.mkdir(parents=True, exist_ok=True)
        hv.save(overlay, save_path, fmt="html")
        print(f"[EDA] Saved anomaly EDA HTML visualization to: {save_path}")

    return overlay
