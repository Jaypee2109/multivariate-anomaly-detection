from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def load_label_times(
    json_path: str | Path,
    dataset_key: str = "realKnownCause/nyc_taxi.csv",
) -> list[pd.Timestamp]:
    """
    Load NAB label timestamps for a given dataset from the labels JSON.

    Parameters
    ----------
    json_path : path-like
        Path to the JSON file containing the NAB labels.
    dataset_key : str
        Key for the specific dataset (e.g. "realKnownCause/nyc_taxi.csv").

    Returns
    -------
    List[pd.Timestamp]
        List of label timestamps as pandas Timestamps.
    """
    json_path = Path(json_path)
    with json_path.open("r") as f:
        label_dict = json.load(f)

    if dataset_key not in label_dict:
        raise KeyError(f"Dataset key '{dataset_key}' not found in {json_path}")

    ts_strs = label_dict[dataset_key]
    label_times = pd.to_datetime(ts_strs)
    return list(label_times)


def make_point_labels_from_times(
    df: pd.DataFrame,
    label_times: list[pd.Timestamp],
    timestamp_col: str = "timestamp",
) -> pd.Series:
    """
    Create point-wise anomaly labels from a list of label timestamps.

    For each row in df, label is True if df[timestamp_col] matches
    (exactly) one of the timestamps in `label_times`.

    The returned Series:
      - has dtype bool
      - has a DatetimeIndex equal to the timestamp column values

    This is convenient for range-based metrics that expect a DatetimeIndex.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing a timestamp column.
    label_times : list of pd.Timestamp
        List of anomaly timestamps from NAB.
    timestamp_col : str
        Name of the timestamp column in df.

    Returns
    -------
    pd.Series
        Boolean Series of labels indexed by timestamp (DatetimeIndex).
    """
    ts = pd.to_datetime(df[timestamp_col])
    label_times_index = pd.DatetimeIndex(label_times)

    # Boolean array: is this timestamp one of the labeled anomaly times?
    labels = ts.isin(label_times_index)

    # Make the index a DatetimeIndex of the timestamps themselves
    labels.index = pd.DatetimeIndex(ts)

    # Ensure boolean dtype
    labels = labels.astype(bool)

    return labels
