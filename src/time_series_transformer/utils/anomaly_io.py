from __future__ import annotations

from pathlib import Path

import pandas as pd


def save_anomaly_artifacts(
    y_test: pd.Series,
    scores_dict: dict[str, pd.Series],
    anomalies_dict: dict[str, pd.Series],
    out_path: str | Path,
) -> None:
    """
    Save anomaly scores and flags for each detector into a single wide CSV.

    Columns:
        timestamp (index)
        value
        <det>_score
        <det>_is_anomaly
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame({"value": y_test})
    for det_name, scores in scores_dict.items():
        df[f"{det_name}_score"] = scores
    for det_name, flags in anomalies_dict.items():
        df[f"{det_name}_is_anomaly"] = flags.astype(int)

    df.to_csv(out_path, index_label="timestamp")
    print(f"[Baseline] Saved anomaly artifacts to: {out_path}")


def load_anomaly_flags_from_artifacts(
    artifacts_path: str | Path,
    display_name_map: dict[str, str] | None = None,
) -> dict[str, pd.Series]:
    """
    Load anomaly flags from the artifacts CSV and return a dict:
        { display_name: pd.Series[bool] }

    It looks for all columns ending with '_is_anomaly' and uses the
    prefix as the detector key.

    Example columns:
        'rolling_is_anomaly'
        'Rolling Z-Score_is_anomaly'
        'arima_is_anomaly'

    You can optionally pass display_name_map to map internal keys
    to prettier labels:

        display_name_map = {
            "rolling": "Rolling Z-Score",
            "Rolling Z-Score": "Rolling Z-Score",
            "arima": "ARIMA Residual",
            ...
        }
    """
    artifacts_path = Path(artifacts_path)
    df = pd.read_csv(artifacts_path, parse_dates=["timestamp"])
    df = df.set_index("timestamp")

    anomalies_dict: dict[str, pd.Series] = {}

    # find all *_is_anomaly columns
    suffix = "_is_anomaly"
    anomaly_cols = [c for c in df.columns if c.endswith(suffix)]

    if not anomaly_cols:
        print(f"[Anomaly IO] No '*{suffix}' columns found in {artifacts_path}")
        return anomalies_dict

    for col in anomaly_cols:
        key = col[: -len(suffix)]  # strip '_is_anomaly'

        # choose display name
        if display_name_map is not None and key in display_name_map:
            display_name = display_name_map[key]
        else:
            # fallback: use key as-is
            display_name = key

        s = df[col].astype(bool)
        s.index = df.index

        anomalies_dict[display_name] = s

    return anomalies_dict
