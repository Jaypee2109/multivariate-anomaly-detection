"""SMD (Server Machine Dataset) loading and preprocessing utilities."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from time_series_transformer.config import SMD_BASE_DIR, SMD_RAW_DIR
from time_series_transformer.exceptions import DataNotFoundError

logger = logging.getLogger(__name__)

N_FEATURES = 38

# Real SMD column names (38 server-monitoring metrics).
# Source: OmniAnomaly / InterFusion papers; order matches the original .txt files.
SMD_COLUMN_NAMES: list[str] = [
    "cpu_r",
    "load_1",
    "load_5",
    "load_15",
    "mem_shmem",
    "mem_u",
    "mem_u_e",
    "total_mem",
    "disk_q",
    "disk_r",
    "disk_rb",
    "disk_svc",
    "disk_u",
    "disk_w",
    "disk_wa",
    "disk_wb",
    "si",
    "so",
    "eth1_fi",
    "eth1_fo",
    "eth1_pi",
    "eth1_po",
    "tcp_tw",
    "tcp_use",
    "active_opens",
    "curr_estab",
    "in_errs",
    "in_segs",
    "listen_overflows",
    "out_rsts",
    "out_segs",
    "passive_opens",
    "retransegs",
    "tcp_timeouts",
    "udp_in_dg",
    "udp_out_dg",
    "udp_rcv_buf_errs",
    "udp_snd_buf_errs",
]


@dataclass
class SMDMachineData:
    """Container for one SMD machine's train / test / labels."""

    machine_id: str
    train_df: pd.DataFrame  # (n_train, n_features)
    test_df: pd.DataFrame  # (n_test, n_features)
    test_labels: pd.Series  # (n_test,) boolean
    scaler: MinMaxScaler


def _feature_columns(n_features: int) -> list[str]:
    """Return real column names if n_features matches, otherwise generic names."""
    if n_features == len(SMD_COLUMN_NAMES):
        return list(SMD_COLUMN_NAMES)
    return [f"f{i}" for i in range(n_features)]


# ---------------------------------------------------------------------------
# Preprocessing: raw .txt → processed .csv
# ---------------------------------------------------------------------------


def preprocess_smd(
    raw_dir: Path = SMD_RAW_DIR,
    processed_dir: Path = SMD_BASE_DIR,
) -> list[str]:
    """Convert raw SMD .txt files to CSVs with proper column headers.

    Reads from ``raw_dir/{train,test,test_label}/*.txt`` and writes to
    ``processed_dir/{train,test,test_label}/*.csv``.

    Returns the list of preprocessed machine IDs.
    """
    train_dir = raw_dir / "train"
    if not train_dir.exists():
        raise DataNotFoundError(f"SMD raw train directory not found: {train_dir}")

    machines = sorted(p.stem for p in train_dir.glob("*.txt"))
    if not machines:
        raise DataNotFoundError(f"No .txt files found in {train_dir}")

    for subdir in ("train", "test", "test_label"):
        (processed_dir / subdir).mkdir(parents=True, exist_ok=True)

    for machine_id in machines:
        for subdir in ("train", "test", "test_label"):
            src = raw_dir / subdir / f"{machine_id}.txt"
            dst = processed_dir / subdir / f"{machine_id}.csv"
            if not src.exists():
                logger.warning("Missing %s/%s.txt — skipping", subdir, machine_id)
                continue

            arr = np.loadtxt(src, delimiter=",", dtype=np.float64)
            if subdir == "test_label":
                # Labels are a 1-D array (one column)
                pd.DataFrame({"is_anomaly": arr.astype(int)}).to_csv(
                    dst, index=False,
                )
            else:
                n_feat = arr.shape[1] if arr.ndim == 2 else 1
                cols = _feature_columns(n_feat)
                pd.DataFrame(arr, columns=cols).to_csv(dst, index=False)

        logger.debug("Preprocessed %s", machine_id)

    # Copy interpretation labels as-is (plain text, not CSV)
    interp_src = raw_dir / "interpretation_label"
    if interp_src.exists():
        interp_dst = processed_dir / "interpretation_label"
        interp_dst.mkdir(parents=True, exist_ok=True)
        for src_file in interp_src.glob("*.txt"):
            dst_file = interp_dst / src_file.name
            dst_file.write_text(src_file.read_text())

    logger.info(
        "Preprocessed %d SMD machines: %s → %s",
        len(machines), raw_dir, processed_dir,
    )
    return machines


# ---------------------------------------------------------------------------
# Machine listing
# ---------------------------------------------------------------------------


def list_smd_machines(base_dir: Path = SMD_BASE_DIR) -> list[str]:
    """Return sorted list of machine IDs available in the dataset.

    Checks the processed directory (.csv) first, falls back to raw (.txt).
    """
    train_dir = base_dir / "train"
    if train_dir.exists():
        csvs = sorted(p.stem for p in train_dir.glob("*.csv"))
        if csvs:
            return csvs
        txts = sorted(p.stem for p in train_dir.glob("*.txt"))
        if txts:
            return txts

    # Fallback: try raw directory
    raw_train = SMD_RAW_DIR / "train"
    if raw_train.exists():
        return sorted(p.stem for p in raw_train.glob("*.txt"))

    raise DataNotFoundError(f"SMD train directory not found: {train_dir}")


# ---------------------------------------------------------------------------
# Machine loading
# ---------------------------------------------------------------------------


def load_smd_machine(
    machine_id: str,
    base_dir: Path = SMD_BASE_DIR,
    normalize: bool = True,
) -> SMDMachineData:
    """Load train, test, and test_label for a single SMD machine.

    Checks for preprocessed CSVs (with headers) first, falls back to raw
    .txt files. Uses real column names for the standard 38-feature layout.

    Parameters
    ----------
    machine_id : str
        E.g. ``"machine-1-1"``.
    base_dir : Path
        Root directory (processed or raw ``ServerMachineDataset``).
    normalize : bool
        If *True*, apply MinMaxScaler (fit on train, transform both).
    """
    train_df, test_df, label_arr = _load_splits(machine_id, base_dir)

    scaler = MinMaxScaler()
    if normalize:
        train_arr = scaler.fit_transform(train_df.values)
        test_arr = scaler.transform(test_df.values)
        train_df = pd.DataFrame(train_arr, columns=train_df.columns)
        test_df = pd.DataFrame(test_arr, columns=test_df.columns)
    else:
        scaler.fit(train_df.values)

    test_labels = pd.Series(label_arr.astype(bool), name="is_anomaly")

    logger.info(
        "Loaded SMD %s: train=%d, test=%d, anomaly_rate=%.2f%%",
        machine_id,
        len(train_df),
        len(test_df),
        test_labels.mean() * 100,
    )

    return SMDMachineData(
        machine_id=machine_id,
        train_df=train_df,
        test_df=test_df,
        test_labels=test_labels,
        scaler=scaler,
    )


def _load_splits(
    machine_id: str, base_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Load train/test/labels, trying processed CSV then raw TXT."""
    # Try processed CSVs first
    csv_train = base_dir / "train" / f"{machine_id}.csv"
    csv_test = base_dir / "test" / f"{machine_id}.csv"
    csv_label = base_dir / "test_label" / f"{machine_id}.csv"

    if csv_train.exists() and csv_test.exists() and csv_label.exists():
        train_df = pd.read_csv(csv_train)
        test_df = pd.read_csv(csv_test)
        label_df = pd.read_csv(csv_label)
        label_arr = label_df["is_anomaly"].values.astype(np.float64)
        return train_df, test_df, label_arr

    # Fall back to raw TXT files (headerless)
    txt_train = base_dir / "train" / f"{machine_id}.txt"
    txt_test = base_dir / "test" / f"{machine_id}.txt"
    txt_label = base_dir / "test_label" / f"{machine_id}.txt"

    # Also try the raw dir if base_dir is the processed dir
    if not txt_train.exists():
        txt_train = SMD_RAW_DIR / "train" / f"{machine_id}.txt"
        txt_test = SMD_RAW_DIR / "test" / f"{machine_id}.txt"
        txt_label = SMD_RAW_DIR / "test_label" / f"{machine_id}.txt"

    for p, desc in [
        (txt_train, "train"),
        (txt_test, "test"),
        (txt_label, "test_label"),
    ]:
        if not p.exists():
            raise DataNotFoundError(f"SMD {desc} file not found: {p}")

    train_arr = np.loadtxt(txt_train, delimiter=",", dtype=np.float64)
    test_arr = np.loadtxt(txt_test, delimiter=",", dtype=np.float64)
    label_arr = np.loadtxt(txt_label, delimiter=",", dtype=np.float64)

    n_features = train_arr.shape[1] if train_arr.ndim == 2 else 1
    feature_cols = _feature_columns(n_features)

    train_df = pd.DataFrame(train_arr, columns=feature_cols)
    test_df = pd.DataFrame(test_arr, columns=feature_cols)
    return train_df, test_df, label_arr


# ---------------------------------------------------------------------------
# Interpretation labels
# ---------------------------------------------------------------------------


def load_smd_interpretation_labels(
    machine_id: str,
    base_dir: Path = SMD_BASE_DIR,
) -> list[tuple[int, int, list[int]]]:
    """Load interpretation labels: ``(start_row, end_row, [feature_indices])``.

    Format per line: ``startrow-endrow:feat1,feat2,...``
    """
    # Try processed dir, then raw dir
    path = base_dir / "interpretation_label" / f"{machine_id}.txt"
    if not path.exists():
        path = SMD_RAW_DIR / "interpretation_label" / f"{machine_id}.txt"
    if not path.exists():
        raise DataNotFoundError(f"Interpretation label not found: {path}")

    result: list[tuple[int, int, list[int]]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            range_part, feat_part = line.split(":")
            start_s, end_s = range_part.split("-")
            features = [int(x) for x in feat_part.split(",")]
            result.append((int(start_s), int(end_s), features))
    return result
