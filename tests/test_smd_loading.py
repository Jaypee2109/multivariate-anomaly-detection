"""Tests for SMD data loading and preprocessing utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture()
def mock_smd_dir(tmp_path):
    """Create a minimal mock SMD directory structure (5 features)."""
    n_rows = 50
    n_features = 5  # smaller than real 38 for speed

    base = tmp_path / "ServerMachineDataset"
    for subdir in ("train", "test", "test_label", "interpretation_label"):
        (base / subdir).mkdir(parents=True)

    rng = np.random.default_rng(42)
    train_data = rng.random((n_rows, n_features))
    test_data = rng.random((n_rows, n_features))
    labels = np.zeros(n_rows, dtype=int)
    labels[10:15] = 1
    labels[30:35] = 1

    machine_id = "machine-1-1"
    np.savetxt(base / "train" / f"{machine_id}.txt", train_data, delimiter=",")
    np.savetxt(base / "test" / f"{machine_id}.txt", test_data, delimiter=",")
    np.savetxt(
        base / "test_label" / f"{machine_id}.txt", labels, delimiter=",", fmt="%d",
    )
    with open(base / "interpretation_label" / f"{machine_id}.txt", "w") as f:
        f.write("10-14:0,1,2\n")
        f.write("30-34:3,4\n")

    return base, machine_id, n_rows, n_features


@pytest.fixture()
def mock_smd_dir_38(tmp_path):
    """Create a mock SMD directory with 38 features (real column names)."""
    n_rows = 50
    n_features = 38

    base = tmp_path / "ServerMachineDataset"
    for subdir in ("train", "test", "test_label", "interpretation_label"):
        (base / subdir).mkdir(parents=True)

    rng = np.random.default_rng(42)
    train_data = rng.random((n_rows, n_features))
    test_data = rng.random((n_rows, n_features))
    labels = np.zeros(n_rows, dtype=int)
    labels[10:15] = 1

    machine_id = "machine-1-1"
    np.savetxt(base / "train" / f"{machine_id}.txt", train_data, delimiter=",")
    np.savetxt(base / "test" / f"{machine_id}.txt", test_data, delimiter=",")
    np.savetxt(
        base / "test_label" / f"{machine_id}.txt", labels, delimiter=",", fmt="%d",
    )
    with open(base / "interpretation_label" / f"{machine_id}.txt", "w") as f:
        f.write("10-14:0,1,2\n")

    return base, machine_id, n_rows, n_features


class TestListSMDMachines:
    def test_lists_machines(self, mock_smd_dir):
        from time_series_transformer.data_pipeline.smd_loading import list_smd_machines

        base, machine_id, _, _ = mock_smd_dir
        machines = list_smd_machines(base)
        assert machines == [machine_id]


class TestLoadSMDMachine:
    def test_load_normalised(self, mock_smd_dir):
        from time_series_transformer.data_pipeline.smd_loading import load_smd_machine

        base, machine_id, n_rows, n_features = mock_smd_dir
        data = load_smd_machine(machine_id, base_dir=base, normalize=True)

        assert data.machine_id == machine_id
        assert isinstance(data.train_df, pd.DataFrame)
        assert data.train_df.shape == (n_rows, n_features)
        assert data.test_df.shape == (n_rows, n_features)
        assert len(data.test_labels) == n_rows
        assert data.test_labels.dtype == bool
        assert data.test_labels.sum() == 10  # 5 + 5 labelled anomalies

        # MinMaxScaler: train values should be in [0, 1]
        assert data.train_df.min().min() >= -1e-9
        assert data.train_df.max().max() <= 1.0 + 1e-9

    def test_load_unnormalised(self, mock_smd_dir):
        from time_series_transformer.data_pipeline.smd_loading import load_smd_machine

        base, machine_id, _, _ = mock_smd_dir
        data = load_smd_machine(machine_id, base_dir=base, normalize=False)
        assert data.train_df is not None

    def test_missing_machine_raises(self, mock_smd_dir):
        from time_series_transformer.data_pipeline.smd_loading import load_smd_machine
        from time_series_transformer.exceptions import DataNotFoundError

        base, _, _, _ = mock_smd_dir
        with pytest.raises(DataNotFoundError):
            load_smd_machine("machine-99-99", base_dir=base)

    def test_generic_columns_for_non38(self, mock_smd_dir):
        """With 5 features, columns should be f0..f4 (generic fallback)."""
        from time_series_transformer.data_pipeline.smd_loading import load_smd_machine

        base, machine_id, _, _ = mock_smd_dir
        data = load_smd_machine(machine_id, base_dir=base, normalize=False)
        assert list(data.train_df.columns) == [f"f{i}" for i in range(5)]

    def test_real_columns_for_38_features(self, mock_smd_dir_38):
        """With 38 features, columns should be real SMD metric names."""
        from time_series_transformer.data_pipeline.smd_loading import (
            SMD_COLUMN_NAMES,
            load_smd_machine,
        )

        base, machine_id, _, _ = mock_smd_dir_38
        data = load_smd_machine(machine_id, base_dir=base, normalize=False)
        assert list(data.train_df.columns) == SMD_COLUMN_NAMES
        assert list(data.test_df.columns) == SMD_COLUMN_NAMES


class TestPreprocessSMD:
    def test_preprocess_creates_csvs(self, mock_smd_dir):
        from time_series_transformer.data_pipeline.smd_loading import preprocess_smd

        base, machine_id, n_rows, _ = mock_smd_dir
        processed = base.parent / "processed_smd"

        machines = preprocess_smd(raw_dir=base, processed_dir=processed)
        assert machine_id in machines

        # CSVs should exist
        assert (processed / "train" / f"{machine_id}.csv").exists()
        assert (processed / "test" / f"{machine_id}.csv").exists()
        assert (processed / "test_label" / f"{machine_id}.csv").exists()

        # Check CSV has headers and correct shape
        train_df = pd.read_csv(processed / "train" / f"{machine_id}.csv")
        assert train_df.shape == (n_rows, 5)
        assert train_df.columns[0] == "f0"  # generic for 5-feature mock

        label_df = pd.read_csv(processed / "test_label" / f"{machine_id}.csv")
        assert "is_anomaly" in label_df.columns
        assert label_df["is_anomaly"].sum() == 10

    def test_preprocess_38_uses_real_names(self, mock_smd_dir_38):
        from time_series_transformer.data_pipeline.smd_loading import (
            SMD_COLUMN_NAMES,
            preprocess_smd,
        )

        base, machine_id, _, _ = mock_smd_dir_38
        processed = base.parent / "processed_smd"

        preprocess_smd(raw_dir=base, processed_dir=processed)

        train_df = pd.read_csv(processed / "train" / f"{machine_id}.csv")
        assert list(train_df.columns) == SMD_COLUMN_NAMES

    def test_load_from_processed_csvs(self, mock_smd_dir):
        """After preprocessing, load_smd_machine should read from CSVs."""
        from time_series_transformer.data_pipeline.smd_loading import (
            load_smd_machine,
            preprocess_smd,
        )

        base, machine_id, n_rows, n_features = mock_smd_dir
        processed = base.parent / "processed_smd"
        preprocess_smd(raw_dir=base, processed_dir=processed)

        data = load_smd_machine(machine_id, base_dir=processed, normalize=False)
        assert data.train_df.shape == (n_rows, n_features)
        assert data.test_df.shape == (n_rows, n_features)
        assert data.test_labels.sum() == 10

    def test_list_machines_from_processed(self, mock_smd_dir):
        from time_series_transformer.data_pipeline.smd_loading import (
            list_smd_machines,
            preprocess_smd,
        )

        base, machine_id, _, _ = mock_smd_dir
        processed = base.parent / "processed_smd"
        preprocess_smd(raw_dir=base, processed_dir=processed)

        machines = list_smd_machines(processed)
        assert machines == [machine_id]

    def test_interpretation_labels_copied(self, mock_smd_dir):
        from time_series_transformer.data_pipeline.smd_loading import (
            load_smd_interpretation_labels,
            preprocess_smd,
        )

        base, machine_id, _, _ = mock_smd_dir
        processed = base.parent / "processed_smd"
        preprocess_smd(raw_dir=base, processed_dir=processed)

        labels = load_smd_interpretation_labels(machine_id, base_dir=processed)
        assert len(labels) == 2
        assert labels[0] == (10, 14, [0, 1, 2])


class TestLoadInterpretationLabels:
    def test_parse_format(self, mock_smd_dir):
        from time_series_transformer.data_pipeline.smd_loading import (
            load_smd_interpretation_labels,
        )

        base, machine_id, _, _ = mock_smd_dir
        labels = load_smd_interpretation_labels(machine_id, base_dir=base)
        assert len(labels) == 2
        assert labels[0] == (10, 14, [0, 1, 2])
        assert labels[1] == (30, 34, [3, 4])
