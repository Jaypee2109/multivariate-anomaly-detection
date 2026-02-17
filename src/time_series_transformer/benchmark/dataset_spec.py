"""Dataset specification for benchmarking."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class DatasetSpec:
    """A single dataset to benchmark against.

    Parameters
    ----------
    name : str
        Human-readable identifier (e.g. ``"NAB_nyc_taxi"``).
    csv_path : Path
        Path to the time-series CSV (must have ``timestamp`` and ``value`` columns).
    labels_path : Path | None
        Optional path to a NAB-style labels JSON.
    labels_key : str | None
        Key inside the JSON to extract label timestamps.
    """

    name: str
    csv_path: Path
    labels_path: Path | None = None
    labels_key: str | None = None

    def has_labels(self) -> bool:
        """Return *True* if ground-truth labels are available."""
        return (
            self.labels_path is not None
            and self.labels_key is not None
            and self.labels_path.exists()
        )


@dataclass
class MultivariateDatasetSpec:
    """An SMD-style multivariate dataset with pre-split train/test.

    Unlike :class:`DatasetSpec` (single CSV split by ratio), this points to
    a directory with separate ``train/``, ``test/``, and ``test_label/``
    sub-directories.
    """

    name: str
    machine_id: str
    base_dir: Path
    normalize: bool = True

    def has_labels(self) -> bool:
        """SMD always ships with test labels."""
        label_path = self.base_dir / "test_label" / f"{self.machine_id}.txt"
        return label_path.exists()
