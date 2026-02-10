"""Startup validation utilities.

Each check returns ``(ok: bool, message: str)`` so callers can decide
whether to abort or just warn.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def check_data_file(path: Path, description: str = "data file") -> tuple[bool, str]:
    """Check whether *path* exists and give a helpful hint if not."""
    if path.exists():
        return True, f"{description} found: {path}"
    return (
        False,
        f"{description} not found: {path}\n"
        "  Run 'python -m time_series_transformer data' to download datasets.",
    )


def check_cuda_available() -> tuple[bool, str]:
    """Report GPU availability."""
    try:
        import torch

        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            return True, f"CUDA available: {name}"
        return False, "CUDA not available — training will use CPU."
    except ImportError:
        return False, "PyTorch not installed."


def check_mlflow_setup(project_root: Path) -> tuple[bool, str]:
    """Verify that MLflow tracking can write to *project_root*."""
    try:
        test_file = project_root / ".write_test"
        test_file.touch()
        test_file.unlink()
    except (PermissionError, OSError) as exc:
        return False, f"Cannot write to project directory {project_root}: {exc}"

    mlruns = project_root / "mlruns"
    if not mlruns.exists():
        try:
            mlruns.mkdir(parents=True, exist_ok=True)
        except (PermissionError, OSError) as exc:
            return False, f"Cannot create mlruns directory: {exc}"

    return True, "MLflow setup OK"


def check_kaggle_credentials() -> tuple[bool, str]:
    """Check whether Kaggle credentials are available."""
    # env-var approach
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True, "Kaggle credentials found (env vars)."

    # file approach
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if kaggle_json.exists():
        return True, f"Kaggle credentials found: {kaggle_json}"

    return (
        False,
        "Kaggle credentials not found.\n"
        "  Set KAGGLE_USERNAME + KAGGLE_KEY env vars, or place kaggle.json in ~/.kaggle/",
    )


def run_checks_for_command(
    command: str,
    *,
    csv_path: Path | None = None,
    project_root: Path | None = None,
    use_mlflow: bool = False,
) -> list[tuple[bool, str]]:
    """Run the relevant pre-flight checks for *command* and return results."""
    results: list[tuple[bool, str]] = []

    if command == "data":
        results.append(check_kaggle_credentials())

    if command == "train":
        if csv_path is not None:
            results.append(check_data_file(csv_path, "Training CSV"))
        results.append(check_cuda_available())

    if use_mlflow and project_root is not None:
        results.append(check_mlflow_setup(project_root))

    return results


def log_check_results(results: list[tuple[bool, str]]) -> bool:
    """Log check results and return True if all passed."""
    all_ok = True
    for ok, msg in results:
        if ok:
            logger.debug("  [OK] %s", msg)
        else:
            logger.warning("  [!!] %s", msg)
            all_ok = False
    return all_ok
