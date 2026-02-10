"""HTTP client for the Anomaly Detection inference API."""

from __future__ import annotations

import os
from typing import Any

import requests

DEFAULT_API_URL = "http://localhost:8000"


class AnomalyClient:
    """Thin wrapper around the inference API used by the dashboard."""

    def __init__(self, base_url: str | None = None, timeout: int = 30) -> None:
        self.base_url = (
            base_url or os.getenv("ANOMALY_API_URL", DEFAULT_API_URL)
        ).rstrip("/")
        self.timeout = timeout
        self._last_error: str | None = None

    @property
    def last_error(self) -> str | None:
        return self._last_error

    # ------------------------------------------------------------------
    # Health / info
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        try:
            resp = requests.get(f"{self.base_url}/health", timeout=5)
            if resp.status_code == 200:
                return resp.json().get("status") == "healthy"
            return False
        except requests.exceptions.RequestException as exc:
            self._last_error = str(exc)
            return False

    def get_health(self) -> dict[str, Any] | None:
        try:
            resp = requests.get(f"{self.base_url}/health", timeout=self.timeout)
            return resp.json() if resp.status_code == 200 else None
        except requests.exceptions.RequestException as exc:
            self._last_error = str(exc)
            return None

    def get_models(self) -> list[dict[str, Any]]:
        try:
            resp = requests.get(f"{self.base_url}/models", timeout=self.timeout)
            if resp.status_code == 200:
                return resp.json().get("models", [])
            return []
        except requests.exceptions.RequestException as exc:
            self._last_error = str(exc)
            return []

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def detect(
        self,
        timestamps: list[str],
        values: list[float],
        models: list[str] | None = None,
    ) -> dict[str, Any] | None:
        payload: dict[str, Any] = {
            "data": {"timestamps": timestamps, "values": values},
        }
        if models:
            payload["models"] = models
        try:
            resp = requests.post(
                f"{self.base_url}/detect",
                json=payload,
                timeout=self.timeout,
            )
            return resp.json() if resp.status_code == 200 else None
        except requests.exceptions.RequestException as exc:
            self._last_error = str(exc)
            return None

    def detect_dashboard(
        self,
        timestamps: list[str],
        values: list[float],
        models: list[str] | None = None,
    ) -> dict[str, Any] | None:
        payload: dict[str, Any] = {
            "data": {"timestamps": timestamps, "values": values},
        }
        if models:
            payload["models"] = models
        try:
            resp = requests.post(
                f"{self.base_url}/detect/dashboard",
                json=payload,
                timeout=self.timeout,
            )
            return resp.json() if resp.status_code == 200 else None
        except requests.exceptions.RequestException as exc:
            self._last_error = str(exc)
            return None


# ---------------------------------------------------------------------------
# Global singleton
# ---------------------------------------------------------------------------

_client: AnomalyClient | None = None


def get_client(base_url: str | None = None) -> AnomalyClient:
    global _client  # noqa: PLW0603
    if _client is None:
        _client = AnomalyClient(base_url)
    return _client
