"""Tests for the /health endpoint."""

from __future__ import annotations


class TestHealthEndpoint:
    def test_healthy_with_models(self, loaded_client):
        resp = loaded_client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert "rolling_zscore" in data["models_loaded"]

    def test_no_models(self, empty_client):
        resp = empty_client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "no_models_loaded"
        assert data["models_loaded"] == []
