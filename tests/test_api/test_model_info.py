"""Tests for the /models endpoints."""

from __future__ import annotations


class TestModelInfoEndpoints:
    def test_models_list(self, loaded_client):
        resp = loaded_client.get("/models")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["models"]) == 1
        assert data["models"][0]["name"] == "rolling_zscore"

    def test_model_detail(self, loaded_client):
        resp = loaded_client.get("/models/rolling_zscore")
        assert resp.status_code == 200
        data = resp.json()
        assert data["model_class"] == "RollingZScoreAnomalyDetector"
        assert "window" in data["parameters"]

    def test_unknown_model_404(self, loaded_client):
        resp = loaded_client.get("/models/nonexistent")
        assert resp.status_code == 404

    def test_no_models_503(self, empty_client):
        resp = empty_client.get("/models")
        assert resp.status_code == 503
