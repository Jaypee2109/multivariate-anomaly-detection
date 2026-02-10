"""Tests for the /detect and /detect/dashboard endpoints."""

from __future__ import annotations


class TestDetectEndpoint:
    def test_success(self, loaded_client, sample_request_data):
        resp = loaded_client.post("/detect", json={"data": sample_request_data})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["results"]) == 1
        result = data["results"][0]
        assert result["model"] == "rolling_zscore"
        assert len(result["points"]) == 100
        assert all(isinstance(p["is_anomaly"], bool) for p in result["points"])

    def test_specific_model(self, loaded_client, sample_request_data):
        resp = loaded_client.post(
            "/detect",
            json={"data": sample_request_data, "models": ["rolling_zscore"]},
        )
        assert resp.status_code == 200
        assert len(resp.json()["results"]) == 1

    def test_unknown_model_400(self, loaded_client, sample_request_data):
        resp = loaded_client.post(
            "/detect",
            json={"data": sample_request_data, "models": ["nonexistent"]},
        )
        assert resp.status_code == 400

    def test_no_models_503(self, empty_client, sample_request_data):
        resp = empty_client.post("/detect", json={"data": sample_request_data})
        assert resp.status_code == 503

    def test_mismatched_lengths_400(self, loaded_client):
        resp = loaded_client.post(
            "/detect",
            json={
                "data": {
                    "timestamps": ["2020-01-01T00:00:00"],
                    "values": [1.0, 2.0],
                }
            },
        )
        assert resp.status_code == 400

    def test_too_few_points_400(self, loaded_client):
        resp = loaded_client.post(
            "/detect",
            json={
                "data": {
                    "timestamps": ["2020-01-01T00:00:00", "2020-01-01T01:00:00"],
                    "values": [1.0, 2.0],
                }
            },
        )
        assert resp.status_code == 400


class TestDashboardEndpoint:
    def test_success(self, loaded_client, sample_request_data):
        resp = loaded_client.post("/detect/dashboard", json={"data": sample_request_data})
        assert resp.status_code == 200
        data = resp.json()
        assert "chart_data" in data
        chart = data["chart_data"]
        assert len(chart["timestamps"]) == 100
        assert "rolling_zscore" in chart["models"]
        series = chart["models"]["rolling_zscore"]
        assert len(series["scores"]) == 100
        assert len(series["anomalies"]) == 100
