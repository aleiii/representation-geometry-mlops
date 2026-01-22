"""Tests for the FastAPI service."""

from __future__ import annotations

from fastapi.testclient import TestClient

from representation_geometry.api import app


client = TestClient(app)


def _set_empty_model_dir(monkeypatch, tmp_path):
    """Force the API to search an empty directory for checkpoints."""
    monkeypatch.setenv("MODEL_DIRS", str(tmp_path))


def _disable_prediction_logging(monkeypatch, tmp_path):
    """Configure prediction logging to use a temp directory."""
    monkeypatch.setenv("PREDICTION_LOG_DIR", str(tmp_path / "api_logs"))
    monkeypatch.setenv("PREDICTION_LOG_ENABLED", "true")


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "healthy"
    assert "device" in payload


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    payload = response.json()
    assert payload["name"] == "Neural Representation Geometry API"
    assert "endpoints" in payload
    assert "prediction_logging" in payload
    assert "enabled" in payload["prediction_logging"]


def test_models_endpoint_empty(monkeypatch, tmp_path):
    _set_empty_model_dir(monkeypatch, tmp_path)
    response = client.get("/models")
    assert response.status_code == 200
    assert response.json() == []


def test_predict_returns_404_without_checkpoints(monkeypatch, tmp_path):
    _set_empty_model_dir(monkeypatch, tmp_path)
    response = client.post(
        "/predict",
        json={
            "image_base64": "aGVsbG8=",
        },
    )
    assert response.status_code == 404


def test_representations_returns_404_without_checkpoints(monkeypatch, tmp_path):
    _set_empty_model_dir(monkeypatch, tmp_path)
    response = client.post(
        "/representations",
        json={
            "image_base64": "aGVsbG8=",
        },
    )
    assert response.status_code == 404


def test_prediction_stats_endpoint_empty(monkeypatch, tmp_path):
    """Test prediction stats endpoint when no predictions exist."""
    _disable_prediction_logging(monkeypatch, tmp_path)

    # Need to reload the module to pick up env var changes
    import representation_geometry.api as api_module

    monkeypatch.setattr(api_module, "PREDICTION_LOG_DIR", tmp_path / "api_logs")
    monkeypatch.setattr(api_module, "PREDICTION_DB_FILE", tmp_path / "api_logs" / "prediction_database.csv")

    response = client.get("/predictions/stats")
    assert response.status_code == 200
    payload = response.json()
    assert payload["total_predictions"] == 0
    assert "logging_enabled" in payload


def test_prediction_stats_endpoint_disabled(monkeypatch, tmp_path):
    """Test prediction stats endpoint when logging is disabled."""
    import representation_geometry.api as api_module

    monkeypatch.setattr(api_module, "PREDICTION_LOG_ENABLED", False)

    response = client.get("/predictions/stats")
    assert response.status_code == 200
    payload = response.json()
    assert payload["logging_enabled"] is False
    assert payload["total_predictions"] == 0
