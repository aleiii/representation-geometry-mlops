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


# ============================================================================
# Monitoring Endpoint Tests
# ============================================================================


def test_monitoring_health_endpoint():
    """Test monitoring health endpoint."""
    response = client.get("/monitoring/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "healthy"
    assert "prediction_logging_enabled" in payload
    assert "prediction_database_exists" in payload
    assert "prediction_count" in payload


def test_monitoring_drift_no_predictions(monkeypatch, tmp_path):
    """Test drift endpoint when no predictions exist."""
    import representation_geometry.api as api_module

    # Configure to use non-existent prediction database
    monkeypatch.setattr(api_module, "PREDICTION_DB_FILE", tmp_path / "nonexistent.csv")

    response = client.post(
        "/monitoring/drift",
        json={"dataset": "cifar10", "n_latest": 10},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "insufficient_data"
    assert payload["total_predictions"] == 0


def test_monitoring_drift_insufficient_predictions(monkeypatch, tmp_path):
    """Test drift endpoint when not enough predictions exist."""
    import representation_geometry.api as api_module
    import pandas as pd

    # Create a prediction database with few entries
    db_path = tmp_path / "predictions.csv"
    df = pd.DataFrame({
        "timestamp": ["2024-01-01T00:00:00"] * 5,
        "image_hash": ["abc"] * 5,
        "predicted_class": [0] * 5,
        "predicted_label": ["cat"] * 5,
        "confidence": [0.9] * 5,
        "model_name": ["mlp"] * 5,
        "dataset": ["cifar10"] * 5,
        "checkpoint": ["test.ckpt"] * 5,
        "brightness": [128.0] * 5,
        "contrast": [50.0] * 5,
        "red_mean": [120.0] * 5,
        "green_mean": [125.0] * 5,
        "blue_mean": [130.0] * 5,
        "red_std": [40.0] * 5,
        "green_std": [42.0] * 5,
        "blue_std": [45.0] * 5,
        "sharpness": [100.0] * 5,
        "saturation_mean": [0.3] * 5,
        "saturation_std": [0.1] * 5,
        "aspect_ratio": [1.0] * 5,
    })
    df.to_csv(db_path, index=False)

    monkeypatch.setattr(api_module, "PREDICTION_DB_FILE", db_path)

    response = client.post(
        "/monitoring/drift",
        json={"dataset": "cifar10", "n_latest": 50},  # Request more than available
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "insufficient_data"
    assert payload["total_predictions"] == 5


def test_root_includes_monitoring_endpoints():
    """Test that root endpoint lists monitoring endpoints."""
    response = client.get("/")
    assert response.status_code == 200
    payload = response.json()
    assert "monitoring_drift" in payload["endpoints"]
    assert "monitoring_health" in payload["endpoints"]


# ============================================================================
# Prometheus Metrics Tests
# ============================================================================


def test_metrics_endpoint():
    """Test that metrics endpoint returns Prometheus format."""
    response = client.get("/metrics")
    assert response.status_code == 200
    # Check content type
    assert "text/plain" in response.headers.get("content-type", "")
    # Check for expected metric names in response
    content = response.text
    # The response should contain Prometheus metric format
    assert "# HELP" in content or "Metrics collection is disabled" in content


def test_metrics_endpoint_format():
    """Test that metrics endpoint returns valid Prometheus format."""
    import representation_geometry.api as api_module

    # Ensure metrics are enabled
    original_value = api_module.METRICS_ENABLED
    api_module.METRICS_ENABLED = True

    try:
        response = client.get("/metrics")
        assert response.status_code == 200
        content = response.text

        # Check for expected metric definitions
        expected_metrics = [
            "api_requests_total",
            "api_request_latency_seconds",
            "predictions_total",
            "prediction_confidence",
            "api_errors_total",
            "models_loaded_total",
        ]

        # At least some metrics should be present
        found_metrics = [m for m in expected_metrics if m in content]
        assert len(found_metrics) > 0, f"No expected metrics found. Content: {content[:500]}"
    finally:
        api_module.METRICS_ENABLED = original_value


def test_metrics_disabled(monkeypatch):
    """Test that metrics endpoint works when disabled."""
    import representation_geometry.api as api_module

    monkeypatch.setattr(api_module, "METRICS_ENABLED", False)

    response = client.get("/metrics")
    assert response.status_code == 200
    assert "disabled" in response.text.lower()


def test_root_includes_metrics_info():
    """Test that root endpoint includes metrics information."""
    response = client.get("/")
    assert response.status_code == 200
    payload = response.json()
    assert "metrics" in payload["endpoints"]
    assert "metrics" in payload
    assert "enabled" in payload["metrics"]
    assert "endpoint" in payload["metrics"]
