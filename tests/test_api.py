"""Tests for the FastAPI service."""

from __future__ import annotations

from fastapi.testclient import TestClient

from representation_geometry.api import app


client = TestClient(app)


def _set_empty_model_dir(monkeypatch, tmp_path):
    """Force the API to search an empty directory for checkpoints."""
    monkeypatch.setenv("MODEL_DIRS", str(tmp_path))


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "healthy"
    assert "device" in payload


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
