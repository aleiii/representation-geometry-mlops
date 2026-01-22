"""Smoke test for W&B-staged model artifacts."""

from __future__ import annotations

import os
import tempfile

import pytest
import wandb


def test_wandb_model_smoke():
    """Load a staged model artifact from W&B and report success."""
    model_name = os.getenv("MODEL_NAME")
    if not model_name:
        pytest.skip("MODEL_NAME not set; skipping staged model test.")

    api_key = os.getenv("WANDB_API_KEY")
    if not api_key:
        pytest.skip("WANDB_API_KEY not set; skipping staged model test.")

    entity = os.getenv("WANDB_ENTITY")
    project = os.getenv("WANDB_PROJECT")
    api = wandb.Api(api_key=api_key, overrides={"entity": entity, "project": project})

    artifact = api.artifact(model_name)
    with tempfile.TemporaryDirectory(prefix="wandb_artifact_") as tmpdir:
        artifact.download(root=tmpdir)

    print("ok")
