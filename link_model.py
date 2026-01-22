"""Promote a staged W&B registry model to production."""

from __future__ import annotations

import os
import sys

import wandb

REGISTRY_PREFIX = "wandb-registry-"


def _parse_registry_info(artifact_path: str, registry_name: str | None, collection: str | None) -> tuple[str, str]:
    registry_value = registry_name
    collection_value = collection

    if artifact_path.startswith(REGISTRY_PREFIX) and ":" in artifact_path:
        registry_and_collection = artifact_path.split(":", 1)[0]
        remainder = registry_and_collection[len(REGISTRY_PREFIX) :]
        if "/" in remainder:
            inferred_registry, inferred_collection = remainder.split("/", 1)
            registry_value = registry_value or inferred_registry
            collection_value = collection_value or inferred_collection

    if not registry_value:
        registry_value = "model"
    if not collection_value:
        parts = artifact_path.split("/")
        if len(parts) >= 3 and ":" in parts[-1]:
            collection_value = parts[-1].split(":", 1)[0]
    if not collection_value:
        raise ValueError("WANDB_COLLECTION is required or must be inferable from MODEL_NAME.")

    return registry_value, collection_value


def main() -> int:
    artifact_path = os.getenv("WANDB_ARTIFACT_PATH") or os.getenv("MODEL_NAME")
    if not artifact_path:
        print("WANDB_ARTIFACT_PATH or MODEL_NAME is required.", file=sys.stderr)
        return 1

    registry_name = os.getenv("WANDB_REGISTRY_NAME")
    collection = os.getenv("WANDB_COLLECTION")
    production_alias = os.getenv("WANDB_PRODUCTION_ALIAS", "production")

    registry_name, collection = _parse_registry_info(artifact_path, registry_name, collection)

    project = os.getenv("WANDB_PROJECT") or "model-registry-ci"
    entity = os.getenv("WANDB_ENTITY")

    run = wandb.init(project=project, entity=entity, job_type="registry_promotion")
    try:
        artifact = run.use_artifact(artifact_path)
        target_path = f"{REGISTRY_PREFIX}{registry_name}/{collection}"
        run.link_artifact(artifact, target_path, aliases=[production_alias])
        print(f"Linked {artifact_path} to {target_path} with alias '{production_alias}'.")
    finally:
        run.finish()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
