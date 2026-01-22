"""W&B model registry helpers."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import wandb
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


def _normalize_path(path_str: str) -> Optional[Path]:
    if not path_str:
        return None
    path = Path(path_str)
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    return path


def _find_latest_ckpt(output_dir: Path) -> Optional[Path]:
    if not output_dir.exists():
        return None
    candidates = sorted(output_dir.glob("*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def _resolve_checkpoint_path(
    trainer: Any,
    output_dir: Path,
    prefer_best: bool = True,
) -> Tuple[Optional[Path], str]:
    checkpoint_callback = getattr(trainer, "checkpoint_callback", None)
    best_path = _normalize_path(getattr(checkpoint_callback, "best_model_path", ""))
    last_path = _normalize_path(getattr(checkpoint_callback, "last_model_path", ""))

    if prefer_best and best_path and best_path.exists():
        return best_path, "best"
    if last_path and last_path.exists():
        return last_path, "last"
    if best_path and best_path.exists():
        return best_path, "best"

    fallback = _find_latest_ckpt(output_dir)
    if fallback:
        return fallback, "latest"

    return None, "unknown"


def _coerce_metric(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


def _sanitize_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    return {key: _coerce_metric(value) for key, value in metrics.items()}


def _ensure_list(value: Any) -> Iterable[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    return [str(value)]


def register_model_artifact(
    wandb_logger: WandbLogger,
    cfg: DictConfig,
    trainer: Any,
    test_results: Optional[list] = None,
    config_path: Optional[Path] = None,
) -> Optional[str]:
    """Log a model checkpoint as a W&B artifact and link it to a registry collection."""
    registry_cfg = cfg.wandb.get("registry", {})
    if not registry_cfg or not registry_cfg.get("enabled", False):
        logger.info("W&B registry logging disabled.")
        return None

    run = wandb_logger.experiment
    if run is None:
        logger.warning("W&B run is not available. Skipping registry logging.")
        return None

    output_dir = Path(cfg.paths.output_dir)
    if not output_dir.is_absolute():
        output_dir = (Path.cwd() / output_dir).resolve()

    checkpoint_path, checkpoint_kind = _resolve_checkpoint_path(
        trainer=trainer,
        output_dir=output_dir,
        prefer_best=registry_cfg.get("use_best", True),
    )
    if not checkpoint_path or not checkpoint_path.exists():
        logger.warning("No checkpoint found for registry logging.")
        return None

    artifact_name = registry_cfg.get("artifact_name") or f"{cfg.model.name}-{cfg.data.name}"
    artifact_type = registry_cfg.get("artifact_type", "model")
    description = registry_cfg.get(
        "description",
        f"{cfg.model.name} on {cfg.data.name} ({checkpoint_kind} checkpoint)",
    )

    metadata = {
        "model_name": cfg.model.name,
        "dataset": cfg.data.name,
        "seed": cfg.seed,
        "checkpoint_kind": checkpoint_kind,
        "run_id": run.id,
        "run_name": run.name,
    }

    if registry_cfg.get("include_metrics", True) and test_results:
        result_payload = test_results[0] if isinstance(test_results, list) and test_results else test_results
        if isinstance(result_payload, dict):
            metadata["test_metrics"] = _sanitize_metrics(result_payload)

    extra_metadata = registry_cfg.get("metadata", {})
    if isinstance(extra_metadata, dict):
        metadata.update(extra_metadata)

    artifact = wandb.Artifact(name=artifact_name, type=artifact_type, description=description, metadata=metadata)
    artifact.add_file(str(checkpoint_path), name=checkpoint_path.name)

    if registry_cfg.get("include_config", True) and config_path and config_path.exists():
        artifact.add_file(str(config_path), name="config.yaml")

    logged_artifact = run.log_artifact(artifact)

    registry_name = registry_cfg.get("name", "model")
    collection = registry_cfg.get("collection") or artifact_name
    target_path = f"wandb-registry-{registry_name}/{collection}"
    registry_aliases = list(_ensure_list(registry_cfg.get("aliases")))
    if registry_cfg.get("link", True):
        if registry_aliases:
            run.link_artifact(logged_artifact, target_path, aliases=registry_aliases)
        else:
            run.link_artifact(logged_artifact, target_path)

    logger.info("Registered model artifact %s at %s", artifact_name, target_path)
    return target_path


def download_registry_model(
    registry_name: str,
    collection: str,
    alias_or_version: str = "latest",
    project: Optional[str] = None,
    entity: Optional[str] = None,
    download_dir: Optional[Path] = None,
) -> Path:
    """Download a model artifact from the W&B registry."""
    if download_dir is None:
        download_dir = Path("models") / "wandb_registry"

    download_dir.mkdir(parents=True, exist_ok=True)
    run = wandb.init(project=project, entity=entity, job_type="registry_download")
    try:
        artifact_ref = f"wandb-registry-{registry_name}/{collection}:{alias_or_version}"
        artifact = run.use_artifact(artifact_ref)
        return Path(artifact.download(root=str(download_dir)))
    finally:
        run.finish()
