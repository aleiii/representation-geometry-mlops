"""FastAPI application for serving model predictions and representations."""

from __future__ import annotations

import base64
import binascii
import logging
import os
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
from fastapi import FastAPI, HTTPException
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, Field
from torchvision import transforms
from PIL import Image

from representation_geometry.model import MLPClassifier, ResNet18Classifier

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Neural Representation Geometry API",
    description="API for serving trained models and inspecting representations",
    version="0.0.1",
)


CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

STL10_CLASSES = [
    "airplane",
    "bird",
    "car",
    "cat",
    "deer",
    "dog",
    "horse",
    "monkey",
    "ship",
    "truck",
]


@dataclass
class LoadedModel:
    model: torch.nn.Module
    transform: transforms.Compose
    model_name: str
    dataset_name: Optional[str]
    checkpoint_path: Path
    config_path: Optional[Path]
    device: torch.device


# Global model cache
MODEL_CACHE: Dict[str, LoadedModel] = {}


class PredictionRequest(BaseModel):
    """Request model for predictions."""

    image_base64: str = Field(..., description="Base64-encoded image (PNG/JPEG).")
    checkpoint: Optional[str] = Field(
        None, description="Path or filename of a .ckpt file. Defaults to newest checkpoint in search paths."
    )
    model_name: Optional[str] = Field(None, description="Model architecture: 'mlp' or 'resnet18'.")
    config_path: Optional[str] = Field(None, description="Optional path to a Hydra config YAML.")
    dataset: Optional[str] = Field(None, description="Dataset name (cifar10 or stl10).")
    top_k: int = Field(5, ge=1, description="Number of top classes to return.")


class PredictionResponse(BaseModel):
    """Response model for predictions."""

    predicted_class: int
    predicted_label: Optional[str]
    confidence: float
    class_probabilities: Dict[int, float]
    model_name: str
    dataset: Optional[str]
    checkpoint: str


class RepresentationRequest(BaseModel):
    """Request for extracting intermediate representations."""

    image_base64: str = Field(..., description="Base64-encoded image (PNG/JPEG).")
    checkpoint: Optional[str] = Field(
        None, description="Path or filename of a .ckpt file. Defaults to newest checkpoint in search paths."
    )
    model_name: Optional[str] = Field(None, description="Model architecture: 'mlp' or 'resnet18'.")
    config_path: Optional[str] = Field(None, description="Optional path to a Hydra config YAML.")
    dataset: Optional[str] = Field(None, description="Dataset name (cifar10 or stl10).")
    layers: Optional[List[str]] = Field(None, description="Optional list of layer names to return.")


class RepresentationResponse(BaseModel):
    """Response containing layer activations."""

    activations: Dict[str, List[float]]
    model_name: str
    dataset: Optional[str]
    checkpoint: str


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    models_loaded: List[str]
    device: str


class ModelInfo(BaseModel):
    """Information about an available checkpoint."""

    checkpoint: str
    model_name: Optional[str]
    dataset: Optional[str]
    config_path: Optional[str]


def _model_search_paths() -> List[Path]:
    env_value = os.getenv("MODEL_DIRS", "")
    if env_value:
        paths = [Path(p).expanduser() for p in env_value.split(os.pathsep) if p]
    else:
        paths = [Path("models"), Path("outputs")]
    return [p for p in paths if p.exists()]


def _find_checkpoints(search_paths: Sequence[Path]) -> List[Path]:
    checkpoints: List[Path] = []
    for root in search_paths:
        if root.is_file() and root.suffix == ".ckpt":
            checkpoints.append(root)
        elif root.is_dir():
            checkpoints.extend(root.rglob("*.ckpt"))
    return checkpoints


def _resolve_checkpoint_path(checkpoint: Optional[str], model_name: Optional[str]) -> Path:
    search_paths = _model_search_paths()
    if checkpoint:
        candidate = Path(checkpoint)
        if candidate.exists():
            return candidate
        if not search_paths:
            raise HTTPException(status_code=404, detail=f"Checkpoint not found: {checkpoint}")
        matches = [p for p in _find_checkpoints(search_paths) if p.name == checkpoint or p.stem == checkpoint]
        if not matches:
            raise HTTPException(status_code=404, detail=f"Checkpoint not found: {checkpoint}")
        matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return matches[0]

    if not search_paths:
        raise HTTPException(
            status_code=404, detail="No model search paths found. Set MODEL_DIRS or provide checkpoint."
        )

    checkpoints = _find_checkpoints(search_paths)
    if model_name:
        checkpoints = [p for p in checkpoints if model_name.lower() in p.name.lower()]
    if not checkpoints:
        raise HTTPException(status_code=404, detail="No checkpoints found. Provide a checkpoint or add one to models/.")

    checkpoints.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return checkpoints[0]


def _resolve_config_path(checkpoint_path: Path, config_path: Optional[str]) -> Optional[Path]:
    if config_path:
        candidate = Path(config_path)
        if candidate.exists():
            return candidate
        raise HTTPException(status_code=404, detail=f"Config file not found: {config_path}")

    candidates: List[Path] = []
    for pattern in ("config*.yaml", "config*.yml"):
        candidates.extend(checkpoint_path.parent.glob(pattern))

    if not candidates:
        return None

    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _load_config(config_path: Optional[Path]) -> Optional[DictConfig]:
    if not config_path:
        return None
    try:
        return OmegaConf.load(config_path)
    except Exception as exc:
        logger.warning("Failed to load config %s: %s", config_path, exc)
        return None


def _infer_model_name(checkpoint_path: Path, model_name: Optional[str], cfg: Optional[DictConfig]) -> str:
    if model_name:
        return model_name.lower()
    if cfg and cfg.get("model") and cfg.model.get("name"):
        return str(cfg.model.name).lower()
    lowered = checkpoint_path.stem.lower()
    if "resnet" in lowered:
        return "resnet18"
    if "mlp" in lowered:
        return "mlp"
    raise HTTPException(status_code=400, detail="Unable to infer model_name. Provide it explicitly.")


def _infer_dataset_name(checkpoint_path: Path, dataset: Optional[str], cfg: Optional[DictConfig]) -> Optional[str]:
    if dataset:
        return dataset.lower()
    if cfg and cfg.get("data") and cfg.data.get("name"):
        return str(cfg.data.name).lower()
    lowered = checkpoint_path.stem.lower()
    if "cifar" in lowered:
        return "cifar10"
    if "stl" in lowered:
        return "stl10"
    return None


def _model_class_for_name(model_name: str):
    if model_name == "mlp":
        return MLPClassifier
    if model_name == "resnet18":
        return ResNet18Classifier
    raise HTTPException(status_code=400, detail=f"Unknown model_name: {model_name}")


def _default_normalization(dataset: Optional[str]) -> tuple[list[float], list[float]]:
    # ImageNet defaults (used in configs)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    if dataset in {"cifar10", "stl10"}:
        return mean, std
    return mean, std


def _infer_image_size(dataset: Optional[str], cfg: Optional[DictConfig]) -> int:
    if cfg and cfg.get("data"):
        data_cfg = cfg.data
        if data_cfg.get("resize_to"):
            return int(data_cfg.resize_to)
        if data_cfg.get("image_size"):
            return int(data_cfg.image_size)
    if dataset == "stl10":
        return 96
    return 32


def _build_transform(dataset: Optional[str], cfg: Optional[DictConfig]) -> transforms.Compose:
    mean, std = _default_normalization(dataset)
    if cfg and cfg.get("data") and cfg.data.get("normalization"):
        norm_cfg = cfg.data.normalization
        if norm_cfg.get("mean"):
            mean = list(norm_cfg.mean)
        if norm_cfg.get("std"):
            std = list(norm_cfg.std)

    image_size = _infer_image_size(dataset, cfg)

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def _decode_image(image_base64: str) -> Image.Image:
    payload = image_base64
    if "," in payload:
        payload = payload.split(",", 1)[1]
    try:
        binary = base64.b64decode(payload)
    except binascii.Error as exc:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image payload: {exc}") from exc

    try:
        image = Image.open(BytesIO(binary)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to decode image: {exc}") from exc
    return image


def _class_names_for_dataset(dataset: Optional[str]) -> Optional[List[str]]:
    if dataset == "cifar10":
        return CIFAR10_CLASSES
    if dataset == "stl10":
        return STL10_CLASSES
    return None


def _load_model_bundle(request: PredictionRequest | RepresentationRequest) -> LoadedModel:
    checkpoint_path = _resolve_checkpoint_path(request.checkpoint, request.model_name)
    config_path = _resolve_config_path(checkpoint_path, request.config_path)
    cfg = _load_config(config_path)

    model_name = _infer_model_name(checkpoint_path, request.model_name, cfg)
    dataset_name = _infer_dataset_name(checkpoint_path, request.dataset, cfg)

    cache_key = str(checkpoint_path.resolve())
    if cache_key in MODEL_CACHE:
        return MODEL_CACHE[cache_key]

    model_cls = _model_class_for_name(model_name)
    try:
        model = model_cls.load_from_checkpoint(str(checkpoint_path), weights_only=False)
    except TypeError:
        # Fallback for Lightning versions that don't expose weights_only
        model = model_cls.load_from_checkpoint(str(checkpoint_path))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load checkpoint: {exc}") from exc

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    if hasattr(model, "register_hooks"):
        try:
            model.register_hooks()
        except Exception as exc:
            logger.warning("Failed to register hooks on %s: %s", model_name, exc)

    transform = _build_transform(dataset_name, cfg)

    bundle = LoadedModel(
        model=model,
        transform=transform,
        model_name=model_name,
        dataset_name=dataset_name,
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        device=device,
    )
    MODEL_CACHE[cache_key] = bundle
    return bundle


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return {
        "status": "healthy",
        "models_loaded": list(MODEL_CACHE.keys()),
        "device": device,
    }


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Neural Representation Geometry API",
        "version": "0.0.1",
        "endpoints": {
            "health": "/health",
            "models": "/models",
            "predict": "/predict",
            "representations": "/representations",
        },
    }


@app.get("/models", response_model=List[ModelInfo])
async def list_models():
    """List available model checkpoints."""
    search_paths = _model_search_paths()
    checkpoints = _find_checkpoints(search_paths)

    results: List[ModelInfo] = []
    for checkpoint in checkpoints:
        config_path = _resolve_config_path(checkpoint, None)
        cfg = _load_config(config_path)
        model_name = None
        dataset_name = None
        if cfg:
            if cfg.get("model") and cfg.model.get("name"):
                model_name = str(cfg.model.name)
            if cfg.get("data") and cfg.data.get("name"):
                dataset_name = str(cfg.data.name)

        results.append(
            ModelInfo(
                checkpoint=str(checkpoint),
                model_name=model_name,
                dataset=dataset_name,
                config_path=str(config_path) if config_path else None,
            )
        )

    return results


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Run inference on a base64-encoded image."""
    bundle = _load_model_bundle(request)
    image = _decode_image(request.image_base64)
    tensor = bundle.transform(image).unsqueeze(0).to(bundle.device)

    with torch.inference_mode():
        logits = bundle.model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu()

    if probs.numel() == 0:
        raise HTTPException(status_code=500, detail="Model returned empty predictions.")

    top_k = min(request.top_k, probs.numel())
    top_probs, top_indices = torch.topk(probs, top_k)
    predicted_class = int(top_indices[0])
    confidence = float(top_probs[0])

    class_probs = {int(i): float(p) for i, p in enumerate(probs)}
    class_names = _class_names_for_dataset(bundle.dataset_name)
    predicted_label = class_names[predicted_class] if class_names else None

    return PredictionResponse(
        predicted_class=predicted_class,
        predicted_label=predicted_label,
        confidence=confidence,
        class_probabilities=class_probs,
        model_name=bundle.model_name,
        dataset=bundle.dataset_name,
        checkpoint=str(bundle.checkpoint_path),
    )


@app.post("/representations", response_model=RepresentationResponse)
async def representations(request: RepresentationRequest):
    """Return intermediate layer activations for a base64-encoded image."""
    bundle = _load_model_bundle(request)
    image = _decode_image(request.image_base64)
    tensor = bundle.transform(image).unsqueeze(0).to(bundle.device)

    if hasattr(bundle.model, "activations"):
        try:
            bundle.model.activations = {}
        except Exception:
            pass

    with torch.inference_mode():
        _ = bundle.model(tensor)

    activations: Dict[str, List[float]] = {}
    raw_activations = getattr(bundle.model, "activations", {})

    for name, activation in raw_activations.items():
        if request.layers and name not in request.layers:
            continue
        if isinstance(activation, torch.Tensor):
            values = activation
            if values.dim() > 2:
                values = values.mean(dim=(-2, -1))
            values = values.squeeze(0).detach().cpu().flatten()
            activations[name] = values.tolist()

    if not activations:
        raise HTTPException(status_code=404, detail="No activations found. Check layer names or model hooks.")

    return RepresentationResponse(
        activations=activations,
        model_name=bundle.model_name,
        dataset=bundle.dataset_name,
        checkpoint=str(bundle.checkpoint_path),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
