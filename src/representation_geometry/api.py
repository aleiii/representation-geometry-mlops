"""FastAPI application for serving model predictions and representations."""

from __future__ import annotations

import base64
import binascii
import csv
import hashlib
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from fastapi import BackgroundTasks, FastAPI, HTTPException
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, Field
from torchvision import transforms
from PIL import Image

from representation_geometry.model import MLPClassifier, ResNet18Classifier

logger = logging.getLogger(__name__)

# Lock for thread-safe CSV writing
_db_write_lock = Lock()

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

# Prediction logging configuration
PREDICTION_LOG_ENABLED = os.getenv("PREDICTION_LOG_ENABLED", "true").lower() in ("true", "1", "yes")
PREDICTION_LOG_DIR = Path(os.getenv("PREDICTION_LOG_DIR", "./api_logs"))
PREDICTION_DB_FILE = PREDICTION_LOG_DIR / "prediction_database.csv"

# CSV columns for prediction database
PREDICTION_DB_COLUMNS = [
    "timestamp",
    "image_hash",
    "predicted_class",
    "predicted_label",
    "confidence",
    "model_name",
    "dataset",
    "checkpoint",
    # Image features for drift detection
    "brightness",
    "contrast",
    "red_mean",
    "green_mean",
    "blue_mean",
    "red_std",
    "green_std",
    "blue_std",
    "sharpness",
    "saturation_mean",
    "saturation_std",
    "aspect_ratio",
]


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


def _extract_image_features(image: Image.Image) -> Dict[str, float]:
    """Extract structured features from an image for drift detection.

    Args:
        image: PIL Image in RGB format

    Returns:
        Dictionary of extracted features
    """
    img_array = np.array(image)

    # Ensure RGB format
    if img_array.ndim == 2:
        img_array = np.stack([img_array] * 3, axis=-1)
    elif img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]

    height, width = img_array.shape[:2]
    features: Dict[str, float] = {}

    # Overall brightness and contrast
    gray = np.mean(img_array, axis=-1)
    features["brightness"] = float(np.mean(gray))
    features["contrast"] = float(np.std(gray))

    # Per-channel statistics
    for i, channel in enumerate(["red", "green", "blue"]):
        features[f"{channel}_mean"] = float(np.mean(img_array[:, :, i]))
        features[f"{channel}_std"] = float(np.std(img_array[:, :, i]))

    # Aspect ratio
    features["aspect_ratio"] = float(width / height)

    # Sharpness estimation via Laplacian variance
    try:
        from scipy import ndimage

        laplacian_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
        laplacian = ndimage.convolve(gray.astype(np.float32), laplacian_kernel)
        features["sharpness"] = float(np.var(laplacian))
    except ImportError:
        features["sharpness"] = 0.0

    # Color saturation
    max_rgb = np.max(img_array, axis=-1)
    min_rgb = np.min(img_array, axis=-1)
    saturation = np.where(max_rgb > 0, (max_rgb - min_rgb) / (max_rgb + 1e-8), 0)
    features["saturation_mean"] = float(np.mean(saturation))
    features["saturation_std"] = float(np.std(saturation))

    return features


def _compute_image_hash(image: Image.Image) -> str:
    """Compute a hash of the image for deduplication."""
    img_bytes = image.tobytes()
    return hashlib.md5(img_bytes).hexdigest()[:16]


def _ensure_prediction_db_exists() -> None:
    """Ensure the prediction database directory and file exist."""
    if not PREDICTION_LOG_ENABLED:
        return

    PREDICTION_LOG_DIR.mkdir(parents=True, exist_ok=True)

    if not PREDICTION_DB_FILE.exists():
        with open(PREDICTION_DB_FILE, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=PREDICTION_DB_COLUMNS)
            writer.writeheader()
        logger.info(f"Created prediction database: {PREDICTION_DB_FILE}")


def _save_prediction_to_db(
    image: Image.Image,
    predicted_class: int,
    predicted_label: Optional[str],
    confidence: float,
    model_name: str,
    dataset: Optional[str],
    checkpoint: str,
) -> None:
    """Save prediction data to the database (runs as background task).

    Args:
        image: The input image
        predicted_class: Predicted class index
        predicted_label: Predicted class label
        confidence: Prediction confidence
        model_name: Name of the model used
        dataset: Dataset name
        checkpoint: Path to the checkpoint used
    """
    if not PREDICTION_LOG_ENABLED:
        return

    try:
        _ensure_prediction_db_exists()

        # Extract features for drift detection
        features = _extract_image_features(image)

        # Build row
        row = {
            "timestamp": datetime.utcnow().isoformat(),
            "image_hash": _compute_image_hash(image),
            "predicted_class": predicted_class,
            "predicted_label": predicted_label or "",
            "confidence": round(confidence, 6),
            "model_name": model_name,
            "dataset": dataset or "",
            "checkpoint": checkpoint,
            **{k: round(v, 4) for k, v in features.items()},
        }

        # Thread-safe write to CSV
        with _db_write_lock:
            with open(PREDICTION_DB_FILE, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=PREDICTION_DB_COLUMNS)
                writer.writerow(row)

        logger.debug(f"Saved prediction to database: class={predicted_class}, confidence={confidence:.4f}")

    except Exception as e:
        logger.error(f"Failed to save prediction to database: {e}")


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
            "prediction_stats": "/predictions/stats",
        },
        "prediction_logging": {
            "enabled": PREDICTION_LOG_ENABLED,
            "log_dir": str(PREDICTION_LOG_DIR),
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


class PredictionStats(BaseModel):
    """Statistics about logged predictions."""

    total_predictions: int
    logging_enabled: bool
    log_file: str
    unique_images: int
    class_distribution: Dict[str, int]
    avg_confidence: float
    earliest_timestamp: Optional[str]
    latest_timestamp: Optional[str]


@app.get("/predictions/stats", response_model=PredictionStats)
async def prediction_stats():
    """Get statistics about logged predictions."""
    if not PREDICTION_LOG_ENABLED:
        return PredictionStats(
            total_predictions=0,
            logging_enabled=False,
            log_file=str(PREDICTION_DB_FILE),
            unique_images=0,
            class_distribution={},
            avg_confidence=0.0,
            earliest_timestamp=None,
            latest_timestamp=None,
        )

    if not PREDICTION_DB_FILE.exists():
        return PredictionStats(
            total_predictions=0,
            logging_enabled=True,
            log_file=str(PREDICTION_DB_FILE),
            unique_images=0,
            class_distribution={},
            avg_confidence=0.0,
            earliest_timestamp=None,
            latest_timestamp=None,
        )

    try:
        import pandas as pd

        df = pd.read_csv(PREDICTION_DB_FILE)

        if len(df) == 0:
            return PredictionStats(
                total_predictions=0,
                logging_enabled=True,
                log_file=str(PREDICTION_DB_FILE),
                unique_images=0,
                class_distribution={},
                avg_confidence=0.0,
                earliest_timestamp=None,
                latest_timestamp=None,
            )

        # Calculate statistics
        class_dist = df["predicted_label"].value_counts().to_dict() if "predicted_label" in df.columns else {}

        return PredictionStats(
            total_predictions=len(df),
            logging_enabled=True,
            log_file=str(PREDICTION_DB_FILE),
            unique_images=df["image_hash"].nunique() if "image_hash" in df.columns else 0,
            class_distribution=class_dist,
            avg_confidence=float(df["confidence"].mean()) if "confidence" in df.columns else 0.0,
            earliest_timestamp=str(df["timestamp"].min()) if "timestamp" in df.columns else None,
            latest_timestamp=str(df["timestamp"].max()) if "timestamp" in df.columns else None,
        )
    except Exception as e:
        logger.error(f"Failed to read prediction stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to read prediction stats: {e}")


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest, background_tasks: BackgroundTasks):
    """Run inference on a base64-encoded image.

    Predictions are logged to a CSV database in the background for drift detection.
    Set PREDICTION_LOG_ENABLED=false to disable logging.
    """
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

    # Log prediction in background (non-blocking)
    background_tasks.add_task(
        _save_prediction_to_db,
        image=image,
        predicted_class=predicted_class,
        predicted_label=predicted_label,
        confidence=confidence,
        model_name=bundle.model_name,
        dataset=bundle.dataset_name,
        checkpoint=str(bundle.checkpoint_path),
    )

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
