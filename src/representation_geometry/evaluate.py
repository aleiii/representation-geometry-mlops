"""Model evaluation and representation similarity analysis."""

import logging
from pathlib import Path
from typing import Dict, Optional

import hydra
import lightning as L
import torch
from omegaconf import DictConfig, OmegaConf

from representation_geometry.data import CIFAR10DataModule, STL10DataModule
from representation_geometry.model import MLPClassifier, ResNet18Classifier, extract_representations
from representation_geometry.utils import Timer

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


def _center(activations: torch.Tensor) -> torch.Tensor:
    return activations - activations.mean(dim=0, keepdim=True)


def linear_cka(activations_x: torch.Tensor, activations_y: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """Compute linear CKA similarity between two activation matrices."""
    x = _center(activations_x.double())
    y = _center(activations_y.double())

    dot_xy = x.T @ y
    numerator = (dot_xy**2).sum()
    denom = torch.sqrt((x.T @ x).pow(2).sum()) * torch.sqrt((y.T @ y).pow(2).sum())
    return numerator / (denom + eps)


def _pca_reduce(activations: torch.Tensor, variance_threshold: float = 0.99, eps: float = 1e-8) -> torch.Tensor:
    x = _center(activations.double())
    if x.shape[0] < 2 or x.shape[1] < 2:
        return x

    u, s, _ = torch.linalg.svd(x, full_matrices=False)
    variances = s.pow(2)
    total = variances.sum()
    if total <= eps:
        return x

    cumulative = torch.cumsum(variances, dim=0) / total
    k = int(torch.searchsorted(cumulative, variance_threshold, right=False).item()) + 1
    k = max(1, min(k, s.numel()))
    return u[:, :k] * s[:k]


def _inv_sqrt(mat: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    eigvals, eigvecs = torch.linalg.eigh(mat)
    eigvals = torch.clamp(eigvals, min=eps)
    inv_sqrt_vals = eigvals.rsqrt()
    return (eigvecs * inv_sqrt_vals) @ eigvecs.T


def svcca_similarity(
    activations_x: torch.Tensor,
    activations_y: torch.Tensor,
    variance_threshold: float = 0.99,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute SVCCA similarity between two activation matrices."""
    x = _pca_reduce(activations_x, variance_threshold=variance_threshold, eps=eps)
    y = _pca_reduce(activations_y, variance_threshold=variance_threshold, eps=eps)

    if x.shape[0] < 2 or y.shape[0] < 2:
        return torch.tensor(0.0)

    x = _center(x)
    y = _center(y)

    n = min(x.shape[0], y.shape[0])
    x = x[:n]
    y = y[:n]

    cov_xx = (x.T @ x) / (n - 1)
    cov_yy = (y.T @ y) / (n - 1)
    cov_xy = (x.T @ y) / (n - 1)

    cov_xx = cov_xx + eps * torch.eye(cov_xx.shape[0], device=cov_xx.device, dtype=cov_xx.dtype)
    cov_yy = cov_yy + eps * torch.eye(cov_yy.shape[0], device=cov_yy.device, dtype=cov_yy.dtype)

    inv_sqrt_xx = _inv_sqrt(cov_xx, eps=eps)
    inv_sqrt_yy = _inv_sqrt(cov_yy, eps=eps)
    t_mat = inv_sqrt_xx @ cov_xy @ inv_sqrt_yy
    s = torch.linalg.svdvals(t_mat)
    return s.mean()


def _truncate_representations(reps: Dict[str, torch.Tensor], max_samples: Optional[int]) -> Dict[str, torch.Tensor]:
    if max_samples is None or max_samples <= 0:
        return reps
    truncated = {}
    for layer, tensor in reps.items():
        truncated[layer] = tensor[:max_samples]
    return truncated


def compute_representation_similarity(
    reps_a: Dict[str, torch.Tensor],
    reps_b: Dict[str, torch.Tensor],
    metrics: list[str],
    max_samples: Optional[int] = None,
    svcca_variance: float = 0.99,
) -> Dict[str, Dict[str, float]]:
    """Compute representation similarity for matching layers."""
    metrics = [metric.lower() for metric in metrics]
    supported = {"cka", "svcca"}
    unknown = set(metrics) - supported
    if unknown:
        raise ValueError(
            f"Unsupported metrics: {', '.join(sorted(unknown))}. Supported: {', '.join(sorted(supported))}"
        )
    reps_a = _truncate_representations(reps_a, max_samples)
    reps_b = _truncate_representations(reps_b, max_samples)

    common_layers = sorted(set(reps_a) & set(reps_b))
    if not common_layers:
        raise ValueError("No overlapping layer names between representations.")

    scores: Dict[str, Dict[str, float]] = {metric: {} for metric in metrics}
    for layer in common_layers:
        x = reps_a[layer].view(reps_a[layer].shape[0], -1)
        y = reps_b[layer].view(reps_b[layer].shape[0], -1)
        n = min(x.shape[0], y.shape[0])
        x = x[:n]
        y = y[:n]

        if "cka" in metrics:
            scores["cka"][layer] = float(linear_cka(x, y).item())
        if "svcca" in metrics:
            scores["svcca"][layer] = float(svcca_similarity(x, y, variance_threshold=svcca_variance).item())

    for metric in list(scores.keys()):
        if not scores[metric]:
            scores.pop(metric)

    return scores


def load_model_from_checkpoint(
    checkpoint_path: str,
    model_type: str = "mlp",
) -> L.LightningModule:
    """Load a trained model from checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint
        model_type: Type of model ('mlp' or 'resnet18')

    Returns:
        Loaded LightningModule
    """
    logger.info(f"Loading model from {checkpoint_path}")

    if model_type.lower() == "mlp":
        model = MLPClassifier.load_from_checkpoint(checkpoint_path)
    elif model_type.lower() == "resnet18":
        model = ResNet18Classifier.load_from_checkpoint(checkpoint_path)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.eval()
    logger.info(f"Model loaded successfully: {model_type}")
    return model


def evaluate_model(
    model: L.LightningModule,
    datamodule: L.LightningDataModule,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict[str, float]:
    """Evaluate model on test set.

    Args:
        model: Trained model
        datamodule: Data module with test set
        device: Device to run evaluation on

    Returns:
        Dictionary of evaluation metrics
    """
    model = model.to(device)
    model.eval()

    datamodule.setup("test")
    test_loader = datamodule.test_dataloader()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with Timer("Model evaluation"):
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(test_loader):
                images = images.to(device)
                labels = labels.to(device)

                logits = model(images)
                loss = torch.nn.functional.cross_entropy(logits, labels)

                total_loss += loss.item() * images.size(0)
                preds = torch.argmax(logits, dim=1)
                total_correct += (preds == labels).sum().item()
                total_samples += images.size(0)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    results = {
        "test_loss": avg_loss,
        "test_accuracy": accuracy,
        "total_samples": total_samples,
    }

    logger.info(f"Evaluation results: loss={avg_loss:.4f}, accuracy={accuracy:.4f}")
    return results


def extract_and_save_representations(
    model: L.LightningModule,
    datamodule: L.LightningDataModule,
    output_path: str,
    layer_names: Optional[list] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """Extract representations and save to disk.

    Args:
        model: Trained model with hooks registered
        datamodule: Data module
        output_path: Path to save representations
        layer_names: Layer names to extract (None = all available)
        device: Device to use
    """
    # Register hooks if not already done
    if isinstance(model, MLPClassifier):
        model.register_hooks()
    elif isinstance(model, ResNet18Classifier):
        model.register_hooks(layer_names)

    datamodule.setup("test")
    test_loader = datamodule.test_dataloader()

    with Timer("Representation extraction"):
        representations = extract_representations(model, test_loader, device)

    # Save representations
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(representations, output_path)
    logger.info(f"Representations saved to {output_path}")

    # Print summary
    for layer_name, activations in representations.items():
        logger.info(f"  {layer_name}: {activations.shape}")


def _extract_representations_for_similarity(
    model: L.LightningModule,
    datamodule: L.LightningDataModule,
    layer_names: Optional[list],
    device: str,
) -> Dict[str, torch.Tensor]:
    if isinstance(model, MLPClassifier):
        model.register_hooks()
    elif isinstance(model, ResNet18Classifier):
        model.register_hooks(layer_names)

    datamodule.setup("test")
    test_loader = datamodule.test_dataloader()
    reps = extract_representations(model, test_loader, device)
    if layer_names:
        reps = {layer: reps[layer] for layer in layer_names if layer in reps}
    return reps


def compare_model_representations(
    model_a: L.LightningModule,
    model_b: L.LightningModule,
    datamodule: L.LightningDataModule,
    metrics: list[str],
    max_samples: Optional[int] = None,
    layer_names: Optional[list[str]] = None,
    svcca_variance: float = 0.99,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict[str, Dict[str, float]]:
    """Compute representation similarity between two models."""
    with Timer("Representation extraction (model A)"):
        reps_a = _extract_representations_for_similarity(model_a, datamodule, layer_names, device)
    with Timer("Representation extraction (model B)"):
        reps_b = _extract_representations_for_similarity(model_b, datamodule, layer_names, device)

    with Timer("Representation similarity"):
        scores = compute_representation_similarity(
            reps_a,
            reps_b,
            metrics=metrics,
            max_samples=max_samples,
            svcca_variance=svcca_variance,
        )

    return scores


@hydra.main(version_base=None, config_path=str(CONFIG_DIR), config_name="config")
def main(cfg: DictConfig) -> None:
    """Main evaluation function using Hydra configuration."""
    logger.info("=" * 80)
    logger.info("Evaluation Configuration:")
    logger.info("=" * 80)
    logger.info(OmegaConf.to_yaml(cfg))
    logger.info("=" * 80)

    # Setup data
    if cfg.data.name.lower() == "cifar10":
        datamodule = CIFAR10DataModule(
            data_dir=cfg.data.data_dir,
            batch_size=cfg.data.batch_size,
            num_workers=cfg.data.num_workers,
        )
    elif cfg.data.name.lower() == "stl10":
        datamodule = STL10DataModule(
            data_dir=cfg.data.data_dir,
            batch_size=cfg.data.batch_size,
            num_workers=cfg.data.num_workers,
        )
    else:
        raise ValueError(f"Unknown dataset: {cfg.data.name}")

    # Load model from checkpoint
    checkpoint_path = cfg.get("checkpoint_path", None)
    if checkpoint_path is None:
        raise ValueError("Please provide checkpoint_path in config or via CLI")

    model = load_model_from_checkpoint(checkpoint_path, cfg.model.name)

    # Evaluate model
    results = evaluate_model(model, datamodule)
    logger.info(f"Test accuracy: {results['test_accuracy']:.4f}")

    # Compare representations if reference checkpoint provided
    reference_checkpoint_path = cfg.get("reference_checkpoint_path", None)
    if reference_checkpoint_path:
        reference_model_type = cfg.get("reference_model_type", cfg.model.name)
        reference_model = load_model_from_checkpoint(reference_checkpoint_path, reference_model_type)

        metrics = cfg.get("similarity_metrics", ["cka", "svcca"])
        max_samples = cfg.get("similarity_max_samples", 2000)
        layer_names = cfg.get("similarity_layers", None)
        svcca_variance = cfg.get("svcca_variance", 0.99)

        scores = compare_model_representations(
            model,
            reference_model,
            datamodule,
            metrics=metrics,
            max_samples=max_samples,
            layer_names=layer_names,
            svcca_variance=svcca_variance,
        )

        for metric_name, layer_scores in scores.items():
            mean_score = sum(layer_scores.values()) / max(len(layer_scores), 1)
            logger.info("%s mean similarity: %.4f", metric_name.upper(), mean_score)
            for layer_name, score in layer_scores.items():
                logger.info("  %s %s: %.4f", metric_name.upper(), layer_name, score)

    # Extract representations if requested
    if cfg.get("extract_representations", False):
        output_path = (
            Path(cfg.paths.output_dir) / "representations" / f"{cfg.model.name}_{cfg.data.name}_seed{cfg.seed}.pt"
        )
        extract_and_save_representations(model, datamodule, str(output_path))


if __name__ == "__main__":
    main()
