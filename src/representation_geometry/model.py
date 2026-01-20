"""Neural network models for representation geometry analysis."""

import logging
from typing import Dict, List, Optional

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# Setup logging
logger = logging.getLogger(__name__)


class MLPClassifier(L.LightningModule):
    """Multi-Layer Perceptron classifier for image classification.

    This model flattens input images and passes them through
    fully connected layers with ReLU activations and optional dropout.
    """

    def __init__(
        self,
        input_size: int = 3072,  # 32*32*3 for CIFAR-10
        hidden_dims: List[int] = [512, 512],
        num_classes: int = 10,
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        optimizer: str = "adam",
        weight_decay: float = 0.0001,
        **kwargs,
    ):
        """Initialize MLP Classifier.

        Args:
            input_size: Flattened input dimension
            hidden_dims: List of hidden layer dimensions
            num_classes: Number of output classes
            dropout: Dropout probability
            learning_rate: Learning rate for optimizer
            optimizer: Optimizer type ('adam' or 'sgd')
            weight_decay: Weight decay for regularization
            **kwargs: Additional optimizer parameters
        """
        super().__init__()
        self.save_hyperparameters()

        self.input_size = input_size
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.optimizer_name = optimizer
        self.weight_decay = weight_decay
        self.optimizer_kwargs = kwargs

        # Build MLP layers
        layers = []
        prev_dim = input_size

        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, num_classes))

        self.network = nn.Sequential(*layers)

        # Store intermediate activations for representation extraction
        self.activations = {}

        logger.info(f"Initialized MLPClassifier: {hidden_dims} -> {num_classes} classes")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the MLP.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width)

        Returns:
            Logits of shape (batch_size, num_classes)
        """
        # Flatten the input
        x = x.view(x.size(0), -1)
        return self.network(x)

    def _shared_step(self, batch, batch_idx, stage: str):
        """Shared step for train/val/test.

        Args:
            batch: Tuple of (images, labels)
            batch_idx: Index of current batch
            stage: One of 'train', 'val', or 'test'

        Returns:
            Loss value
        """
        images, labels = batch
        logits = self(images)
        loss = F.cross_entropy(logits, labels)

        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()

        # Log metrics
        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"{stage}_acc", acc, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def training_step(self, batch, batch_idx):
        """Training step."""
        return self._shared_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        return self._shared_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        """Test step."""
        return self._shared_step(batch, batch_idx, "test")

    def configure_optimizers(self):
        """Configure optimizer and optional learning rate scheduler."""
        if self.optimizer_name.lower() == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                **self.optimizer_kwargs.get("adam", {}),
            )
        elif self.optimizer_name.lower() == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                momentum=self.optimizer_kwargs.get("momentum", 0.9),
                **self.optimizer_kwargs.get("sgd", {}),
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_name}")

        # Optional learning rate scheduler
        scheduler_config = self.optimizer_kwargs.get("scheduler", {})
        if scheduler_config.get("use", False):
            if scheduler_config["type"] == "step":
                scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer,
                    step_size=scheduler_config.get("step_size", 30),
                    gamma=scheduler_config.get("gamma", 0.1),
                )
            elif scheduler_config["type"] == "cosine":
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=scheduler_config.get("T_max", 100),
                    eta_min=scheduler_config.get("eta_min", 0),
                )
            else:
                return optimizer

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                },
            }

        return optimizer

    def register_hooks(self, layer_indices: Optional[List[int]] = None):
        """Register forward hooks to capture intermediate activations.

        Args:
            layer_indices: Indices of layers to hook (None = hook all ReLU layers)
        """
        self.activations = {}

        if layer_indices is None:
            # Hook all ReLU layers by default
            layer_indices = [i for i, layer in enumerate(self.network) if isinstance(layer, nn.ReLU)]

        def get_activation(name):
            def hook(model, input, output):
                self.activations[name] = output.detach()

            return hook

        for idx in layer_indices:
            layer_name = f"layer_{idx}"
            self.network[idx].register_forward_hook(get_activation(layer_name))
            logger.debug(f"Registered hook at layer {idx}")


class ResNet18Classifier(L.LightningModule):
    """ResNet-18 classifier for image classification.

    Uses torchvision's ResNet-18 architecture, optionally modified
    for smaller images like CIFAR-10.
    """

    def __init__(
        self,
        num_classes: int = 10,
        pretrained: bool = False,
        modify_first_conv: bool = True,
        learning_rate: float = 0.01,
        optimizer: str = "sgd",
        momentum: float = 0.9,
        weight_decay: float = 0.0005,
        **kwargs,
    ):
        """Initialize ResNet-18 Classifier.

        Args:
            num_classes: Number of output classes
            pretrained: Whether to use pretrained ImageNet weights
            modify_first_conv: Adapt first conv layer for 32x32 images (CIFAR-10)
            learning_rate: Learning rate for optimizer
            optimizer: Optimizer type ('sgd' or 'adam')
            momentum: Momentum for SGD
            weight_decay: Weight decay for regularization
            **kwargs: Additional optimizer parameters
        """
        super().__init__()
        self.save_hyperparameters()

        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.optimizer_name = optimizer
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.optimizer_kwargs = kwargs

        # Load ResNet-18
        if pretrained:
            weights = models.ResNet18_Weights.DEFAULT
            self.model = models.resnet18(weights=weights)
        else:
            self.model = models.resnet18(weights=None)

        # Modify first conv layer for smaller images (CIFAR-10: 32x32)
        if modify_first_conv:
            # Original: Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
            # Modified: Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
            self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            # Remove max pooling for small images
            self.model.maxpool = nn.Identity()

        # Replace final fully connected layer
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

        # Store intermediate activations
        self.activations = {}

        logger.info(f"Initialized ResNet18Classifier: {num_classes} classes, pretrained={pretrained}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ResNet-18.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width)

        Returns:
            Logits of shape (batch_size, num_classes)
        """
        return self.model(x)

    def _shared_step(self, batch, batch_idx, stage: str):
        """Shared step for train/val/test.

        Args:
            batch: Tuple of (images, labels)
            batch_idx: Index of current batch
            stage: One of 'train', 'val', or 'test'

        Returns:
            Loss value
        """
        images, labels = batch
        logits = self(images)
        loss = F.cross_entropy(logits, labels)

        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()

        # Log metrics
        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"{stage}_acc", acc, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def training_step(self, batch, batch_idx):
        """Training step."""
        return self._shared_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        return self._shared_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        """Test step."""
        return self._shared_step(batch, batch_idx, "test")

    def configure_optimizers(self):
        """Configure optimizer and optional learning rate scheduler."""
        if self.optimizer_name.lower() == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.learning_rate,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
                **self.optimizer_kwargs.get("sgd", {}),
            )
        elif self.optimizer_name.lower() == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                **self.optimizer_kwargs.get("adam", {}),
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_name}")

        # Optional learning rate scheduler
        scheduler_config = self.optimizer_kwargs.get("scheduler", {})
        if scheduler_config.get("use", False):
            if scheduler_config["type"] == "step":
                scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer,
                    step_size=scheduler_config.get("step_size", 30),
                    gamma=scheduler_config.get("gamma", 0.1),
                )
            elif scheduler_config["type"] == "cosine":
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=scheduler_config.get("T_max", 100),
                    eta_min=scheduler_config.get("eta_min", 0),
                )
            else:
                return optimizer

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                },
            }

        return optimizer

    def register_hooks(self, layer_names: Optional[List[str]] = None):
        """Register forward hooks to capture intermediate activations.

        Args:
            layer_names: Names of ResNet layers to hook
                       (e.g., ['conv1', 'layer1', 'layer2', 'layer3', 'layer4'])
                       If None, hooks all major layers
        """
        self.activations = {}

        if layer_names is None:
            layer_names = ["conv1", "layer1", "layer2", "layer3", "layer4", "avgpool"]

        def get_activation(name):
            def hook(model, input, output):
                self.activations[name] = output.detach()

            return hook

        for name in layer_names:
            if hasattr(self.model, name):
                layer = getattr(self.model, name)
                layer.register_forward_hook(get_activation(name))
                logger.debug(f"Registered hook at {name}")


def extract_representations(
    model: L.LightningModule,
    dataloader: torch.utils.data.DataLoader,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict[str, torch.Tensor]:
    """Extract intermediate layer representations from a model.

    Args:
        model: PyTorch Lightning model with registered hooks
        dataloader: DataLoader to extract representations from
        device: Device to run inference on

    Returns:
        Dictionary mapping layer names to activation tensors
    """
    model = model.to(device)
    model.eval()

    all_activations = {}

    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(dataloader):
            images = images.to(device)
            _ = model(images)

            # Collect activations from this batch
            for layer_name, activation in model.activations.items():
                if layer_name not in all_activations:
                    all_activations[layer_name] = []
                # Flatten spatial dimensions if needed
                if len(activation.shape) > 2:
                    activation = activation.mean(dim=[-2, -1])  # Global average pooling
                all_activations[layer_name].append(activation.cpu())

    # Concatenate all batches
    for layer_name in all_activations:
        all_activations[layer_name] = torch.cat(all_activations[layer_name], dim=0)
        logger.info(f"Extracted {layer_name}: shape {all_activations[layer_name].shape}")

    return all_activations


if __name__ == "__main__":
    # Test model instantiation
    print("Testing MLPClassifier...")
    mlp = MLPClassifier()
    x = torch.randn(4, 3, 32, 32)
    output = mlp(x)
    print(f"MLP output shape: {output.shape}")

    print("\nTesting ResNet18Classifier...")
    resnet = ResNet18Classifier()
    output = resnet(x)
    print(f"ResNet-18 output shape: {output.shape}")

    print("\nModel tests passed!")
