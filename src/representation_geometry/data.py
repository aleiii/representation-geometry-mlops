"""Data loading and preprocessing for CIFAR-10 and STL-10 datasets."""

import logging
from pathlib import Path
from typing import Annotated, Optional

import lightning as L
import torch
import typer
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# Setup logging
logger = logging.getLogger(__name__)

data_sync = typer.Typer(add_completion=False, no_args_is_help=True)


class CIFAR10DataModule(L.LightningDataModule):
    """PyTorch Lightning DataModule for CIFAR-10 dataset.

    This module handles data loading, preprocessing, and augmentation
    for the CIFAR-10 dataset (32x32 RGB images, 10 classes).
    """

    def __init__(
        self,
        data_dir: str = "./data/raw",
        batch_size: int = 128,
        num_workers: int = 4,
        pin_memory: bool = True,
        train_val_split: float = 0.9,
        augment_train: bool = True,
        normalize_mean: tuple = (0.485, 0.456, 0.406),
        normalize_std: tuple = (0.229, 0.224, 0.225),
    ):
        """Initialize CIFAR-10 DataModule.

        Args:
            data_dir: Root directory for dataset storage
            batch_size: Batch size for dataloaders
            num_workers: Number of worker processes for data loading
            pin_memory: Whether to pin memory for faster GPU transfer
            train_val_split: Fraction of training data to use for training (rest for validation)
            augment_train: Whether to apply data augmentation to training set
            normalize_mean: Mean for normalization (ImageNet default)
            normalize_std: Std for normalization (ImageNet default)
        """
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.train_val_split = train_val_split
        self.augment_train = augment_train
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std

        # Dataset splits
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        logger.info(f"Initialized CIFAR10DataModule with batch_size={batch_size}")

    def prepare_data(self):
        """Download CIFAR-10 dataset if not already present."""
        # Download train and test sets
        datasets.CIFAR10(self.data_dir, train=True, download=True)
        datasets.CIFAR10(self.data_dir, train=False, download=True)
        logger.info(f"CIFAR-10 dataset prepared in {self.data_dir}")

    def setup(self, stage: Optional[str] = None):
        """Setup train/val/test datasets with appropriate transforms.

        Args:
            stage: Current stage ('fit', 'validate', 'test', or 'predict')
        """
        # Define transforms
        if self.augment_train:
            train_transform = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(self.normalize_mean, self.normalize_std),
                ]
            )
        else:
            train_transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(self.normalize_mean, self.normalize_std),
                ]
            )

        test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(self.normalize_mean, self.normalize_std),
            ]
        )

        # Setup for training/validation
        if stage == "fit" or stage is None:
            full_train = datasets.CIFAR10(self.data_dir, train=True, transform=train_transform)

            # Split into train and validation
            train_size = int(len(full_train) * self.train_val_split)
            val_size = len(full_train) - train_size

            self.train_dataset, self.val_dataset = random_split(
                full_train,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(42),  # Reproducible split
            )

            # Apply test transform to validation set
            self.val_dataset.dataset.transform = test_transform

            logger.info(f"Train size: {len(self.train_dataset)}, Val size: {len(self.val_dataset)}")

        # Setup for testing
        if stage == "test" or stage is None:
            self.test_dataset = datasets.CIFAR10(self.data_dir, train=False, transform=test_transform)
            logger.info(f"Test size: {len(self.test_dataset)}")

    def train_dataloader(self):
        """Return training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        """Return validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        """Return test dataloader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )


class STL10DataModule(L.LightningDataModule):
    """PyTorch Lightning DataModule for STL-10 dataset.

    This module handles data loading, preprocessing, and augmentation
    for the STL-10 dataset (96x96 RGB images, 10 classes).
    """

    def __init__(
        self,
        data_dir: str = "./data/raw",
        batch_size: int = 64,
        num_workers: int = 4,
        pin_memory: bool = True,
        resize_to: int = 96,
        normalize_mean: tuple = (0.485, 0.456, 0.406),
        normalize_std: tuple = (0.229, 0.224, 0.225),
        augment_train: bool = True,
    ):
        """Initialize STL-10 DataModule.

        Args:
            data_dir: Root directory for dataset storage
            batch_size: Batch size for dataloaders (smaller due to larger images)
            num_workers: Number of worker processes for data loading
            pin_memory: Whether to pin memory for faster GPU transfer
            resize_to: Target size to resize images (default 96x96)
            normalize_mean: Mean for normalization (ImageNet default)
            normalize_std: Std for normalization (ImageNet default)
            augment_train: Whether to apply data augmentation
        """
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.resize_to = resize_to
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std
        self.augment_train = augment_train

        # Dataset splits
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        logger.info(f"Initialized STL10DataModule with batch_size={batch_size}, resize_to={resize_to}")

    def prepare_data(self):
        """Download STL-10 dataset if not already present."""
        # Download train and test splits
        datasets.STL10(self.data_dir, split="train", download=True)
        datasets.STL10(self.data_dir, split="test", download=True)
        logger.info(f"STL-10 dataset prepared in {self.data_dir}")

    def setup(self, stage: Optional[str] = None):
        """Setup train/val/test datasets with appropriate transforms.

        Args:
            stage: Current stage ('fit', 'validate', 'test', or 'predict')
        """
        # Define transforms
        if self.augment_train:
            train_transform = transforms.Compose(
                [
                    transforms.Resize(self.resize_to),
                    transforms.RandomCrop(self.resize_to, padding=12),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(self.normalize_mean, self.normalize_std),
                ]
            )
        else:
            train_transform = transforms.Compose(
                [
                    transforms.Resize(self.resize_to),
                    transforms.ToTensor(),
                    transforms.Normalize(self.normalize_mean, self.normalize_std),
                ]
            )

        test_transform = transforms.Compose(
            [
                transforms.Resize(self.resize_to),
                transforms.ToTensor(),
                transforms.Normalize(self.normalize_mean, self.normalize_std),
            ]
        )

        # Setup for training/validation
        if stage == "fit" or stage is None:
            # STL-10 has a separate train set (5000 images)
            self.train_dataset = datasets.STL10(self.data_dir, split="train", transform=train_transform)

            # Use test set for validation (8000 images)
            # Or could split train set further
            self.val_dataset = datasets.STL10(self.data_dir, split="test", transform=test_transform)

            logger.info(f"Train size: {len(self.train_dataset)}, Val size: {len(self.val_dataset)}")

        # Setup for testing
        if stage == "test" or stage is None:
            self.test_dataset = datasets.STL10(self.data_dir, split="test", transform=test_transform)
            logger.info(f"Test size: {len(self.test_dataset)}")

    def train_dataloader(self):
        """Return training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        """Return validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        """Return test dataloader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )


@data_sync.command()
def sync(
    data_dir: Annotated[Path, typer.Option("--data-dir", help="Directory for raw datasets")] = Path("data/raw"),
    datasets: Annotated[
        list[str],
        typer.Option(
            "--dataset",
            help="Datasets to ensure are available. Repeat flag for multiple.",
        ),
    ] = ["cifar10", "stl10"],
) -> None:
    """Download datasets via torchvision."""
    logging.basicConfig(level=logging.INFO)
    data_dir.mkdir(parents=True, exist_ok=True)

    normalized = {dataset.strip().lower() for dataset in datasets if dataset.strip()}
    if not normalized:
        raise typer.BadParameter("Provide at least one --dataset (e.g. --dataset cifar10).")

    if "cifar10" in normalized:
        dm = CIFAR10DataModule(data_dir=str(data_dir))
        dm.prepare_data()
        dm.setup()
        logger.info("CIFAR-10 ready in %s", data_dir)

    if "stl10" in normalized:
        dm = STL10DataModule(data_dir=str(data_dir))
        dm.prepare_data()
        dm.setup()
        logger.info("STL-10 ready in %s", data_dir)


def main():
    """Entry point for the rep-geom-data command."""
    data_sync()


if __name__ == "__main__":
    main()
