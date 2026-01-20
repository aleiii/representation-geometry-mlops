"""Training script with Hydra configuration and W&B logging."""

import logging
from pathlib import Path
from typing import Optional

import hydra
import lightning as L
from hydra.utils import instantiate
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.profilers import Profiler
from omegaconf import DictConfig, OmegaConf

from representation_geometry.data import CIFAR10DataModule, STL10DataModule
from representation_geometry.model import MLPClassifier, ResNet18Classifier

# Setup logging (Hydra automatically handles file logging to run directory)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


def get_datamodule(cfg: DictConfig) -> L.LightningDataModule:
    """Create data module based on configuration.

    Args:
        cfg: Hydra configuration

    Returns:
        LightningDataModule for specified dataset
    """
    dataset_name = cfg.data.name.lower()

    if dataset_name == "cifar10":
        logger.info("Creating CIFAR-10 DataModule")
        return CIFAR10DataModule(
            data_dir=cfg.data.data_dir,
            batch_size=cfg.data.batch_size,
            num_workers=cfg.data.num_workers,
            pin_memory=cfg.data.pin_memory,
            normalize_mean=cfg.data.normalization.mean,
            normalize_std=cfg.data.normalization.std,
        )
    elif dataset_name == "stl10":
        logger.info("Creating STL-10 DataModule")
        return STL10DataModule(
            data_dir=cfg.data.data_dir,
            batch_size=cfg.data.batch_size,
            num_workers=cfg.data.num_workers,
            pin_memory=cfg.data.pin_memory,
            resize_to=cfg.data.get("resize_to", 96),
            normalize_mean=cfg.data.normalization.mean,
            normalize_std=cfg.data.normalization.std,
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def get_model(cfg: DictConfig) -> L.LightningModule:
    """Create model based on configuration.

    Args:
        cfg: Hydra configuration

    Returns:
        LightningModule for specified model
    """
    model_name = cfg.model.name.lower()

    if model_name == "mlp":
        logger.info("Creating MLP Classifier")
        return MLPClassifier(
            input_size=cfg.model.input_size,
            hidden_dims=cfg.model.hidden_dims,
            num_classes=cfg.model.num_classes,
            dropout=cfg.model.dropout,
            learning_rate=cfg.model.learning_rate,
            optimizer=cfg.model.optimizer,
            weight_decay=cfg.model.weight_decay,
            adam=cfg.model.get("adam", {}),
            scheduler=cfg.model.get("scheduler", {}),
        )
    elif model_name == "resnet18":
        logger.info("Creating ResNet-18 Classifier")
        return ResNet18Classifier(
            num_classes=cfg.model.num_classes,
            pretrained=cfg.model.get("pretrained", False),
            modify_first_conv=cfg.model.get("modify_first_conv", True),
            learning_rate=cfg.model.learning_rate,
            optimizer=cfg.model.optimizer,
            momentum=cfg.model.get("momentum", 0.9),
            weight_decay=cfg.model.weight_decay,
            sgd=cfg.model.get("sgd", {}),
            scheduler=cfg.model.get("scheduler", {}),
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")


def setup_wandb_logger(cfg: DictConfig) -> Optional[WandbLogger]:
    """Setup Weights & Biases logger.

    Args:
        cfg: Hydra configuration

    Returns:
        WandbLogger instance or None if W&B is disabled
    """
    if not cfg.get("wandb", None):
        logger.info("W&B logging disabled")
        return None

    # Create experiment name
    exp_name = cfg.experiment.get("name")
    if exp_name is None:
        exp_name = f"{cfg.model.name}_{cfg.data.name}_seed{cfg.seed}"

    # Setup W&B logger
    wandb_logger = WandbLogger(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=exp_name,
        save_dir=cfg.wandb.save_dir,
        log_model=cfg.wandb.get("log_model", False),
        tags=cfg.experiment.get("tags", []),
        notes=cfg.experiment.get("notes", ""),
        config=OmegaConf.to_container(cfg, resolve=True),
        group=cfg.wandb.get("group", None),
    )

    logger.info(f"W&B logging initialized: project={cfg.wandb.project}, name={exp_name}")
    return wandb_logger


def setup_callbacks(cfg: DictConfig) -> list:
    """Setup training callbacks.

    Args:
        cfg: Hydra configuration

    Returns:
        List of Lightning callbacks
    """
    callbacks = []

    # Model checkpoint callback
    checkpoint_dir = Path(cfg.paths.output_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=f"{cfg.model.name}_{cfg.data.name}_seed{cfg.seed}" + "_{epoch:02d}_{val_loss:.3f}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        save_last=True,
        verbose=True,
    )
    callbacks.append(checkpoint_callback)
    logger.info(f"Model checkpoints will be saved to {checkpoint_dir}")

    # Early stopping callback (optional)
    if cfg.get("early_stopping", {}).get("enabled", False):
        early_stop_callback = EarlyStopping(
            monitor="val_loss",
            patience=cfg.early_stopping.get("patience", 10),
            mode="min",
            verbose=True,
        )
        callbacks.append(early_stop_callback)
        logger.info("Early stopping enabled")

    return callbacks


def setup_profiler(cfg: DictConfig) -> Optional[Profiler]:
    """Setup profiler based on configuration.

    Args:
        cfg: Hydra configuration

    Returns:
        Profiler instance or None if profiling is disabled
    """
    if cfg.get("profiler") is None:
        logger.info("Profiling disabled")
        return None

    # Create profiler output directory
    profiler_dir = Path(cfg.profiler.get("dirpath", "profiler"))
    profiler_dir.mkdir(parents=True, exist_ok=True)

    # Instantiate profiler from config
    profiler = instantiate(cfg.profiler)
    logger.info(f"Profiler enabled: {type(profiler).__name__}")
    logger.info(f"Profiler output directory: {profiler_dir}")

    return profiler


@hydra.main(version_base=None, config_path=str(CONFIG_DIR), config_name="config")
def main(cfg: DictConfig) -> None:
    """Main training function.

    Args:
        cfg: Hydra configuration loaded from configs/
    """
    # Log the Hydra run directory (where all outputs will be saved)
    logger.info(f"Hydra run directory: {Path.cwd()}")

    # Create output directories
    Path(cfg.paths.output_dir).mkdir(parents=True, exist_ok=True)

    # Print configuration
    logger.info("=" * 80)
    logger.info("Training Configuration:")
    logger.info("=" * 80)
    logger.info(OmegaConf.to_yaml(cfg))
    logger.info("=" * 80)

    # Set random seed for reproducibility
    L.seed_everything(cfg.seed, workers=True)
    logger.info(f"Random seed set to {cfg.seed}")

    # Setup data module
    logger.info("Setting up data module...")
    datamodule = get_datamodule(cfg)
    datamodule.prepare_data()
    datamodule.setup("fit")

    # Setup model
    logger.info("Setting up model...")
    model = get_model(cfg)

    # Setup W&B logger
    wandb_logger = setup_wandb_logger(cfg)

    # Setup callbacks
    callbacks = setup_callbacks(cfg)

    # Setup profiler
    profiler = setup_profiler(cfg)

    # Create Lightning Trainer
    logger.info("Creating Lightning Trainer...")
    trainer = L.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        precision=cfg.trainer.get("precision", 32),
        logger=wandb_logger,
        callbacks=callbacks,
        profiler=profiler,
        log_every_n_steps=cfg.trainer.get("log_every_n_steps", 10),
        check_val_every_n_epoch=cfg.trainer.get("check_val_every_n_epoch", 1),
        deterministic=True,  # For reproducibility
        enable_progress_bar=True,
    )

    # Train the model
    logger.info("Starting training...")
    logger.info(f"Model: {cfg.model.name}, Dataset: {cfg.data.name}, Seed: {cfg.seed}")

    trainer.fit(model, datamodule)

    # Test the model on test set
    logger.info("Evaluating on test set...")
    datamodule.setup("test")
    test_results = trainer.test(model, datamodule)

    # Log final results
    logger.info("=" * 80)
    logger.info("Training Complete!")
    logger.info(f"Test results: {test_results}")
    logger.info(f"Best model checkpoint: {trainer.checkpoint_callback.best_model_path}")
    logger.info(f"All outputs saved to: {Path.cwd()}")
    logger.info("=" * 80)

    # Save final configuration
    config_save_path = Path(cfg.paths.output_dir) / f"config_seed{cfg.seed}.yaml"
    with open(config_save_path, "w") as f:
        f.write(OmegaConf.to_yaml(cfg))
    logger.info(f"Configuration saved to {config_save_path}")

    # Finish W&B run
    if wandb_logger:
        wandb_logger.experiment.finish()


if __name__ == "__main__":
    main()
