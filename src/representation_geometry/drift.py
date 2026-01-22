"""Data drift detection for image classification models.

This module provides functionality to detect data drift between reference
(training) data and current (production) data using the Evidently library.

For images, we extract structured features (brightness, contrast, color statistics)
since Evidently works with tabular data.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import torch
import typer
from PIL import Image
from torchvision import datasets
from tqdm import tqdm

from evidently import ColumnMapping
from evidently.metric_preset import DataDriftPreset, DataQualityPreset, TargetDriftPreset
from evidently.report import Report
from evidently.test_suite import TestSuite
from evidently.tests import (
    TestNumberOfDriftedColumns,
    TestNumberOfMissingValues,
    TestShareOfDriftedColumns,
)

logger = logging.getLogger(__name__)

app = typer.Typer(add_completion=False, no_args_is_help=True)

# Class labels for datasets
CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

STL10_CLASSES = ["airplane", "bird", "car", "cat", "deer", "dog", "horse", "monkey", "ship", "truck"]


def extract_image_features(image: Image.Image | torch.Tensor | np.ndarray) -> dict:
    """Extract structured features from an image for drift detection.

    Features extracted:
    - brightness: Mean pixel value (0-255 scale)
    - contrast: Standard deviation of pixel values
    - red_mean, green_mean, blue_mean: Per-channel mean values
    - red_std, green_std, blue_std: Per-channel standard deviations
    - aspect_ratio: Width / Height ratio
    - sharpness: Estimated via Laplacian variance

    Args:
        image: PIL Image, torch Tensor, or numpy array

    Returns:
        Dictionary of extracted features
    """
    # Convert to numpy array if needed
    if isinstance(image, torch.Tensor):
        # Assume shape (C, H, W) or (H, W, C)
        img_array = image.numpy()
        if img_array.ndim == 3 and img_array.shape[0] in (1, 3):
            img_array = np.transpose(img_array, (1, 2, 0))
        # Denormalize if values are in [-1, 1] or [0, 1] range
        if img_array.max() <= 1.0:
            img_array = (img_array * 255).astype(np.uint8)
    elif isinstance(image, Image.Image):
        img_array = np.array(image)
    else:
        img_array = np.array(image)

    # Ensure RGB format
    if img_array.ndim == 2:
        img_array = np.stack([img_array] * 3, axis=-1)
    elif img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]

    # Ensure uint8 range
    if img_array.max() <= 1.0:
        img_array = (img_array * 255).astype(np.uint8)

    height, width = img_array.shape[:2]

    # Calculate features
    features = {}

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
    # Higher values indicate sharper images
    laplacian_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    try:
        from scipy import ndimage

        laplacian = ndimage.convolve(gray.astype(np.float32), laplacian_kernel)
        features["sharpness"] = float(np.var(laplacian))
    except ImportError:
        features["sharpness"] = 0.0

    # Color saturation (using HSV-like measure)
    max_rgb = np.max(img_array, axis=-1)
    min_rgb = np.min(img_array, axis=-1)
    saturation = np.where(max_rgb > 0, (max_rgb - min_rgb) / (max_rgb + 1e-8), 0)
    features["saturation_mean"] = float(np.mean(saturation))
    features["saturation_std"] = float(np.std(saturation))

    return features


def extract_features_from_dataset(
    dataset: torch.utils.data.Dataset,
    max_samples: int | None = None,
    include_labels: bool = True,
) -> pd.DataFrame:
    """Extract features from all images in a dataset.

    Args:
        dataset: PyTorch dataset with (image, label) items
        max_samples: Maximum number of samples to process (None = all)
        include_labels: Whether to include target labels

    Returns:
        DataFrame with extracted features for each image
    """
    features_list = []
    n_samples = len(dataset) if max_samples is None else min(max_samples, len(dataset))

    for i in tqdm(range(n_samples), desc="Extracting features"):
        # Get raw image (before normalization)
        if hasattr(dataset, "data"):
            # CIFAR-10 style dataset
            if hasattr(dataset, "targets"):
                label = dataset.targets[i]
            else:
                label = dataset.labels[i] if hasattr(dataset, "labels") else 0
            img_data = dataset.data[i]
            if isinstance(img_data, np.ndarray):
                image = Image.fromarray(img_data)
            else:
                image = img_data
        else:
            # Generic dataset
            item = dataset[i]
            if isinstance(item, tuple):
                image, label = item[0], item[1] if len(item) > 1 else 0
            else:
                image, label = item, 0
            if isinstance(image, torch.Tensor):
                # Convert back to PIL for consistent processing
                if image.shape[0] in (1, 3):
                    image = image.permute(1, 2, 0)
                image = Image.fromarray((image.numpy() * 255).astype(np.uint8))

        # Extract features
        features = extract_image_features(image)
        features["sample_id"] = i

        if include_labels:
            features["target"] = int(label)

        features_list.append(features)

    return pd.DataFrame(features_list)


def extract_features_from_predictions(
    prediction_db_path: Path,
    images_dir: Path | None = None,
) -> pd.DataFrame:
    """Load features from a prediction database CSV file.

    If images_dir is provided, re-extracts features from saved images.
    Otherwise, assumes the CSV already contains feature columns.

    Args:
        prediction_db_path: Path to prediction database CSV
        images_dir: Optional directory containing saved images

    Returns:
        DataFrame with features
    """
    if not prediction_db_path.exists():
        raise FileNotFoundError(f"Prediction database not found: {prediction_db_path}")

    df = pd.read_csv(prediction_db_path)

    # Check if features already exist
    feature_cols = ["brightness", "contrast", "red_mean", "green_mean", "blue_mean"]
    if all(col in df.columns for col in feature_cols):
        return df

    # If no features, we need image data to extract them
    if images_dir is None:
        logger.warning("No image features in database and no images_dir provided. Using available columns.")
        return df

    # Re-extract features from saved images
    features_list = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting features from images"):
        if "image_hash" in row:
            img_path = images_dir / f"{row['image_hash']}.png"
            if img_path.exists():
                image = Image.open(img_path)
                features = extract_image_features(image)
                features["target"] = row.get("predicted_class", row.get("prediction", 0))
                features_list.append(features)

    return pd.DataFrame(features_list)


def create_drift_report(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    output_path: Path | None = None,
    include_quality: bool = True,
    include_target_drift: bool = True,
) -> Report:
    """Create an Evidently drift detection report.

    Args:
        reference_data: Reference (training) data features
        current_data: Current (production) data features
        output_path: Optional path to save HTML report
        include_quality: Include data quality metrics
        include_target_drift: Include target/prediction drift analysis

    Returns:
        Evidently Report object
    """
    # Define metrics to include
    metrics = [DataDriftPreset()]

    if include_quality:
        metrics.append(DataQualityPreset())

    if include_target_drift and "target" in reference_data.columns and "target" in current_data.columns:
        metrics.append(TargetDriftPreset())

    # Create column mapping
    column_mapping = ColumnMapping()
    if "target" in reference_data.columns:
        column_mapping.target = "target"

    # Exclude non-feature columns
    exclude_cols = {"sample_id", "timestamp", "image_hash", "time"}
    feature_cols = [c for c in reference_data.columns if c not in exclude_cols and c != "target"]
    column_mapping.numerical_features = feature_cols

    # Create and run report
    report = Report(metrics=metrics)
    report.run(
        reference_data=reference_data,
        current_data=current_data,
        column_mapping=column_mapping,
    )

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        report.save_html(str(output_path))
        logger.info(f"Drift report saved to {output_path}")

    return report


def create_drift_test_suite(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    drift_threshold: float = 0.3,
) -> TestSuite:
    """Create an Evidently test suite for drift detection.

    Args:
        reference_data: Reference (training) data features
        current_data: Current (production) data features
        drift_threshold: Threshold for share of drifted columns (0.0-1.0)

    Returns:
        Evidently TestSuite object with test results
    """
    # Define column mapping
    column_mapping = ColumnMapping()
    if "target" in reference_data.columns:
        column_mapping.target = "target"

    exclude_cols = {"sample_id", "timestamp", "image_hash", "time"}
    feature_cols = [c for c in reference_data.columns if c not in exclude_cols and c != "target"]
    column_mapping.numerical_features = feature_cols

    # Create test suite
    test_suite = TestSuite(
        tests=[
            TestNumberOfMissingValues(),
            TestShareOfDriftedColumns(lt=drift_threshold),
            TestNumberOfDriftedColumns(),
        ]
    )

    test_suite.run(
        reference_data=reference_data,
        current_data=current_data,
        column_mapping=column_mapping,
    )

    return test_suite


def load_reference_dataset(
    dataset_name: str = "cifar10",
    data_dir: str = "./data/raw",
    split: str = "test",
    max_samples: int | None = None,
) -> pd.DataFrame:
    """Load and extract features from a reference dataset.

    Args:
        dataset_name: Dataset name ('cifar10' or 'stl10')
        data_dir: Directory containing dataset
        split: Dataset split to use ('train' or 'test')
        max_samples: Maximum samples to process

    Returns:
        DataFrame with extracted features
    """
    data_path = Path(data_dir)

    if dataset_name.lower() == "cifar10":
        dataset = datasets.CIFAR10(
            data_path,
            train=(split == "train"),
            download=True,
        )
    elif dataset_name.lower() == "stl10":
        dataset = datasets.STL10(
            data_path,
            split=split,
            download=True,
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return extract_features_from_dataset(dataset, max_samples=max_samples)


# CLI Commands
@app.command()
def check(
    dataset: Annotated[str, typer.Option("--dataset", "-d", help="Reference dataset (cifar10 or stl10)")] = "cifar10",
    data_dir: Annotated[Path, typer.Option("--data-dir", help="Dataset directory")] = Path("./data/raw"),
    prediction_db: Annotated[
        Path, typer.Option("--prediction-db", "-p", help="Path to prediction database CSV")
    ] = Path("./api_logs/prediction_database.csv"),
    output: Annotated[Path, typer.Option("--output", "-o", help="Output path for HTML report")] = Path(
        "./reports/drift_report.html"
    ),
    max_samples: Annotated[int, typer.Option("--max-samples", "-n", help="Max samples from reference dataset")] = 1000,
    drift_threshold: Annotated[
        float, typer.Option("--threshold", "-t", help="Drift threshold for tests (0.0-1.0)")
    ] = 0.3,
) -> None:
    """Check for data drift between reference dataset and production predictions."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    logger.info(f"Loading reference dataset: {dataset}")
    reference_data = load_reference_dataset(
        dataset_name=dataset,
        data_dir=str(data_dir),
        split="test",
        max_samples=max_samples,
    )
    logger.info(f"Reference data shape: {reference_data.shape}")

    # Load current/production data
    if not prediction_db.exists():
        logger.error(f"Prediction database not found: {prediction_db}")
        logger.info("Run the API and make some predictions first to generate the database.")
        raise typer.Exit(1)

    logger.info(f"Loading prediction database: {prediction_db}")
    current_data = pd.read_csv(prediction_db)

    # Check if we need to add target column
    if "target" not in current_data.columns and "predicted_class" in current_data.columns:
        current_data["target"] = current_data["predicted_class"]

    logger.info(f"Current data shape: {current_data.shape}")

    # Check for required feature columns
    required_features = ["brightness", "contrast", "red_mean"]
    if not all(f in current_data.columns for f in required_features):
        logger.error("Prediction database missing required feature columns.")
        logger.info("Ensure the API is configured to save image features with predictions.")
        raise typer.Exit(1)

    # Align columns between datasets
    common_cols = list(set(reference_data.columns) & set(current_data.columns))
    if "target" in reference_data.columns and "target" in current_data.columns:
        if "target" not in common_cols:
            common_cols.append("target")

    reference_data = reference_data[common_cols]
    current_data = current_data[common_cols]

    # Create drift report
    logger.info("Generating drift report...")
    create_drift_report(
        reference_data=reference_data,
        current_data=current_data,
        output_path=output,
    )

    # Run test suite
    logger.info("Running drift tests...")
    test_suite = create_drift_test_suite(
        reference_data=reference_data,
        current_data=current_data,
        drift_threshold=drift_threshold,
    )

    # Print results
    results = test_suite.as_dict()
    all_passed = results.get("summary", {}).get("all_passed", False)

    print("\n" + "=" * 50)
    print("DRIFT TEST RESULTS")
    print("=" * 50)

    for test in results.get("tests", []):
        status = "PASS" if test.get("status") == "SUCCESS" else "FAIL"
        print(f"  [{status}] {test.get('name', 'Unknown test')}")

    print("=" * 50)
    if all_passed:
        print("All drift tests PASSED")
    else:
        print("Some drift tests FAILED - review the report for details")
    print(f"\nFull report saved to: {output}")


@app.command()
def generate_reference(
    dataset: Annotated[str, typer.Option("--dataset", "-d", help="Dataset (cifar10 or stl10)")] = "cifar10",
    data_dir: Annotated[Path, typer.Option("--data-dir", help="Dataset directory")] = Path("./data/raw"),
    output: Annotated[Path, typer.Option("--output", "-o", help="Output path for reference features CSV")] = Path(
        "./data/reference_features.csv"
    ),
    max_samples: Annotated[int | None, typer.Option("--max-samples", "-n", help="Max samples (None = all)")] = None,
    split: Annotated[str, typer.Option("--split", "-s", help="Dataset split (train or test)")] = "test",
) -> None:
    """Generate and save reference features from a dataset."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    logger.info(f"Loading {dataset} dataset ({split} split)...")
    features_df = load_reference_dataset(
        dataset_name=dataset,
        data_dir=str(data_dir),
        split=split,
        max_samples=max_samples,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(output, index=False)
    logger.info(f"Reference features saved to: {output}")
    logger.info(f"Shape: {features_df.shape}")
    print(f"\nFeature columns: {list(features_df.columns)}")


@app.command()
def simulate_drift(
    dataset: Annotated[str, typer.Option("--dataset", "-d", help="Dataset (cifar10 or stl10)")] = "cifar10",
    data_dir: Annotated[Path, typer.Option("--data-dir", help="Dataset directory")] = Path("./data/raw"),
    output: Annotated[Path, typer.Option("--output", "-o", help="Output path for HTML report")] = Path(
        "./reports/simulated_drift_report.html"
    ),
    drift_type: Annotated[
        str, typer.Option("--drift-type", "-t", help="Type of drift (brightness, noise, blur)")
    ] = "brightness",
    drift_amount: Annotated[float, typer.Option("--amount", "-a", help="Amount of drift to apply")] = 50.0,
    n_samples: Annotated[int, typer.Option("--n-samples", "-n", help="Number of samples")] = 500,
) -> None:
    """Simulate data drift and generate a report to test drift detection.

    This is useful for testing whether the drift detection pipeline works correctly.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    logger.info(f"Loading {dataset} dataset...")

    # Load reference data
    reference_df = load_reference_dataset(
        dataset_name=dataset,
        data_dir=str(data_dir),
        split="test",
        max_samples=n_samples,
    )

    # Create drifted data by modifying features
    drifted_df = reference_df.copy()

    if drift_type == "brightness":
        logger.info(f"Applying brightness drift: +{drift_amount}")
        drifted_df["brightness"] = drifted_df["brightness"] + drift_amount
        drifted_df["red_mean"] = drifted_df["red_mean"] + drift_amount * 0.8
        drifted_df["green_mean"] = drifted_df["green_mean"] + drift_amount * 0.8
        drifted_df["blue_mean"] = drifted_df["blue_mean"] + drift_amount * 0.8
    elif drift_type == "noise":
        logger.info(f"Applying noise drift: contrast +{drift_amount}")
        drifted_df["contrast"] = drifted_df["contrast"] + drift_amount
        drifted_df["sharpness"] = drifted_df["sharpness"] * 0.5
    elif drift_type == "blur":
        logger.info(f"Applying blur drift: sharpness -{drift_amount}%")
        drifted_df["sharpness"] = drifted_df["sharpness"] * (1 - drift_amount / 100)
        drifted_df["contrast"] = drifted_df["contrast"] * 0.9
    else:
        logger.warning(f"Unknown drift type: {drift_type}, using random noise")
        for col in ["brightness", "contrast", "red_mean", "green_mean", "blue_mean"]:
            drifted_df[col] = drifted_df[col] + np.random.normal(0, drift_amount, len(drifted_df))

    # Generate report
    logger.info("Generating drift report...")
    create_drift_report(
        reference_data=reference_df,
        current_data=drifted_df,
        output_path=output,
    )

    # Run tests
    test_suite = create_drift_test_suite(reference_df, drifted_df)
    results = test_suite.as_dict()

    print("\n" + "=" * 50)
    print(f"SIMULATED {drift_type.upper()} DRIFT TEST")
    print("=" * 50)

    for test in results.get("tests", []):
        status = "PASS" if test.get("status") == "SUCCESS" else "FAIL"
        print(f"  [{status}] {test.get('name', 'Unknown test')}")

    print(f"\nReport saved to: {output}")


def main():
    """Entry point for the rep-geom-drift command."""
    app()


if __name__ == "__main__":
    main()
