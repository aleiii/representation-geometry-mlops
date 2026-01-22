#!/usr/bin/env python
"""Standalone script for checking data drift in production predictions.

This script provides a convenient way to run drift detection analysis
comparing reference dataset features against production prediction data.

Usage Examples:
    # Check drift using default settings (CIFAR-10 reference vs prediction_database.csv)
    python scripts/check_drift.py

    # Check drift with STL-10 as reference
    python scripts/check_drift.py --dataset stl10

    # Specify custom paths
    python scripts/check_drift.py \
        --prediction-db ./api_logs/predictions.csv \
        --output ./reports/my_drift_report.html

    # Generate reference features only (save for later comparison)
    python scripts/check_drift.py --generate-reference --output ./data/cifar10_features.csv

    # Simulate drift to test the detection pipeline
    python scripts/check_drift.py --simulate --drift-type brightness --amount 50
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add src to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from representation_geometry.drift import (
    create_drift_report,
    create_drift_test_suite,
    load_reference_dataset,
)
import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def check_drift(
    dataset: str,
    data_dir: Path,
    prediction_db: Path,
    output: Path,
    max_samples: int,
    drift_threshold: float,
) -> bool:
    """Run drift detection analysis.

    Args:
        dataset: Reference dataset name (cifar10 or stl10)
        data_dir: Directory containing the dataset
        prediction_db: Path to prediction database CSV
        output: Output path for HTML report
        max_samples: Maximum samples from reference dataset
        drift_threshold: Threshold for drift test failure

    Returns:
        True if all tests passed, False otherwise
    """
    logger.info(f"Loading reference dataset: {dataset}")
    reference_data = load_reference_dataset(
        dataset_name=dataset,
        data_dir=str(data_dir),
        split="test",
        max_samples=max_samples,
    )
    logger.info(f"Reference data: {len(reference_data)} samples, {len(reference_data.columns)} features")

    # Load current/production data
    if not prediction_db.exists():
        logger.error(f"Prediction database not found: {prediction_db}")
        logger.info("To generate sample data, use: python scripts/check_drift.py --simulate")
        return False

    logger.info(f"Loading prediction database: {prediction_db}")
    current_data = pd.read_csv(prediction_db)

    # Ensure target column exists
    if "target" not in current_data.columns:
        if "predicted_class" in current_data.columns:
            current_data["target"] = current_data["predicted_class"]
        elif "prediction" in current_data.columns:
            current_data["target"] = current_data["prediction"]

    logger.info(f"Current data: {len(current_data)} samples")

    # Check for required feature columns
    required_features = ["brightness", "contrast", "red_mean"]
    missing_features = [f for f in required_features if f not in current_data.columns]

    if missing_features:
        logger.error(f"Missing required feature columns: {missing_features}")
        logger.info("The prediction database must contain image features.")
        logger.info("Make sure the API is saving features with each prediction.")
        return False

    # Align columns
    common_cols = list(set(reference_data.columns) & set(current_data.columns))
    logger.info(f"Common feature columns: {len(common_cols)}")

    reference_aligned = reference_data[common_cols]
    current_aligned = current_data[common_cols]

    # Generate drift report
    logger.info("Generating drift report...")
    create_drift_report(
        reference_data=reference_aligned,
        current_data=current_aligned,
        output_path=output,
    )

    # Run test suite
    logger.info("Running drift tests...")
    test_suite = create_drift_test_suite(
        reference_data=reference_aligned,
        current_data=current_aligned,
        drift_threshold=drift_threshold,
    )

    # Display results
    results = test_suite.as_dict()
    all_passed = results.get("summary", {}).get("all_passed", False)

    print("\n" + "=" * 60)
    print("DATA DRIFT TEST RESULTS")
    print("=" * 60)

    for test in results.get("tests", []):
        status = "PASS" if test.get("status") == "SUCCESS" else "FAIL"
        name = test.get("name", "Unknown test")
        print(f"  [{status}] {name}")

    print("=" * 60)

    if all_passed:
        print("Result: ALL TESTS PASSED - No significant drift detected")
    else:
        print("Result: SOME TESTS FAILED - Drift detected, review report for details")

    print(f"\nDetailed report saved to: {output}")
    print(f"Open in browser: file://{output.absolute()}")

    return all_passed


def generate_reference(
    dataset: str,
    data_dir: Path,
    output: Path,
    max_samples: int | None,
    split: str,
) -> None:
    """Generate and save reference features from a dataset."""
    logger.info(f"Loading {dataset} dataset ({split} split)...")

    features_df = load_reference_dataset(
        dataset_name=dataset,
        data_dir=str(data_dir),
        split=split,
        max_samples=max_samples,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(output, index=False)

    print(f"\nReference features saved to: {output}")
    print(f"Shape: {features_df.shape}")
    print(f"Columns: {list(features_df.columns)}")
    print("\nFeature statistics:")
    print(features_df.describe().round(2).to_string())


def simulate_drift(
    dataset: str,
    data_dir: Path,
    output: Path,
    drift_type: str,
    drift_amount: float,
    n_samples: int,
) -> None:
    """Simulate data drift and generate a test report."""
    logger.info(f"Loading {dataset} dataset for drift simulation...")

    reference_df = load_reference_dataset(
        dataset_name=dataset,
        data_dir=str(data_dir),
        split="test",
        max_samples=n_samples,
    )

    # Create drifted data
    drifted_df = reference_df.copy()

    print(f"\nSimulating {drift_type} drift with amount={drift_amount}")

    if drift_type == "brightness":
        drifted_df["brightness"] += drift_amount
        drifted_df["red_mean"] += drift_amount * 0.8
        drifted_df["green_mean"] += drift_amount * 0.8
        drifted_df["blue_mean"] += drift_amount * 0.8
    elif drift_type == "contrast":
        drifted_df["contrast"] += drift_amount
        drifted_df["red_std"] += drift_amount * 0.5
        drifted_df["green_std"] += drift_amount * 0.5
        drifted_df["blue_std"] += drift_amount * 0.5
    elif drift_type == "blur":
        drifted_df["sharpness"] *= 1 - drift_amount / 100
        drifted_df["contrast"] *= 0.9
    elif drift_type == "saturation":
        drifted_df["saturation_mean"] *= 1 + drift_amount / 100
    else:
        # Random noise on all features
        for col in ["brightness", "contrast", "red_mean", "green_mean", "blue_mean"]:
            if col in drifted_df.columns:
                drifted_df[col] += np.random.normal(0, drift_amount, len(drifted_df))

    # Generate report
    logger.info("Generating simulated drift report...")
    create_drift_report(
        reference_data=reference_df,
        current_data=drifted_df,
        output_path=output,
    )

    # Run tests
    test_suite = create_drift_test_suite(reference_df, drifted_df)
    results = test_suite.as_dict()

    print("\n" + "=" * 60)
    print(f"SIMULATED {drift_type.upper()} DRIFT TEST")
    print("=" * 60)

    for test in results.get("tests", []):
        status = "PASS" if test.get("status") == "SUCCESS" else "FAIL"
        print(f"  [{status}] {test.get('name', 'Unknown')}")

    print("=" * 60)
    print(f"\nReport saved to: {output}")
    print(f"Open in browser: file://{output.absolute()}")


def main():
    parser = argparse.ArgumentParser(
        description="Check data drift between reference dataset and production predictions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic drift check
  python scripts/check_drift.py

  # Check with STL-10 reference
  python scripts/check_drift.py --dataset stl10

  # Generate reference features
  python scripts/check_drift.py --generate-reference -o data/reference.csv

  # Simulate brightness drift
  python scripts/check_drift.py --simulate --drift-type brightness --amount 50

Available drift types for simulation:
  - brightness: Increases overall image brightness
  - contrast: Increases image contrast/variance
  - blur: Reduces image sharpness
  - saturation: Increases color saturation
        """,
    )

    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--generate-reference",
        action="store_true",
        help="Generate reference features CSV instead of checking drift",
    )
    mode_group.add_argument(
        "--simulate",
        action="store_true",
        help="Simulate drift to test the detection pipeline",
    )

    # Common options
    parser.add_argument(
        "--dataset",
        "-d",
        default="cifar10",
        choices=["cifar10", "stl10"],
        help="Reference dataset (default: cifar10)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("./data/raw"),
        help="Dataset directory (default: ./data/raw)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output path for report/features",
    )
    parser.add_argument(
        "--max-samples",
        "-n",
        type=int,
        default=1000,
        help="Max samples from reference dataset (default: 1000)",
    )

    # Drift check options
    parser.add_argument(
        "--prediction-db",
        "-p",
        type=Path,
        default=Path("./api_logs/prediction_database.csv"),
        help="Path to prediction database CSV",
    )
    parser.add_argument(
        "--threshold",
        "-t",
        type=float,
        default=0.3,
        help="Drift threshold for test failure (default: 0.3)",
    )

    # Simulation options
    parser.add_argument(
        "--drift-type",
        default="brightness",
        choices=["brightness", "contrast", "blur", "saturation", "random"],
        help="Type of drift to simulate (default: brightness)",
    )
    parser.add_argument(
        "--amount",
        "-a",
        type=float,
        default=50.0,
        help="Amount of drift to apply (default: 50.0)",
    )

    # Reference generation options
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "test"],
        help="Dataset split to use for reference (default: test)",
    )

    args = parser.parse_args()

    # Set default output paths based on mode
    if args.output is None:
        if args.generate_reference:
            args.output = Path(f"./data/{args.dataset}_reference_features.csv")
        elif args.simulate:
            args.output = Path("./reports/simulated_drift_report.html")
        else:
            args.output = Path("./reports/drift_report.html")

    # Run appropriate mode
    if args.generate_reference:
        generate_reference(
            dataset=args.dataset,
            data_dir=args.data_dir,
            output=args.output,
            max_samples=args.max_samples if args.max_samples > 0 else None,
            split=args.split,
        )
    elif args.simulate:
        simulate_drift(
            dataset=args.dataset,
            data_dir=args.data_dir,
            output=args.output,
            drift_type=args.drift_type,
            drift_amount=args.amount,
            n_samples=args.max_samples,
        )
    else:
        success = check_drift(
            dataset=args.dataset,
            data_dir=args.data_dir,
            prediction_db=args.prediction_db,
            output=args.output,
            max_samples=args.max_samples,
            drift_threshold=args.threshold,
        )
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
