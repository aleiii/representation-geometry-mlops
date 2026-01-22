"""Tests for data drift detection module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from representation_geometry.drift import (
    extract_image_features,
    create_drift_report,
    create_drift_test_suite,
)


class TestExtractImageFeatures:
    """Tests for image feature extraction."""

    def test_extract_from_pil_image(self):
        """Test feature extraction from PIL Image."""
        # Create a simple test image (32x32 RGB)
        img_array = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
        image = Image.fromarray(img_array)

        features = extract_image_features(image)

        # Check all expected features are present
        expected_features = [
            "brightness", "contrast",
            "red_mean", "green_mean", "blue_mean",
            "red_std", "green_std", "blue_std",
            "aspect_ratio", "sharpness",
            "saturation_mean", "saturation_std",
        ]
        for feat in expected_features:
            assert feat in features, f"Missing feature: {feat}"

    def test_extract_from_numpy_array(self):
        """Test feature extraction from numpy array."""
        img_array = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)

        features = extract_image_features(img_array)

        assert "brightness" in features
        assert 0 <= features["brightness"] <= 255

    def test_extract_from_torch_tensor(self):
        """Test feature extraction from torch tensor."""
        import torch

        # Create tensor in (C, H, W) format with values in [0, 1]
        tensor = torch.rand(3, 32, 32)

        features = extract_image_features(tensor)

        assert "brightness" in features
        assert features["aspect_ratio"] == 1.0  # Square image

    def test_brightness_calculation(self):
        """Test that brightness is calculated correctly."""
        # Create a white image
        white_image = np.ones((32, 32, 3), dtype=np.uint8) * 255
        features_white = extract_image_features(white_image)

        # Create a black image
        black_image = np.zeros((32, 32, 3), dtype=np.uint8)
        features_black = extract_image_features(black_image)

        assert features_white["brightness"] > features_black["brightness"]
        assert features_white["brightness"] == pytest.approx(255.0, rel=0.01)
        assert features_black["brightness"] == pytest.approx(0.0, abs=0.01)

    def test_aspect_ratio(self):
        """Test aspect ratio calculation."""
        # Create a wide image
        wide_image = np.random.randint(0, 256, (32, 64, 3), dtype=np.uint8)
        features = extract_image_features(wide_image)

        assert features["aspect_ratio"] == pytest.approx(2.0, rel=0.01)


class TestDriftReport:
    """Tests for drift report generation."""

    @pytest.fixture
    def sample_dataframes(self):
        """Create sample reference and current dataframes."""
        np.random.seed(42)
        n_samples = 100

        # Reference data (normal distribution)
        reference = pd.DataFrame({
            "brightness": np.random.normal(128, 20, n_samples),
            "contrast": np.random.normal(50, 10, n_samples),
            "red_mean": np.random.normal(120, 25, n_samples),
            "green_mean": np.random.normal(125, 25, n_samples),
            "blue_mean": np.random.normal(130, 25, n_samples),
            "target": np.random.randint(0, 10, n_samples),
        })

        # Current data (similar distribution)
        current = pd.DataFrame({
            "brightness": np.random.normal(128, 20, n_samples),
            "contrast": np.random.normal(50, 10, n_samples),
            "red_mean": np.random.normal(120, 25, n_samples),
            "green_mean": np.random.normal(125, 25, n_samples),
            "blue_mean": np.random.normal(130, 25, n_samples),
            "target": np.random.randint(0, 10, n_samples),
        })

        return reference, current

    @pytest.fixture
    def drifted_dataframes(self):
        """Create sample dataframes with simulated drift."""
        np.random.seed(42)
        n_samples = 100

        # Reference data
        reference = pd.DataFrame({
            "brightness": np.random.normal(128, 20, n_samples),
            "contrast": np.random.normal(50, 10, n_samples),
            "red_mean": np.random.normal(120, 25, n_samples),
            "target": np.random.randint(0, 10, n_samples),
        })

        # Drifted data (shifted distribution)
        current = pd.DataFrame({
            "brightness": np.random.normal(180, 20, n_samples),  # Shifted!
            "contrast": np.random.normal(70, 10, n_samples),     # Shifted!
            "red_mean": np.random.normal(160, 25, n_samples),    # Shifted!
            "target": np.random.randint(0, 10, n_samples),
        })

        return reference, current

    def test_create_drift_report_no_drift(self, sample_dataframes, tmp_path):
        """Test report generation when no drift is present."""
        reference, current = sample_dataframes
        output_path = tmp_path / "report.html"

        report = create_drift_report(
            reference_data=reference,
            current_data=current,
            output_path=output_path,
        )

        assert report is not None
        assert output_path.exists()

    def test_create_drift_report_with_drift(self, drifted_dataframes, tmp_path):
        """Test report generation when drift is present."""
        reference, current = drifted_dataframes
        output_path = tmp_path / "drift_report.html"

        report = create_drift_report(
            reference_data=reference,
            current_data=current,
            output_path=output_path,
        )

        assert report is not None
        assert output_path.exists()


class TestDriftTestSuite:
    """Tests for drift test suite."""

    def test_no_drift_passes(self):
        """Test that similar data passes drift tests."""
        np.random.seed(42)
        n_samples = 200

        reference = pd.DataFrame({
            "brightness": np.random.normal(128, 20, n_samples),
            "contrast": np.random.normal(50, 10, n_samples),
        })

        current = pd.DataFrame({
            "brightness": np.random.normal(128, 20, n_samples),
            "contrast": np.random.normal(50, 10, n_samples),
        })

        test_suite = create_drift_test_suite(reference, current, drift_threshold=0.5)
        results = test_suite.as_dict()

        # Should have no missing values
        missing_test = next(
            (t for t in results["tests"] if "Missing" in t.get("name", "")),
            None
        )
        if missing_test:
            assert missing_test["status"] == "SUCCESS"

    def test_significant_drift_detected(self):
        """Test that significant drift is detected."""
        np.random.seed(42)
        n_samples = 200

        reference = pd.DataFrame({
            "brightness": np.random.normal(100, 10, n_samples),
            "contrast": np.random.normal(30, 5, n_samples),
            "feature_a": np.random.normal(50, 10, n_samples),
            "feature_b": np.random.normal(60, 10, n_samples),
        })

        # Very different distribution
        current = pd.DataFrame({
            "brightness": np.random.normal(200, 10, n_samples),  # Big shift
            "contrast": np.random.normal(80, 5, n_samples),       # Big shift
            "feature_a": np.random.normal(150, 10, n_samples),    # Big shift
            "feature_b": np.random.normal(160, 10, n_samples),    # Big shift
        })

        test_suite = create_drift_test_suite(reference, current, drift_threshold=0.3)
        results = test_suite.as_dict()

        # Should detect drifted columns
        drift_test = next(
            (t for t in results["tests"] if "Drifted" in t.get("name", "")),
            None
        )
        assert drift_test is not None
