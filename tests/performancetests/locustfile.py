"""Locust load testing script for the Neural Representation Geometry API.

Usage:
    # Start the API server first:
    # uvicorn representation_geometry.api:app --host 0.0.0.0 --port 8000

    # Run Locust with web UI:
    locust -f tests/performancetests/locustfile.py --host http://localhost:8000

    # Run Locust in headless mode:
    locust -f tests/performancetests/locustfile.py --host http://localhost:8000 \
           --headless -u 10 -r 2 -t 60s

Options:
    -u, --users: Number of concurrent users
    -r, --spawn-rate: Rate of spawning users per second
    -t, --run-time: Total run time (e.g., 60s, 5m)
    --host: Base URL of the API server

Environment Variables:
    DATA_DIR: Path to data directory (default: ./data/raw)
"""

from __future__ import annotations

import base64
import io
import logging
import os
import random
from pathlib import Path

from locust import HttpUser, between, task

# Try to import required libraries
try:
    from PIL import Image

    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    from torchvision import datasets

    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False

logger = logging.getLogger(__name__)

# Default data directory (relative to project root)
DEFAULT_DATA_DIR = Path("./data/raw")


class ImagePool:
    """Pool of test images loaded from CIFAR10/STL10 datasets.

    Falls back to generating random images if datasets are not available.
    """

    _cifar10_images: list[str] = []
    _stl10_images: list[str] = []
    _initialized: bool = False

    @classmethod
    def initialize(cls, data_dir: Path | None = None, num_samples: int = 50) -> None:
        """Load test images from datasets.

        Args:
            data_dir: Path to data directory containing datasets
            num_samples: Number of images to preload from each dataset
        """
        if cls._initialized:
            return

        data_dir = data_dir or Path(os.getenv("DATA_DIR", str(DEFAULT_DATA_DIR)))
        data_dir = Path(data_dir)

        # Try to load CIFAR10 images
        cls._cifar10_images = cls._load_cifar10_images(data_dir, num_samples)
        if cls._cifar10_images:
            logger.info(f"Loaded {len(cls._cifar10_images)} CIFAR10 test images")
        else:
            logger.warning("CIFAR10 dataset not available, will generate random images")

        # Try to load STL10 images
        cls._stl10_images = cls._load_stl10_images(data_dir, num_samples)
        if cls._stl10_images:
            logger.info(f"Loaded {len(cls._stl10_images)} STL10 test images")
        else:
            logger.warning("STL10 dataset not available, will generate random images")

        cls._initialized = True

    @classmethod
    def _load_cifar10_images(cls, data_dir: Path, num_samples: int) -> list[str]:
        """Load CIFAR10 test images as base64 strings."""
        if not HAS_TORCHVISION or not HAS_PIL:
            return []

        try:
            # Load CIFAR10 test dataset (without transforms to get raw PIL images)
            dataset = datasets.CIFAR10(root=str(data_dir), train=False, download=False)

            # Sample random indices
            indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
            images = []

            for idx in indices:
                img, _ = dataset[idx]  # img is a PIL Image
                images.append(cls._pil_to_base64(img))

            return images
        except Exception as e:
            logger.debug(f"Failed to load CIFAR10: {e}")
            return []

    @classmethod
    def _load_stl10_images(cls, data_dir: Path, num_samples: int) -> list[str]:
        """Load STL10 test images as base64 strings."""
        if not HAS_TORCHVISION or not HAS_PIL:
            return []

        try:
            # Load STL10 test dataset
            dataset = datasets.STL10(root=str(data_dir), split="test", download=False)

            # Sample random indices
            indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
            images = []

            for idx in indices:
                img, _ = dataset[idx]  # img is a PIL Image
                images.append(cls._pil_to_base64(img))

            return images
        except Exception as e:
            logger.debug(f"Failed to load STL10: {e}")
            return []

    @staticmethod
    def _pil_to_base64(image: Image.Image) -> str:
        """Convert a PIL Image to base64-encoded PNG string."""
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode("utf-8")

    @classmethod
    def _generate_random_image(cls, width: int, height: int) -> str:
        """Generate a random RGB image as base64 string (fallback)."""
        if HAS_PIL:
            pixels = bytes(random.randint(0, 255) for _ in range(width * height * 3))
            image = Image.frombytes("RGB", (width, height), pixels)
            return cls._pil_to_base64(image)
        else:
            # Minimal valid PNG (1x1 red pixel)
            minimal_png = (
                b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
                b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00"
                b"\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00"
                b"\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
            )
            return base64.b64encode(minimal_png).decode("utf-8")

    @classmethod
    def get_cifar10_image(cls) -> str:
        """Get a random CIFAR10 image (32x32), or generate one if unavailable."""
        if not cls._initialized:
            cls.initialize()

        if cls._cifar10_images:
            return random.choice(cls._cifar10_images)
        return cls._generate_random_image(32, 32)

    @classmethod
    def get_stl10_image(cls) -> str:
        """Get a random STL10 image (96x96), or generate one if unavailable."""
        if not cls._initialized:
            cls.initialize()

        if cls._stl10_images:
            return random.choice(cls._stl10_images)
        return cls._generate_random_image(96, 96)

    @classmethod
    def get_random_image(cls, large: bool = False) -> str:
        """Get a random test image.

        Args:
            large: If True, prefer STL10 (96x96), otherwise CIFAR10 (32x32)
        """
        if large:
            return cls.get_stl10_image()
        return cls.get_cifar10_image()


class APIUser(HttpUser):
    """Simulates a user interacting with the Neural Representation Geometry API.

    This user performs a mix of lightweight health checks and heavier
    prediction/representation requests to simulate realistic API usage patterns.
    Uses real images from CIFAR10/STL10 test datasets when available.
    """

    # Wait between 1-3 seconds between tasks
    wait_time = between(1, 3)

    def on_start(self):
        """Called when a simulated user starts.

        Initializes the image pool if not already done.
        """
        ImagePool.initialize()

    @task(10)
    def health_check(self):
        """Test the /health endpoint (high frequency).

        This is typically used for readiness/liveness probes in production.
        """
        with self.client.get("/health", catch_response=True) as response:
            if response.status_code == 200:
                json_response = response.json()
                if json_response.get("status") == "healthy":
                    response.success()
                else:
                    response.failure(f"Unhealthy status: {json_response}")
            else:
                response.failure(f"Status code: {response.status_code}")

    @task(5)
    def root_endpoint(self):
        """Test the root / endpoint (medium frequency).

        Returns API information and available endpoints.
        """
        with self.client.get("/", catch_response=True) as response:
            if response.status_code == 200:
                json_response = response.json()
                if "endpoints" in json_response:
                    response.success()
                else:
                    response.failure("Missing endpoints in response")
            else:
                response.failure(f"Status code: {response.status_code}")

    @task(3)
    def list_models(self):
        """Test the /models endpoint (low-medium frequency).

        Lists all available model checkpoints.
        """
        with self.client.get("/models", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status code: {response.status_code}")

    @task(2)
    def prediction_stats(self):
        """Test the /predictions/stats endpoint (low frequency).

        Returns statistics about logged predictions.
        """
        with self.client.get("/predictions/stats", catch_response=True) as response:
            if response.status_code == 200:
                json_response = response.json()
                if "total_predictions" in json_response:
                    response.success()
                else:
                    response.failure("Missing total_predictions in response")
            else:
                response.failure(f"Status code: {response.status_code}")

    @task(5)
    def predict_cifar10(self):
        """Test /predict with CIFAR10 images (medium frequency).

        Sends a test image from CIFAR10 dataset for classification.
        Note: This requires a model checkpoint to be available.
        """
        payload = {
            "image_base64": ImagePool.get_cifar10_image(),
            "top_k": 5,
            "dataset": "cifar10",
        }

        with self.client.post("/predict", json=payload, catch_response=True) as response:
            if response.status_code == 200:
                json_response = response.json()
                if "predicted_class" in json_response and "confidence" in json_response:
                    response.success()
                else:
                    response.failure("Missing prediction fields in response")
            elif response.status_code == 404:
                # No checkpoint available - mark as expected failure
                response.failure("No checkpoint available (404)")
            else:
                response.failure(f"Status code: {response.status_code}")

    @task(2)
    def predict_stl10(self):
        """Test /predict with STL10 images (low frequency).

        Sends a test image from STL10 dataset for classification.
        """
        payload = {
            "image_base64": ImagePool.get_stl10_image(),
            "top_k": 5,
            "dataset": "stl10",
        }

        with self.client.post("/predict", json=payload, catch_response=True) as response:
            if response.status_code == 200:
                json_response = response.json()
                if "predicted_class" in json_response:
                    response.success()
                else:
                    response.failure("Missing prediction fields in response")
            elif response.status_code == 404:
                response.failure("No checkpoint available (404)")
            else:
                response.failure(f"Status code: {response.status_code}")

    @task(3)
    def representations_cifar10(self):
        """Test /representations with CIFAR10 images (low-medium frequency).

        Extracts intermediate layer activations from the model.
        Note: This requires a model checkpoint to be available.
        """
        payload = {
            "image_base64": ImagePool.get_cifar10_image(),
            "dataset": "cifar10",
        }

        with self.client.post("/representations", json=payload, catch_response=True) as response:
            if response.status_code == 200:
                json_response = response.json()
                if "activations" in json_response:
                    response.success()
                else:
                    response.failure("Missing activations in response")
            elif response.status_code == 404:
                # No checkpoint or activations available
                response.failure("No checkpoint or activations available (404)")
            else:
                response.failure(f"Status code: {response.status_code}")

    @task(1)
    def representations_stl10(self):
        """Test /representations with STL10 images (rare).

        Tests with 96x96 images from STL10 dataset.
        """
        payload = {
            "image_base64": ImagePool.get_stl10_image(),
            "dataset": "stl10",
        }

        with self.client.post("/representations", json=payload, catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            elif response.status_code == 404:
                response.failure("No checkpoint available (404)")
            else:
                response.failure(f"Status code: {response.status_code}")


class LightweightUser(HttpUser):
    """A lightweight user that only tests GET endpoints.

    Use this for stress testing endpoint availability without model inference.
    To use: locust -f locustfile.py --class LightweightUser
    """

    wait_time = between(0.5, 1.5)

    @task(5)
    def health_check(self):
        """Test /health endpoint."""
        self.client.get("/health")

    @task(3)
    def root_endpoint(self):
        """Test / endpoint."""
        self.client.get("/")

    @task(2)
    def list_models(self):
        """Test /models endpoint."""
        self.client.get("/models")

    @task(1)
    def prediction_stats(self):
        """Test /predictions/stats endpoint."""
        self.client.get("/predictions/stats")


class InferenceUser(HttpUser):
    """A user focused on inference endpoints only.

    Use this for load testing the model inference specifically.
    Uses real images from CIFAR10/STL10 test datasets when available.
    To use: locust -f locustfile.py --class InferenceUser
    """

    wait_time = between(0.5, 2)

    def on_start(self):
        """Initialize image pool."""
        ImagePool.initialize()

    @task(3)
    def predict_cifar10(self):
        """Test /predict with CIFAR10 images."""
        payload = {
            "image_base64": ImagePool.get_cifar10_image(),
            "top_k": 5,
            "dataset": "cifar10",
        }
        self.client.post("/predict", json=payload)

    @task(1)
    def predict_stl10(self):
        """Test /predict with STL10 images."""
        payload = {
            "image_base64": ImagePool.get_stl10_image(),
            "top_k": 5,
            "dataset": "stl10",
        }
        self.client.post("/predict", json=payload)

    @task(2)
    def representations_cifar10(self):
        """Test /representations with CIFAR10 images."""
        payload = {
            "image_base64": ImagePool.get_cifar10_image(),
            "dataset": "cifar10",
        }
        self.client.post("/representations", json=payload)

    @task(1)
    def representations_stl10(self):
        """Test /representations with STL10 images."""
        payload = {
            "image_base64": ImagePool.get_stl10_image(),
            "dataset": "stl10",
        }
        self.client.post("/representations", json=payload)
