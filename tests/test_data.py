"""Tests for data modules."""

from torch.utils.data import DataLoader

from representation_geometry.data import CIFAR10DataModule, STL10DataModule


class TestCIFAR10DataModule:
    """Test cases for CIFAR10DataModule."""

    def test_cifar10_initialization(self):
        """Test that CIFAR10DataModule initializes correctly."""
        dm = CIFAR10DataModule(batch_size=32, num_workers=0)
        assert dm.batch_size == 32
        assert dm.num_workers == 0
        assert dm.train_dataset is None
        assert dm.val_dataset is None
        assert dm.test_dataset is None

    def test_cifar10_prepare_data(self):
        """Test data preparation (download)."""
        dm = CIFAR10DataModule(batch_size=32, num_workers=0)
        # This should download the dataset if not present
        dm.prepare_data()

    def test_cifar10_setup_fit(self):
        """Test setup for training."""
        dm = CIFAR10DataModule(batch_size=32, num_workers=0)
        dm.prepare_data()
        dm.setup("fit")

        assert dm.train_dataset is not None
        assert dm.val_dataset is not None
        assert len(dm.train_dataset) > 0
        assert len(dm.val_dataset) > 0

    def test_cifar10_setup_test(self):
        """Test setup for testing."""
        dm = CIFAR10DataModule(batch_size=32, num_workers=0)
        dm.prepare_data()
        dm.setup("test")

        assert dm.test_dataset is not None
        assert len(dm.test_dataset) == 10000  # CIFAR-10 test set size

    def test_cifar10_train_dataloader(self):
        """Test train dataloader creation."""
        dm = CIFAR10DataModule(batch_size=32, num_workers=0)
        dm.prepare_data()
        dm.setup("fit")

        train_loader = dm.train_dataloader()
        assert isinstance(train_loader, DataLoader)

        # Test one batch
        batch = next(iter(train_loader))
        images, labels = batch
        assert images.shape[0] <= 32  # batch size
        assert images.shape[1:] == (3, 32, 32)  # CIFAR-10 image shape
        assert labels.shape[0] <= 32

    def test_cifar10_val_dataloader(self):
        """Test validation dataloader creation."""
        dm = CIFAR10DataModule(batch_size=32, num_workers=0)
        dm.prepare_data()
        dm.setup("fit")

        val_loader = dm.val_dataloader()
        assert isinstance(val_loader, DataLoader)

        # Test one batch
        batch = next(iter(val_loader))
        images, labels = batch
        assert images.shape[1:] == (3, 32, 32)

    def test_cifar10_test_dataloader(self):
        """Test test dataloader creation."""
        dm = CIFAR10DataModule(batch_size=32, num_workers=0)
        dm.prepare_data()
        dm.setup("test")

        test_loader = dm.test_dataloader()
        assert isinstance(test_loader, DataLoader)

        # Test one batch
        batch = next(iter(test_loader))
        images, labels = batch
        assert images.shape[1:] == (3, 32, 32)


class TestSTL10DataModule:
    """Test cases for STL10DataModule."""

    def test_stl10_initialization(self):
        """Test that STL10DataModule initializes correctly."""
        dm = STL10DataModule(batch_size=16, num_workers=0, resize_to=96)
        assert dm.batch_size == 16
        assert dm.num_workers == 0
        assert dm.resize_to == 96
        assert dm.train_dataset is None

    def test_stl10_prepare_data(self):
        """Test data preparation (download)."""
        dm = STL10DataModule(batch_size=16, num_workers=0)
        # This should download the dataset if not present
        dm.prepare_data()

    def test_stl10_setup_fit(self):
        """Test setup for training."""
        dm = STL10DataModule(batch_size=16, num_workers=0)
        dm.prepare_data()
        dm.setup("fit")

        assert dm.train_dataset is not None
        assert dm.val_dataset is not None
        assert len(dm.train_dataset) == 5000  # STL-10 train set size
        assert len(dm.val_dataset) == 8000  # STL-10 test set (used as val)

    def test_stl10_setup_test(self):
        """Test setup for testing."""
        dm = STL10DataModule(batch_size=16, num_workers=0)
        dm.prepare_data()
        dm.setup("test")

        assert dm.test_dataset is not None
        assert len(dm.test_dataset) == 8000  # STL-10 test set size

    def test_stl10_train_dataloader(self):
        """Test train dataloader creation."""
        dm = STL10DataModule(batch_size=16, num_workers=0, resize_to=96)
        dm.prepare_data()
        dm.setup("fit")

        train_loader = dm.train_dataloader()
        assert isinstance(train_loader, DataLoader)

        # Test one batch
        batch = next(iter(train_loader))
        images, labels = batch
        assert images.shape[0] <= 16  # batch size
        assert images.shape[1:] == (3, 96, 96)  # Resized STL-10 image shape
        assert labels.shape[0] <= 16

    def test_stl10_image_resize(self):
        """Test that images are properly resized."""
        dm = STL10DataModule(batch_size=16, num_workers=0, resize_to=64)
        dm.prepare_data()
        dm.setup("fit")

        train_loader = dm.train_dataloader()
        batch = next(iter(train_loader))
        images, _ = batch
        assert images.shape[2:] == (64, 64)  # Should be resized to 64x64
