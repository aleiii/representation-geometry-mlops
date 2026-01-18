"""Tests for model architectures."""
import pytest
import torch

from representation_geometry.model import (
    MLPClassifier,
    ResNet18Classifier,
    extract_representations,
)


class TestMLPClassifier:
    """Test cases for MLPClassifier."""

    def test_mlp_initialization(self):
        """Test MLP initialization with default parameters."""
        model = MLPClassifier()
        assert model.input_size == 3072
        assert model.num_classes == 10
        assert model.hidden_dims == [512, 512]

    def test_mlp_custom_architecture(self):
        """Test MLP with custom architecture."""
        model = MLPClassifier(
            input_size=1024,
            hidden_dims=[256, 128],
            num_classes=5,
        )
        assert model.input_size == 1024
        assert model.num_classes == 5
        assert model.hidden_dims == [256, 128]

    def test_mlp_forward_pass(self):
        """Test forward pass with CIFAR-10 sized input."""
        model = MLPClassifier()
        x = torch.randn(4, 3, 32, 32)  # Batch of 4 CIFAR-10 images
        output = model(x)
        
        assert output.shape == (4, 10)  # Batch size 4, 10 classes

    def test_mlp_forward_pass_different_batch_sizes(self):
        """Test forward pass with different batch sizes."""
        model = MLPClassifier()
        
        for batch_size in [1, 8, 16, 32]:
            x = torch.randn(batch_size, 3, 32, 32)
            output = model(x)
            assert output.shape == (batch_size, 10)

    def test_mlp_gradient_flow(self):
        """Test that gradients flow through the model."""
        model = MLPClassifier()
        x = torch.randn(4, 3, 32, 32)
        labels = torch.randint(0, 10, (4,))
        
        output = model(x)
        loss = torch.nn.functional.cross_entropy(output, labels)
        loss.backward()
        
        # Check that gradients exist
        for param in model.parameters():
            assert param.grad is not None

    def test_mlp_hook_registration(self):
        """Test registering hooks on MLP layers."""
        model = MLPClassifier()
        model.register_hooks()
        
        x = torch.randn(2, 3, 32, 32)
        _ = model(x)
        
        # Check that activations were captured
        assert len(model.activations) > 0
        for layer_name, activation in model.activations.items():
            assert activation.shape[0] == 2  # Batch size

    def test_mlp_configure_optimizers(self):
        """Test optimizer configuration."""
        model = MLPClassifier(optimizer="adam", learning_rate=0.001)
        optimizer = model.configure_optimizers()
        assert optimizer is not None


class TestResNet18Classifier:
    """Test cases for ResNet18Classifier."""

    def test_resnet_initialization(self):
        """Test ResNet-18 initialization."""
        model = ResNet18Classifier()
        assert model.num_classes == 10

    def test_resnet_custom_classes(self):
        """Test ResNet-18 with custom number of classes."""
        model = ResNet18Classifier(num_classes=100)
        assert model.num_classes == 100

    def test_resnet_forward_pass(self):
        """Test forward pass with CIFAR-10 sized input."""
        model = ResNet18Classifier(modify_first_conv=True)
        x = torch.randn(4, 3, 32, 32)
        output = model(x)
        
        assert output.shape == (4, 10)

    def test_resnet_forward_pass_different_sizes(self):
        """Test forward pass with different input sizes."""
        model = ResNet18Classifier(modify_first_conv=False)
        
        # Test with larger images (224x224 like ImageNet)
        x = torch.randn(2, 3, 224, 224)
        output = model(x)
        assert output.shape == (2, 10)

    def test_resnet_modified_first_conv(self):
        """Test that first conv is modified for small images."""
        model = ResNet18Classifier(modify_first_conv=True)
        
        # Check that first conv has smaller kernel
        first_conv = model.model.conv1
        assert first_conv.kernel_size == (3, 3)
        assert first_conv.stride == (1, 1)

    def test_resnet_gradient_flow(self):
        """Test that gradients flow through ResNet."""
        model = ResNet18Classifier()
        x = torch.randn(4, 3, 32, 32)
        labels = torch.randint(0, 10, (4,))
        
        output = model(x)
        loss = torch.nn.functional.cross_entropy(output, labels)
        loss.backward()
        
        # Check that gradients exist
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None

    def test_resnet_hook_registration(self):
        """Test registering hooks on ResNet layers."""
        model = ResNet18Classifier()
        model.register_hooks(['conv1', 'layer1', 'layer4'])
        
        x = torch.randn(2, 3, 32, 32)
        _ = model(x)
        
        # Check that activations were captured
        assert 'conv1' in model.activations
        assert 'layer1' in model.activations
        assert 'layer4' in model.activations

    def test_resnet_configure_optimizers(self):
        """Test optimizer configuration."""
        model = ResNet18Classifier(optimizer="sgd", learning_rate=0.01)
        optimizer = model.configure_optimizers()
        assert optimizer is not None


class TestRepresentationExtraction:
    """Test representation extraction utilities."""

    def test_extract_representations_mlp(self):
        """Test extracting representations from MLP."""
        model = MLPClassifier()
        model.register_hooks()
        model.eval()
        
        # Create a simple dataloader
        from torch.utils.data import TensorDataset, DataLoader
        images = torch.randn(10, 3, 32, 32)
        labels = torch.randint(0, 10, (10,))
        dataset = TensorDataset(images, labels)
        dataloader = DataLoader(dataset, batch_size=5)
        
        representations = extract_representations(model, dataloader, device="cpu")
        
        assert len(representations) > 0
        for layer_name, activations in representations.items():
            assert activations.shape[0] == 10  # Total samples
            assert len(activations.shape) == 2  # Should be flattened

    def test_extract_representations_resnet(self):
        """Test extracting representations from ResNet."""
        model = ResNet18Classifier()
        model.register_hooks(['conv1', 'layer1'])
        model.eval()
        
        # Create a simple dataloader
        from torch.utils.data import TensorDataset, DataLoader
        images = torch.randn(10, 3, 32, 32)
        labels = torch.randint(0, 10, (10,))
        dataset = TensorDataset(images, labels)
        dataloader = DataLoader(dataset, batch_size=5)
        
        representations = extract_representations(model, dataloader, device="cpu")
        
        assert 'conv1' in representations
        assert 'layer1' in representations
        assert representations['conv1'].shape[0] == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
