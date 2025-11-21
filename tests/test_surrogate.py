"""
Tests for the Surrogate Modeling module.

Tests registry, models, trainer, evaluator, and datasets.
"""

import pytest
import tempfile
import shutil
from pathlib import Path

import numpy as np


# Check for PyTorch
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test data."""
    tmp = tempfile.mkdtemp()
    yield tmp
    shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture
def models_dir(temp_dir):
    """Create a models directory."""
    models_path = Path(temp_dir) / "models"
    models_path.mkdir(parents=True, exist_ok=True)
    return str(models_path)


class TestSurrogateRegistry:
    """Tests for the SurrogateRegistry class."""

    def test_registry_initialization(self, models_dir):
        """Test registry initialization."""
        from aero.surrogate.registry import SurrogateRegistry

        registry = SurrogateRegistry(models_dir)
        assert registry is not None
        assert len(registry) == 0

    def test_register_model(self, models_dir):
        """Test registering a model."""
        from aero.surrogate.registry import SurrogateRegistry, SurrogateInfo

        registry = SurrogateRegistry(models_dir)

        info = SurrogateInfo(
            name="test_model",
            model_type="mlp",
            input_dim=2,
            output_dim=1,
            path=f"{models_dir}/test_model.pt",
            metadata={"epochs": 100},
        )

        registry.register(info)

        assert len(registry) == 1
        assert "test_model" in registry

    def test_get_model(self, models_dir):
        """Test getting a model by name."""
        from aero.surrogate.registry import SurrogateRegistry, SurrogateInfo

        registry = SurrogateRegistry(models_dir)

        info = SurrogateInfo(
            name="my_model",
            model_type="pinn",
            input_dim=3,
            output_dim=2,
            path=f"{models_dir}/my_model.pt",
            metadata={},
        )

        registry.register(info)

        retrieved = registry.get("my_model")
        assert retrieved is not None
        assert retrieved.name == "my_model"
        assert retrieved.model_type == "pinn"
        assert retrieved.input_dim == 3

    def test_get_nonexistent_model(self, models_dir):
        """Test getting a model that doesn't exist."""
        from aero.surrogate.registry import SurrogateRegistry

        registry = SurrogateRegistry(models_dir)
        result = registry.get("nonexistent")
        assert result is None

    def test_remove_model(self, models_dir):
        """Test removing a model."""
        from aero.surrogate.registry import SurrogateRegistry, SurrogateInfo

        registry = SurrogateRegistry(models_dir)

        info = SurrogateInfo(
            name="to_remove",
            model_type="mlp",
            input_dim=1,
            output_dim=1,
            path=f"{models_dir}/to_remove.pt",
            metadata={},
        )

        registry.register(info)
        assert len(registry) == 1

        success = registry.remove("to_remove")
        assert success is True
        assert len(registry) == 0

    def test_list_models(self, models_dir):
        """Test listing all models."""
        from aero.surrogate.registry import SurrogateRegistry, SurrogateInfo

        registry = SurrogateRegistry(models_dir)

        for i in range(3):
            info = SurrogateInfo(
                name=f"model_{i}",
                model_type="mlp",
                input_dim=2,
                output_dim=1,
                path=f"{models_dir}/model_{i}.pt",
                metadata={},
            )
            registry.register(info)

        models = registry.list_models()
        assert len(models) == 3

    def test_registry_persistence(self, models_dir):
        """Test that registry persists across instances."""
        from aero.surrogate.registry import SurrogateRegistry, SurrogateInfo

        registry1 = SurrogateRegistry(models_dir)
        info = SurrogateInfo(
            name="persistent_model",
            model_type="mlp",
            input_dim=2,
            output_dim=1,
            path=f"{models_dir}/persistent.pt",
            metadata={"test": True},
        )
        registry1.register(info)

        # Create new registry instance
        registry2 = SurrogateRegistry(models_dir)

        assert len(registry2) == 1
        assert "persistent_model" in registry2

    def test_get_models_by_type(self, models_dir):
        """Test filtering models by type."""
        from aero.surrogate.registry import SurrogateRegistry, SurrogateInfo

        registry = SurrogateRegistry(models_dir)

        for i in range(2):
            registry.register(SurrogateInfo(
                name=f"mlp_{i}",
                model_type="mlp",
                input_dim=2,
                output_dim=1,
                path=f"{models_dir}/mlp_{i}.pt",
                metadata={},
            ))

        registry.register(SurrogateInfo(
            name="pinn_0",
            model_type="pinn",
            input_dim=2,
            output_dim=1,
            path=f"{models_dir}/pinn_0.pt",
            metadata={},
        ))

        mlp_models = registry.get_models_by_type("mlp")
        pinn_models = registry.get_models_by_type("pinn")

        assert len(mlp_models) == 2
        assert len(pinn_models) == 1


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestSurrogateModels:
    """Tests for surrogate model architectures."""

    def test_mlp_creation(self):
        """Test MLP model creation."""
        from aero.surrogate.models import MLP

        model = MLP(
            input_dim=2,
            output_dim=1,
            hidden_dims=[64, 64],
        )

        assert model is not None
        assert model.input_dim == 2
        assert model.output_dim == 1

    def test_mlp_forward(self):
        """Test MLP forward pass."""
        from aero.surrogate.models import MLP

        model = MLP(
            input_dim=3,
            output_dim=2,
            hidden_dims=[32, 32],
        )

        x = torch.randn(10, 3)
        y = model(x)

        assert y.shape == (10, 2)

    def test_pinn_creation(self):
        """Test PINN model creation."""
        from aero.surrogate.models import PINNSurrogate

        model = PINNSurrogate(
            input_dim=2,
            output_dim=1,
            hidden_dims=[64, 64, 64],
        )

        assert model is not None
        assert model.input_dim == 2

    def test_pinn_forward(self):
        """Test PINN forward pass."""
        from aero.surrogate.models import PINNSurrogate

        model = PINNSurrogate(
            input_dim=2,
            output_dim=1,
            hidden_dims=[32, 32],
        )

        x = torch.randn(10, 2)
        y = model(x)

        assert y.shape == (10, 1)

    def test_create_model_factory(self):
        """Test model factory function."""
        from aero.surrogate.models import create_model

        mlp = create_model("mlp", 2, 1, [64, 64])
        pinn = create_model("pinn", 2, 1, [64, 64])

        assert mlp is not None
        assert pinn is not None

    def test_create_model_unknown_type(self):
        """Test factory with unknown model type."""
        from aero.surrogate.models import create_model

        with pytest.raises(ValueError):
            create_model("unknown_type", 2, 1, [64])


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestSurrogateDatasets:
    """Tests for surrogate datasets."""

    def test_tabular_dataset(self):
        """Test TabularDataset creation."""
        from aero.surrogate.datasets import TabularDataset

        X = np.random.randn(100, 3).astype(np.float32)
        y = np.random.randn(100, 1).astype(np.float32)

        dataset = TabularDataset(X, y, normalize=True)

        assert len(dataset) == 100

        x_sample, y_sample = dataset[0]
        assert x_sample.shape == (3,)
        assert y_sample.shape == (1,)

    def test_tabular_dataset_no_normalize(self):
        """Test TabularDataset without normalization."""
        from aero.surrogate.datasets import TabularDataset

        X = np.random.randn(50, 2).astype(np.float32)
        y = np.random.randn(50, 1).astype(np.float32)

        dataset = TabularDataset(X, y, normalize=False)

        assert len(dataset) == 50


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestSurrogateTrainer:
    """Tests for SurrogateTrainer."""

    def test_train_mlp_simple(self, models_dir):
        """Test training a simple MLP on synthetic data."""
        from aero.surrogate.registry import SurrogateRegistry
        from aero.surrogate.trainer import SurrogateTrainer
        from aero.surrogate.datasets import TabularDataset

        # Create simple dataset: y = 2x + 1
        X = np.random.randn(100, 1).astype(np.float32)
        y = (2 * X + 1).astype(np.float32)

        dataset = TabularDataset(X, y, normalize=True)

        # Create registry and trainer
        registry = SurrogateRegistry(models_dir)
        trainer = SurrogateTrainer(registry, device="cpu")

        # Train
        info = trainer.train_mlp(
            name="test_linear",
            dataset=dataset,
            input_dim=1,
            output_dim=1,
            hidden_dims=[16, 16],
            epochs=10,
            batch_size=32,
            lr=0.01,
        )

        assert info is not None
        assert info.name == "test_linear"
        assert info.model_type == "mlp"
        assert Path(info.path).exists()
        assert "test_linear" in registry

    def test_train_pinn_simple(self, models_dir):
        """Test training a simple PINN on synthetic data."""
        from aero.surrogate.registry import SurrogateRegistry
        from aero.surrogate.trainer import SurrogateTrainer
        from aero.surrogate.datasets import TabularDataset

        # Create simple dataset
        X = np.random.randn(100, 2).astype(np.float32)
        y = (X[:, 0:1] * X[:, 1:2]).astype(np.float32)

        dataset = TabularDataset(X, y, normalize=True)

        # Create registry and trainer
        registry = SurrogateRegistry(models_dir)
        trainer = SurrogateTrainer(registry, device="cpu")

        # Train
        info = trainer.train_pinn(
            name="test_pinn",
            dataset=dataset,
            input_dim=2,
            output_dim=1,
            hidden_dims=[16, 16],
            epochs=10,
            batch_size=32,
            lr=0.01,
        )

        assert info is not None
        assert info.model_type == "pinn"

    def test_load_trained_model(self, models_dir):
        """Test loading a trained model."""
        from aero.surrogate.registry import SurrogateRegistry
        from aero.surrogate.trainer import SurrogateTrainer
        from aero.surrogate.datasets import TabularDataset

        # Train a model
        X = np.random.randn(50, 1).astype(np.float32)
        y = (X ** 2).astype(np.float32)
        dataset = TabularDataset(X, y, normalize=True)

        registry = SurrogateRegistry(models_dir)
        trainer = SurrogateTrainer(registry, device="cpu")

        trainer.train_mlp(
            name="loadable_model",
            dataset=dataset,
            input_dim=1,
            output_dim=1,
            hidden_dims=[16],
            epochs=5,
            batch_size=16,
            lr=0.01,
        )

        # Load it
        model, checkpoint = trainer.load_model("loadable_model")

        assert model is not None
        assert checkpoint is not None
        assert checkpoint["input_dim"] == 1
        assert checkpoint["output_dim"] == 1


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestSurrogateEvaluator:
    """Tests for surrogate evaluation utilities."""

    def test_evaluate_model(self, models_dir):
        """Test model evaluation."""
        from aero.surrogate.registry import SurrogateRegistry
        from aero.surrogate.trainer import SurrogateTrainer
        from aero.surrogate.evaluator import evaluate_model
        from aero.surrogate.datasets import TabularDataset

        # Train a model
        X = np.random.randn(100, 2).astype(np.float32)
        y = (X[:, 0:1] + X[:, 1:2]).astype(np.float32)
        dataset = TabularDataset(X, y, normalize=True)

        registry = SurrogateRegistry(models_dir)
        trainer = SurrogateTrainer(registry, device="cpu")

        trainer.train_mlp(
            name="eval_model",
            dataset=dataset,
            input_dim=2,
            output_dim=1,
            hidden_dims=[32, 32],
            epochs=20,
            batch_size=32,
            lr=0.01,
        )

        model, _ = trainer.load_model("eval_model")

        # Evaluate
        metrics = evaluate_model(model, dataset, device="cpu")

        assert "mse" in metrics
        assert "mae" in metrics
        assert "r2" in metrics
        assert metrics["num_samples"] > 0

    def test_predict_single(self, models_dir):
        """Test single prediction."""
        from aero.surrogate.models import MLP
        from aero.surrogate.evaluator import predict_single

        model = MLP(input_dim=2, output_dim=1, hidden_dims=[16])
        model.eval()

        x = np.array([1.0, 2.0], dtype=np.float32)
        y = predict_single(model, x, device="cpu")

        assert y is not None
        assert isinstance(y, np.ndarray)

    def test_predict_batch(self, models_dir):
        """Test batch prediction."""
        from aero.surrogate.models import MLP
        from aero.surrogate.evaluator import predict_batch

        model = MLP(input_dim=3, output_dim=2, hidden_dims=[16])
        model.eval()

        inputs = np.random.randn(20, 3).astype(np.float32)
        outputs = predict_batch(model, inputs, device="cpu")

        assert outputs.shape == (20, 2)

    def test_predict_with_normalization(self, models_dir):
        """Test prediction with normalization."""
        from aero.surrogate.models import MLP
        from aero.surrogate.evaluator import predict_single, predict_batch

        model = MLP(input_dim=2, output_dim=1, hidden_dims=[16])
        model.eval()

        normalization = {
            "input_mean": [0.0, 0.0],
            "input_std": [1.0, 1.0],
            "output_mean": [0.0],
            "output_std": [1.0],
        }

        x = np.array([1.0, 2.0], dtype=np.float32)
        y = predict_single(model, x, device="cpu", normalization=normalization)

        assert y is not None


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestEndToEnd:
    """End-to-end tests for the surrogate system."""

    def test_full_workflow(self, models_dir):
        """Test complete surrogate workflow: train -> save -> load -> predict."""
        from aero.surrogate.registry import SurrogateRegistry
        from aero.surrogate.trainer import SurrogateTrainer
        from aero.surrogate.evaluator import predict_single, evaluate_model
        from aero.surrogate.datasets import TabularDataset

        # Create dataset: y = sin(x)
        X = np.linspace(-np.pi, np.pi, 200).reshape(-1, 1).astype(np.float32)
        y = np.sin(X).astype(np.float32)
        dataset = TabularDataset(X, y, normalize=True)

        # Train
        registry = SurrogateRegistry(models_dir)
        trainer = SurrogateTrainer(registry, device="cpu")

        info = trainer.train_mlp(
            name="sin_surrogate",
            dataset=dataset,
            input_dim=1,
            output_dim=1,
            hidden_dims=[32, 32],
            epochs=50,
            batch_size=32,
            lr=0.01,
            metadata={"sim_type": "test_function"},
        )

        # Verify registration
        assert "sin_surrogate" in registry

        # Load and evaluate
        model, checkpoint = trainer.load_model("sin_surrogate")
        metrics = evaluate_model(model, dataset, device="cpu")

        # Should have learned reasonably well
        assert metrics["mse"] < 0.5  # Loose bound for quick test

        # Test prediction
        x_test = np.array([0.0], dtype=np.float32)
        normalization = checkpoint.get("normalization")
        y_pred = predict_single(model, x_test, device="cpu", normalization=normalization)

        # sin(0) should be close to 0
        assert abs(y_pred) < 1.0  # Loose bound

    def test_model_metadata(self, models_dir):
        """Test that model metadata is preserved."""
        from aero.surrogate.registry import SurrogateRegistry
        from aero.surrogate.trainer import SurrogateTrainer
        from aero.surrogate.datasets import TabularDataset

        X = np.random.randn(50, 1).astype(np.float32)
        y = X.copy()
        dataset = TabularDataset(X, y, normalize=False)

        registry = SurrogateRegistry(models_dir)
        trainer = SurrogateTrainer(registry, device="cpu")

        info = trainer.train_mlp(
            name="metadata_test",
            dataset=dataset,
            input_dim=1,
            output_dim=1,
            hidden_dims=[8],
            epochs=5,
            batch_size=16,
            lr=0.01,
            metadata={
                "sim_type": "heat_1d",
                "custom_key": "custom_value",
            },
        )

        # Check metadata
        assert info.metadata["sim_type"] == "heat_1d"
        assert info.metadata["custom_key"] == "custom_value"
        assert "epochs" in info.metadata
        assert info.metadata["epochs"] == 5

        # Verify persistence
        registry2 = SurrogateRegistry(models_dir)
        loaded_info = registry2.get("metadata_test")
        assert loaded_info.metadata["sim_type"] == "heat_1d"
