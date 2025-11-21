"""
Surrogate Modeling / SciML Layer for Aero Agent.

Provides:
- Model registry for tracking trained surrogates
- Dataset builders from DataStore
- MLP and PINN-like models
- Training and evaluation utilities
- Prediction APIs
"""

from aero.surrogate.registry import SurrogateInfo, SurrogateRegistry
from aero.surrogate.models import MLP, PINNSurrogate, create_model
from aero.surrogate.trainer import SurrogateTrainer
from aero.surrogate.evaluator import evaluate_model, predict_single, predict_batch
from aero.surrogate.datasets import (
    SimulationFieldDataset,
    TimeSeriesDataset,
    TabularDataset,
    build_dataset_from_simulations,
    build_dataset_from_timeseries,
)

# Check for PyTorch availability
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Default registry instance
_default_registry: "SurrogateRegistry | None" = None


def get_default_registry() -> "SurrogateRegistry":
    """Get the default surrogate registry."""
    global _default_registry
    if _default_registry is None:
        raise RuntimeError(
            "Default surrogate registry not initialized. "
            "Call set_default_registry() first."
        )
    return _default_registry


def set_default_registry(registry: "SurrogateRegistry") -> None:
    """Set the default surrogate registry."""
    global _default_registry
    _default_registry = registry


__all__ = [
    # Registry
    "SurrogateInfo",
    "SurrogateRegistry",
    "get_default_registry",
    "set_default_registry",
    # Models
    "MLP",
    "PINNSurrogate",
    "create_model",
    # Training
    "SurrogateTrainer",
    # Evaluation
    "evaluate_model",
    "predict_single",
    "predict_batch",
    # Datasets
    "SimulationFieldDataset",
    "TimeSeriesDataset",
    "TabularDataset",
    "build_dataset_from_simulations",
    "build_dataset_from_timeseries",
    # Flags
    "TORCH_AVAILABLE",
]
