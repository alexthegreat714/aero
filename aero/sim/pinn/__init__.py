"""
PINN (Physics-Informed Neural Networks) module for Aero Agent.

Provides neural network-based PDE solvers using PyTorch.
"""

from aero.sim.pinn.pinn_stub import PINNSolver
from aero.sim.pinn.pinn_model import (
    PINNModel,
    ResidualPINNModel,
    create_pinn_model,
    get_device,
    TORCH_AVAILABLE,
)
from aero.sim.pinn.pinn_trainer import (
    PINNTrainer,
    TrainingConfig,
    TrainingResult,
    check_pytorch_cuda,
)

__all__ = [
    # Legacy stub
    "PINNSolver",
    # Model classes
    "PINNModel",
    "ResidualPINNModel",
    "create_pinn_model",
    "get_device",
    "TORCH_AVAILABLE",
    # Trainer
    "PINNTrainer",
    "TrainingConfig",
    "TrainingResult",
    "check_pytorch_cuda",
]
