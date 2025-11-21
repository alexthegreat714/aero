"""
PINN Trainer for Aero Agent.

Provides training loop and loss computation for Physics-Informed Neural Networks.
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from aero.sim.pinn.pinn_model import PINNModel, TORCH_AVAILABLE, get_device

logger = logging.getLogger(__name__)

if TORCH_AVAILABLE:
    import torch
    import torch.nn as nn
    import torch.optim as optim
else:
    torch = None


@dataclass
class TrainingConfig:
    """Configuration for PINN training."""

    epochs: int = 10000
    learning_rate: float = 1e-3
    batch_size: Optional[int] = None  # None = full batch
    optimizer: str = "adam"  # adam, lbfgs, sgd
    scheduler: Optional[str] = None  # None, step, cosine
    lambda_physics: float = 1.0
    lambda_data: float = 1.0
    lambda_bc: float = 10.0
    print_every: int = 1000
    save_every: Optional[int] = None
    save_path: Optional[str] = None
    early_stopping: bool = False
    patience: int = 1000
    min_delta: float = 1e-8


@dataclass
class TrainingResult:
    """Result from PINN training."""

    final_loss: float
    epochs_trained: int
    loss_history: List[float] = field(default_factory=list)
    physics_loss_history: List[float] = field(default_factory=list)
    data_loss_history: List[float] = field(default_factory=list)
    bc_loss_history: List[float] = field(default_factory=list)
    training_time: float = 0.0
    converged: bool = False


class PINNTrainer:
    """
    Trainer for Physics-Informed Neural Networks.

    Handles:
    - Training loop with configurable optimizers
    - Physics loss computation (PDE residual)
    - Boundary condition loss
    - Data loss (for supervised learning)
    - Device management (CPU/CUDA)
    - Model saving/loading

    Example:
        model = PINNModel([2, 64, 64, 1])
        trainer = PINNTrainer(model, config=TrainingConfig(epochs=5000))

        # Define physics loss (e.g., heat equation: du/dt = alpha * d²u/dx²)
        def physics_loss(model, x_colloc):
            x_colloc.requires_grad = True
            u = model(x_colloc)

            # Compute gradients
            du = torch.autograd.grad(u, x_colloc, torch.ones_like(u), create_graph=True)[0]
            du_dx, du_dt = du[:, 0:1], du[:, 1:2]

            d2u_dx2 = torch.autograd.grad(du_dx, x_colloc, torch.ones_like(du_dx), create_graph=True)[0][:, 0:1]

            # PDE residual: du/dt - alpha * d²u/dx²
            alpha = 0.01
            residual = du_dt - alpha * d2u_dx2

            return torch.mean(residual ** 2)

        trainer.set_physics_loss(physics_loss)
        trainer.set_collocation_points(x_colloc)
        result = trainer.train()
    """

    def __init__(
        self,
        model: PINNModel,
        config: Optional[TrainingConfig] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize the trainer.

        Args:
            model: PINNModel to train
            config: Training configuration
            device: Device to use ('cpu', 'cuda', or None for auto)
        """
        if not TORCH_AVAILABLE:
            logger.error("PyTorch required for PINNTrainer")
            return

        self.model = model
        self.config = config or TrainingConfig()
        self.device = device or get_device()

        # Move model to device
        self.model = self.model.to(self.device)

        # Create optimizer
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()

        # Training data
        self.collocation_points: Optional[torch.Tensor] = None
        self.bc_points: Optional[torch.Tensor] = None
        self.bc_values: Optional[torch.Tensor] = None
        self.data_points: Optional[torch.Tensor] = None
        self.data_values: Optional[torch.Tensor] = None

        # Custom loss functions
        self._physics_loss_fn: Optional[Callable] = None
        self._bc_loss_fn: Optional[Callable] = None

        # Training state
        self.loss_history: List[float] = []
        self.best_loss = float('inf')
        self.epochs_without_improvement = 0

        logger.info(f"PINNTrainer initialized on {self.device}")

    def _create_optimizer(self):
        """Create optimizer based on config."""
        if self.config.optimizer == "adam":
            return optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        elif self.config.optimizer == "lbfgs":
            return optim.LBFGS(
                self.model.parameters(),
                lr=self.config.learning_rate,
                max_iter=20,
                history_size=50,
            )
        elif self.config.optimizer == "sgd":
            return optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                momentum=0.9,
            )
        else:
            logger.warning(f"Unknown optimizer '{self.config.optimizer}', using Adam")
            return optim.Adam(self.model.parameters(), lr=self.config.learning_rate)

    def _create_scheduler(self):
        """Create learning rate scheduler."""
        if self.config.scheduler == "step":
            return optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.9)
        elif self.config.scheduler == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.epochs,
            )
        return None

    def set_collocation_points(self, points: np.ndarray) -> None:
        """Set collocation points for physics loss evaluation."""
        self.collocation_points = torch.tensor(
            points, dtype=torch.float32, device=self.device
        )
        logger.info(f"Set {len(points)} collocation points")

    def set_boundary_data(self, points: np.ndarray, values: np.ndarray) -> None:
        """Set boundary condition data."""
        self.bc_points = torch.tensor(points, dtype=torch.float32, device=self.device)
        self.bc_values = torch.tensor(values, dtype=torch.float32, device=self.device)
        logger.info(f"Set {len(points)} boundary points")

    def set_data(self, points: np.ndarray, values: np.ndarray) -> None:
        """Set supervised training data."""
        self.data_points = torch.tensor(points, dtype=torch.float32, device=self.device)
        self.data_values = torch.tensor(values, dtype=torch.float32, device=self.device)
        logger.info(f"Set {len(points)} data points")

    def set_physics_loss(self, loss_fn: Callable) -> None:
        """
        Set custom physics loss function.

        The function should have signature: loss_fn(model, collocation_points) -> torch.Tensor
        """
        self._physics_loss_fn = loss_fn

    def set_bc_loss(self, loss_fn: Callable) -> None:
        """
        Set custom boundary condition loss function.

        The function should have signature: loss_fn(model, bc_points, bc_values) -> torch.Tensor
        """
        self._bc_loss_fn = loss_fn

    def compute_physics_loss(self) -> torch.Tensor:
        """Compute physics loss (PDE residual)."""
        if self._physics_loss_fn is None or self.collocation_points is None:
            return torch.tensor(0.0, device=self.device)

        return self._physics_loss_fn(self.model, self.collocation_points)

    def compute_bc_loss(self) -> torch.Tensor:
        """Compute boundary condition loss."""
        if self.bc_points is None or self.bc_values is None:
            return torch.tensor(0.0, device=self.device)

        if self._bc_loss_fn is not None:
            return self._bc_loss_fn(self.model, self.bc_points, self.bc_values)

        # Default MSE loss
        predictions = self.model(self.bc_points)
        return torch.mean((predictions - self.bc_values) ** 2)

    def compute_data_loss(self) -> torch.Tensor:
        """Compute supervised data loss."""
        if self.data_points is None or self.data_values is None:
            return torch.tensor(0.0, device=self.device)

        predictions = self.model(self.data_points)
        return torch.mean((predictions - self.data_values) ** 2)

    def compute_total_loss(self) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total weighted loss.

        Returns:
            Tuple of (total_loss, loss_components_dict)
        """
        physics_loss = self.compute_physics_loss()
        bc_loss = self.compute_bc_loss()
        data_loss = self.compute_data_loss()

        total_loss = (
            self.config.lambda_physics * physics_loss
            + self.config.lambda_bc * bc_loss
            + self.config.lambda_data * data_loss
        )

        components = {
            "physics": physics_loss.item(),
            "bc": bc_loss.item(),
            "data": data_loss.item(),
            "total": total_loss.item(),
        }

        return total_loss, components

    def train_step(self) -> Dict[str, float]:
        """Execute one training step."""
        self.model.train()

        if self.config.optimizer == "lbfgs":
            # L-BFGS requires closure
            def closure():
                self.optimizer.zero_grad()
                loss, _ = self.compute_total_loss()
                loss.backward()
                return loss

            self.optimizer.step(closure)
            _, components = self.compute_total_loss()
        else:
            self.optimizer.zero_grad()
            loss, components = self.compute_total_loss()
            loss.backward()
            self.optimizer.step()

        if self.scheduler:
            self.scheduler.step()

        return components

    def train(self) -> TrainingResult:
        """
        Run the full training loop.

        Returns:
            TrainingResult with loss history and statistics
        """
        if not TORCH_AVAILABLE:
            logger.error("PyTorch not available for training")
            return TrainingResult(final_loss=float('inf'), epochs_trained=0)

        logger.info(f"Starting training for {self.config.epochs} epochs")
        start_time = time.time()

        result = TrainingResult(final_loss=0, epochs_trained=0)

        for epoch in range(self.config.epochs):
            components = self.train_step()

            # Record history
            result.loss_history.append(components["total"])
            result.physics_loss_history.append(components["physics"])
            result.bc_loss_history.append(components["bc"])
            result.data_loss_history.append(components["data"])

            # Early stopping check
            if self.config.early_stopping:
                if components["total"] < self.best_loss - self.config.min_delta:
                    self.best_loss = components["total"]
                    self.epochs_without_improvement = 0
                else:
                    self.epochs_without_improvement += 1

                if self.epochs_without_improvement >= self.config.patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    result.converged = True
                    break

            # Logging
            if epoch % self.config.print_every == 0:
                logger.info(
                    f"Epoch {epoch}: total={components['total']:.6e}, "
                    f"physics={components['physics']:.6e}, "
                    f"bc={components['bc']:.6e}, "
                    f"data={components['data']:.6e}"
                )

            # Save checkpoint
            if self.config.save_every and epoch % self.config.save_every == 0:
                if self.config.save_path:
                    self.save_checkpoint(f"{self.config.save_path}_epoch{epoch}.pt")

            result.epochs_trained = epoch + 1

        result.training_time = time.time() - start_time
        result.final_loss = result.loss_history[-1] if result.loss_history else float('inf')

        logger.info(
            f"Training completed in {result.training_time:.2f}s, "
            f"final loss: {result.final_loss:.6e}"
        )

        return result

    def save_checkpoint(self, path: str) -> None:
        """Save training checkpoint."""
        torch.save({
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'loss_history': self.loss_history,
            'config': self.config,
        }, path)
        logger.info(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str) -> None:
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.loss_history = checkpoint.get('loss_history', [])
        logger.info(f"Checkpoint loaded from {path}")

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Make predictions on new data."""
        self.model.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(x, dtype=torch.float32, device=self.device)
            y = self.model(x_tensor)
            return y.cpu().numpy()


def check_pytorch_cuda() -> Dict[str, bool]:
    """
    Check PyTorch and CUDA availability.

    Returns:
        Dictionary with availability status
    """
    result = {
        "pytorch_available": TORCH_AVAILABLE,
        "cuda_available": False,
        "cuda_device": None,
    }

    if TORCH_AVAILABLE:
        result["cuda_available"] = torch.cuda.is_available()
        if result["cuda_available"]:
            result["cuda_device"] = torch.cuda.get_device_name(0)

    return result
