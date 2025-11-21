"""
Neural Network Models for Surrogate Modeling.

Provides MLP and PINN-like architectures for learning surrogates
of simulation outputs.
"""

import logging
from typing import List, Optional, Callable

logger = logging.getLogger(__name__)

# Check for PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Create dummy classes for type hints
    class nn:  # noqa
        Module = object


if TORCH_AVAILABLE:

    class MLP(nn.Module):
        """
        Simple Multi-Layer Perceptron for surrogate modeling.

        Maps input features to output predictions using fully-connected
        layers with configurable architecture.

        Example:
            model = MLP(input_dim=2, output_dim=1, hidden_dims=[64, 64])
            x = torch.randn(32, 2)
            y = model(x)  # Shape: (32, 1)
        """

        def __init__(
            self,
            input_dim: int,
            output_dim: int,
            hidden_dims: List[int],
            activation: str = "relu",
            dropout: float = 0.0,
        ):
            """
            Initialize the MLP.

            Args:
                input_dim: Input feature dimension
                output_dim: Output dimension
                hidden_dims: List of hidden layer sizes
                activation: Activation function ("relu", "tanh", "gelu")
                dropout: Dropout probability (0 = no dropout)
            """
            super().__init__()

            self.input_dim = input_dim
            self.output_dim = output_dim
            self.hidden_dims = hidden_dims

            # Select activation function
            if activation == "relu":
                self.activation = nn.ReLU()
            elif activation == "tanh":
                self.activation = nn.Tanh()
            elif activation == "gelu":
                self.activation = nn.GELU()
            else:
                self.activation = nn.ReLU()

            # Build layers
            layers = []
            prev_dim = input_dim

            for hidden_dim in hidden_dims:
                layers.append(nn.Linear(prev_dim, hidden_dim))
                layers.append(self.activation)
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
                prev_dim = hidden_dim

            # Output layer (no activation)
            layers.append(nn.Linear(prev_dim, output_dim))

            self.network = nn.Sequential(*layers)

            # Initialize weights
            self._init_weights()

        def _init_weights(self):
            """Initialize weights using Xavier initialization."""
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            """
            Forward pass.

            Args:
                x: Input tensor of shape (batch_size, input_dim)

            Returns:
                Output tensor of shape (batch_size, output_dim)
            """
            return self.network(x)

        def predict(self, x: "torch.Tensor") -> "torch.Tensor":
            """Alias for forward (for compatibility)."""
            return self.forward(x)


    class PINNSurrogate(nn.Module):
        """
        Physics-Informed Neural Network style surrogate.

        Similar architecture to MLP but designed for physics-based
        problems. Can optionally compute gradients for physics loss.

        Note: Full physics loss implementation is a future extension.
        Currently uses the same forward pass as MLP.

        Example:
            model = PINNSurrogate(input_dim=2, output_dim=1, hidden_dims=[64, 64, 64])
            x = torch.randn(32, 2, requires_grad=True)
            y = model(x)
        """

        def __init__(
            self,
            input_dim: int,
            output_dim: int,
            hidden_dims: List[int],
            activation: str = "tanh",
        ):
            """
            Initialize the PINN surrogate.

            Args:
                input_dim: Input dimension (e.g., 2 for (x, t))
                output_dim: Output dimension (e.g., 1 for u(x,t))
                hidden_dims: List of hidden layer sizes
                activation: Activation function (default: tanh for smooth gradients)
            """
            super().__init__()

            self.input_dim = input_dim
            self.output_dim = output_dim
            self.hidden_dims = hidden_dims

            # PINN typically uses tanh for smooth derivatives
            if activation == "tanh":
                self.activation = nn.Tanh()
            elif activation == "relu":
                self.activation = nn.ReLU()
            elif activation == "gelu":
                self.activation = nn.GELU()
            elif activation == "sin":
                # Sine activation (SIREN-style)
                self.activation = SineActivation()
            else:
                self.activation = nn.Tanh()

            # Build layers
            layers = []
            prev_dim = input_dim

            for hidden_dim in hidden_dims:
                layers.append(nn.Linear(prev_dim, hidden_dim))
                layers.append(self.activation)
                prev_dim = hidden_dim

            layers.append(nn.Linear(prev_dim, output_dim))

            self.network = nn.Sequential(*layers)

            # Initialize weights (important for PINNs)
            self._init_weights()

        def _init_weights(self):
            """Initialize weights using Xavier initialization."""
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            """
            Forward pass.

            Args:
                x: Input tensor of shape (batch_size, input_dim)

            Returns:
                Output tensor of shape (batch_size, output_dim)
            """
            return self.network(x)

        def predict(self, x: "torch.Tensor") -> "torch.Tensor":
            """Alias for forward."""
            return self.forward(x)

        def compute_gradients(
            self,
            x: "torch.Tensor",
            y: Optional["torch.Tensor"] = None,
        ) -> "torch.Tensor":
            """
            Compute gradients of output with respect to input.

            Useful for physics-informed loss computation.

            Args:
                x: Input tensor (requires_grad=True)
                y: Optional pre-computed output (if None, will compute)

            Returns:
                Gradient tensor of shape (batch_size, output_dim, input_dim)
            """
            if y is None:
                y = self.forward(x)

            grads = []
            for i in range(y.shape[1]):
                grad_i = torch.autograd.grad(
                    y[:, i].sum(),
                    x,
                    create_graph=True,
                    retain_graph=True,
                )[0]
                grads.append(grad_i)

            return torch.stack(grads, dim=1)


    class SineActivation(nn.Module):
        """Sine activation function (SIREN-style)."""

        def __init__(self, omega: float = 30.0):
            super().__init__()
            self.omega = omega

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return torch.sin(self.omega * x)


    class ResidualMLP(nn.Module):
        """
        MLP with residual connections for deeper networks.

        Uses skip connections to improve gradient flow in deeper
        architectures.
        """

        def __init__(
            self,
            input_dim: int,
            output_dim: int,
            hidden_dims: List[int],
            activation: str = "relu",
        ):
            super().__init__()

            self.input_dim = input_dim
            self.output_dim = output_dim

            # Select activation
            if activation == "relu":
                self.activation = nn.ReLU()
            elif activation == "tanh":
                self.activation = nn.Tanh()
            else:
                self.activation = nn.ReLU()

            # Input projection
            self.input_proj = nn.Linear(input_dim, hidden_dims[0])

            # Residual blocks
            self.blocks = nn.ModuleList()
            for i in range(len(hidden_dims) - 1):
                self.blocks.append(
                    ResidualBlock(hidden_dims[i], hidden_dims[i + 1], self.activation)
                )

            # Output projection
            self.output_proj = nn.Linear(hidden_dims[-1], output_dim)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            x = self.input_proj(x)
            x = self.activation(x)

            for block in self.blocks:
                x = block(x)

            return self.output_proj(x)


    class ResidualBlock(nn.Module):
        """Single residual block."""

        def __init__(self, in_dim: int, out_dim: int, activation: nn.Module):
            super().__init__()
            self.linear1 = nn.Linear(in_dim, out_dim)
            self.linear2 = nn.Linear(out_dim, out_dim)
            self.activation = activation

            # Skip connection projection if dimensions differ
            self.skip = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            residual = self.skip(x)
            x = self.activation(self.linear1(x))
            x = self.linear2(x)
            return self.activation(x + residual)


def create_model(
    model_type: str,
    input_dim: int,
    output_dim: int,
    hidden_dims: List[int],
    **kwargs,
) -> "nn.Module":
    """
    Factory function to create a model by type.

    Args:
        model_type: Type of model ("mlp", "pinn", "residual")
        input_dim: Input dimension
        output_dim: Output dimension
        hidden_dims: Hidden layer dimensions
        **kwargs: Additional model-specific arguments

    Returns:
        Initialized model

    Raises:
        ValueError: If model_type is unknown
        RuntimeError: If PyTorch is not available
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for surrogate models")

    model_type = model_type.lower()

    if model_type == "mlp":
        return MLP(input_dim, output_dim, hidden_dims, **kwargs)
    elif model_type == "pinn":
        return PINNSurrogate(input_dim, output_dim, hidden_dims, **kwargs)
    elif model_type == "residual":
        return ResidualMLP(input_dim, output_dim, hidden_dims, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


else:
    # Dummy implementations when PyTorch is not available
    class MLP:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch is required for MLP models")

    class PINNSurrogate:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch is required for PINN models")

    def create_model(*args, **kwargs):
        raise RuntimeError("PyTorch is required for surrogate models")
