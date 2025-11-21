"""
Physics-Informed Neural Network (PINN) model for Aero Agent.

Provides PyTorch MLP architecture for solving PDEs using neural networks.
The network learns to satisfy both data constraints and physical laws.
"""

import logging
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Try to import PyTorch
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    logger.warning("PyTorch not available - PINN model will be non-functional")


def get_device(prefer_cuda: bool = True) -> str:
    """
    Detect available device for computation.

    Args:
        prefer_cuda: If True, use CUDA when available

    Returns:
        Device string ('cuda' or 'cpu')
    """
    if not TORCH_AVAILABLE:
        return "cpu"

    if prefer_cuda and torch.cuda.is_available():
        device = "cuda"
        logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"

    return device


class PINNModel(nn.Module if TORCH_AVAILABLE else object):
    """
    Multi-Layer Perceptron for Physics-Informed Neural Networks.

    Architecture: input -> [hidden layers with activation] -> output

    The network approximates the solution u(x, t) of a PDE.
    Automatic differentiation enables computing derivatives for physics loss.

    Args:
        layers: List of layer dimensions [input_dim, hidden1, hidden2, ..., output_dim]
        activation: Activation function ('tanh', 'relu', 'gelu', 'sin')

    Example:
        # For 2D input (x, t) and 1D output (u)
        model = PINNModel([2, 64, 64, 64, 1])

        # Forward pass
        x = torch.tensor([[0.5, 0.0], [0.5, 0.1]])
        u = model(x)

        # Compute gradients for physics loss
        x.requires_grad = True
        u = model(x)
        du_dx = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u))[0]
    """

    def __init__(
        self,
        layers: List[int],
        activation: str = "tanh",
    ):
        """
        Initialize the PINN model.

        Args:
            layers: Layer dimensions [input, hidden..., output]
            activation: Activation function name
        """
        if not TORCH_AVAILABLE:
            logger.error("PyTorch required for PINNModel")
            return

        super().__init__()

        self.layers_config = layers
        self.activation_name = activation

        # Build network layers
        self.linears = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.linears.append(nn.Linear(layers[i], layers[i + 1]))

        # Select activation function
        self.activation = self._get_activation(activation)

        # Initialize weights using Xavier initialization
        self._init_weights()

        logger.info(f"PINNModel created: {layers}, activation={activation}")

    def _get_activation(self, name: str):
        """Get activation function by name."""
        activations = {
            "tanh": nn.Tanh(),
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "sigmoid": nn.Sigmoid(),
            "softplus": nn.Softplus(),
        }

        if name == "sin":
            # Custom sine activation (good for periodic solutions)
            return lambda x: torch.sin(x)

        if name not in activations:
            logger.warning(f"Unknown activation '{name}', using tanh")
            return nn.Tanh()

        return activations[name]

    def _init_weights(self) -> None:
        """Initialize weights using Xavier/Glorot initialization."""
        for linear in self.linears:
            nn.init.xavier_normal_(linear.weight)
            nn.init.zeros_(linear.bias)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        for i, linear in enumerate(self.linears[:-1]):
            x = linear(x)
            x = self.activation(x)

        # Last layer without activation
        x = self.linears[-1](x)
        return x

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Make predictions on numpy array.

        Args:
            x: Input array (N, input_dim)

        Returns:
            Predictions (N, output_dim)
        """
        self.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(x, dtype=torch.float32)
            if next(self.parameters()).is_cuda:
                x_tensor = x_tensor.cuda()
            y = self(x_tensor)
            return y.cpu().numpy()

    @property
    def num_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save(self, path: str) -> None:
        """Save model state to file."""
        torch.save({
            'state_dict': self.state_dict(),
            'layers': self.layers_config,
            'activation': self.activation_name,
        }, path)
        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str) -> "PINNModel":
        """Load model from file."""
        checkpoint = torch.load(path, map_location='cpu')
        model = cls(
            layers=checkpoint['layers'],
            activation=checkpoint.get('activation', 'tanh'),
        )
        model.load_state_dict(checkpoint['state_dict'])
        logger.info(f"Model loaded from {path}")
        return model


class ResidualPINNModel(nn.Module if TORCH_AVAILABLE else object):
    """
    PINN with residual (skip) connections.

    Residual connections can help with training deeper networks
    and preserving gradient flow.
    """

    def __init__(
        self,
        layers: List[int],
        activation: str = "tanh",
    ):
        if not TORCH_AVAILABLE:
            return

        super().__init__()

        self.input_dim = layers[0]
        self.output_dim = layers[-1]
        self.hidden_dims = layers[1:-1]

        # Input projection
        self.input_layer = nn.Linear(self.input_dim, self.hidden_dims[0])

        # Hidden layers with residual connections
        self.hidden_layers = nn.ModuleList()
        for i in range(len(self.hidden_dims) - 1):
            block = nn.Sequential(
                nn.Linear(self.hidden_dims[i], self.hidden_dims[i + 1]),
                nn.Tanh() if activation == "tanh" else nn.ReLU(),
            )
            self.hidden_layers.append(block)

        # Output layer
        self.output_layer = nn.Linear(self.hidden_dims[-1], self.output_dim)

        # Initialize
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        x = torch.tanh(self.input_layer(x))

        for layer in self.hidden_layers:
            # Residual connection (only if dimensions match)
            residual = x
            x = layer(x)
            if x.shape == residual.shape:
                x = x + residual

        return self.output_layer(x)


def create_pinn_model(
    input_dim: int = 2,
    output_dim: int = 1,
    hidden_layers: int = 4,
    hidden_dim: int = 64,
    activation: str = "tanh",
    residual: bool = False,
) -> "PINNModel":
    """
    Factory function to create a PINN model.

    Args:
        input_dim: Input dimension (e.g., 2 for (x, t))
        output_dim: Output dimension
        hidden_layers: Number of hidden layers
        hidden_dim: Dimension of each hidden layer
        activation: Activation function
        residual: Use residual connections

    Returns:
        PINNModel instance
    """
    layers = [input_dim] + [hidden_dim] * hidden_layers + [output_dim]

    if residual and TORCH_AVAILABLE:
        return ResidualPINNModel(layers, activation)

    return PINNModel(layers, activation)
