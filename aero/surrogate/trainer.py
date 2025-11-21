"""
Surrogate Model Trainer for Aero Agent.

Provides unified training loop for MLP and PINN surrogate models.
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from aero.surrogate.registry import SurrogateInfo, SurrogateRegistry

logger = logging.getLogger(__name__)

# Check for PyTorch
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset, random_split
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class SurrogateTrainer:
    """
    Trainer for surrogate models.

    Handles training loop, validation, saving, and registration
    of trained models.

    Example:
        registry = SurrogateRegistry("./data/models")
        trainer = SurrogateTrainer(registry)

        info = trainer.train_mlp(
            name="heat1d_v1",
            dataset=dataset,
            input_dim=2,
            output_dim=1,
            hidden_dims=[64, 64],
            epochs=100,
            batch_size=32,
            lr=0.001,
        )
    """

    def __init__(
        self,
        registry: SurrogateRegistry,
        device: Optional[str] = None,
    ):
        """
        Initialize the trainer.

        Args:
            registry: Model registry for saving trained models
            device: Device to train on ("cpu", "cuda", or None for auto)
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for SurrogateTrainer")

        self.registry = registry
        self.models_dir = registry.models_dir

        # Determine device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"SurrogateTrainer initialized (device: {self.device})")

    def train_mlp(
        self,
        name: str,
        dataset: "Dataset",
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int],
        epochs: int = 100,
        batch_size: int = 32,
        lr: float = 0.001,
        validation_split: float = 0.1,
        activation: str = "relu",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SurrogateInfo:
        """
        Train an MLP surrogate model.

        Args:
            name: Model name for registration
            dataset: Training dataset
            input_dim: Input dimension
            output_dim: Output dimension
            hidden_dims: Hidden layer dimensions
            epochs: Number of training epochs
            batch_size: Batch size
            lr: Learning rate
            validation_split: Fraction for validation
            activation: Activation function
            metadata: Additional metadata to store

        Returns:
            SurrogateInfo for the trained model
        """
        from aero.surrogate.models import MLP

        model = MLP(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            activation=activation,
        )

        return self._train_model(
            name=name,
            model=model,
            model_type="mlp",
            dataset=dataset,
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            validation_split=validation_split,
            metadata=metadata,
        )

    def train_pinn(
        self,
        name: str,
        dataset: "Dataset",
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int],
        epochs: int = 100,
        batch_size: int = 32,
        lr: float = 0.001,
        validation_split: float = 0.1,
        activation: str = "tanh",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SurrogateInfo:
        """
        Train a PINN-style surrogate model.

        Args:
            name: Model name for registration
            dataset: Training dataset
            input_dim: Input dimension
            output_dim: Output dimension
            hidden_dims: Hidden layer dimensions
            epochs: Number of training epochs
            batch_size: Batch size
            lr: Learning rate
            validation_split: Fraction for validation
            activation: Activation function
            metadata: Additional metadata to store

        Returns:
            SurrogateInfo for the trained model
        """
        from aero.surrogate.models import PINNSurrogate

        model = PINNSurrogate(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            activation=activation,
        )

        return self._train_model(
            name=name,
            model=model,
            model_type="pinn",
            dataset=dataset,
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            validation_split=validation_split,
            metadata=metadata,
        )

    def _train_model(
        self,
        name: str,
        model: "nn.Module",
        model_type: str,
        dataset: "Dataset",
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int],
        epochs: int,
        batch_size: int,
        lr: float,
        validation_split: float,
        metadata: Optional[Dict[str, Any]],
    ) -> SurrogateInfo:
        """Internal training loop."""
        start_time = time.time()

        # Move model to device
        model = model.to(self.device)

        # Split dataset
        n_total = len(dataset)
        n_val = int(n_total * validation_split)
        n_train = n_total - n_val

        if n_val > 0:
            train_dataset, val_dataset = random_split(dataset, [n_train, n_val])
        else:
            train_dataset = dataset
            val_dataset = None

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
        )

        val_loader = None
        if val_dataset is not None and len(val_dataset) > 0:
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
            )

        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # Training loop
        best_val_loss = float("inf")
        best_model_state = None
        train_losses = []
        val_losses = []

        logger.info(
            f"Training {model_type} model '{name}': "
            f"{n_train} train, {n_val} val samples, {epochs} epochs"
        )

        for epoch in range(epochs):
            # Training
            model.train()
            epoch_train_loss = 0.0
            n_batches = 0

            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                predictions = model(batch_x)
                loss = criterion(predictions, batch_y)
                loss.backward()
                optimizer.step()

                epoch_train_loss += loss.item()
                n_batches += 1

            avg_train_loss = epoch_train_loss / max(n_batches, 1)
            train_losses.append(avg_train_loss)

            # Validation
            val_loss = None
            if val_loader is not None:
                model.eval()
                epoch_val_loss = 0.0
                n_val_batches = 0

                with torch.no_grad():
                    for batch_x, batch_y in val_loader:
                        batch_x = batch_x.to(self.device)
                        batch_y = batch_y.to(self.device)

                        predictions = model(batch_x)
                        loss = criterion(predictions, batch_y)
                        epoch_val_loss += loss.item()
                        n_val_batches += 1

                val_loss = epoch_val_loss / max(n_val_batches, 1)
                val_losses.append(val_loss)

                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = model.state_dict().copy()
            else:
                # No validation: save based on training loss
                if avg_train_loss < best_val_loss:
                    best_val_loss = avg_train_loss
                    best_model_state = model.state_dict().copy()

            # Log progress
            if (epoch + 1) % max(1, epochs // 10) == 0 or epoch == 0:
                val_str = f", val_loss={val_loss:.6f}" if val_loss else ""
                logger.debug(
                    f"Epoch {epoch + 1}/{epochs}: "
                    f"train_loss={avg_train_loss:.6f}{val_str}"
                )

        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        # Save model
        model_path = self.models_dir / f"{name}.pt"
        torch.save({
            "model_state_dict": model.state_dict(),
            "model_type": model_type,
            "input_dim": input_dim,
            "output_dim": output_dim,
            "hidden_dims": hidden_dims,
            "normalization": self._get_normalization(dataset),
        }, model_path)

        training_time = time.time() - start_time

        # Build metadata
        final_metadata = {
            "model_type": model_type,
            "hidden_dims": hidden_dims,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "device": self.device,
            "train_samples": n_train,
            "val_samples": n_val,
            "final_train_loss": train_losses[-1] if train_losses else None,
            "final_val_loss": val_losses[-1] if val_losses else None,
            "best_val_loss": best_val_loss,
            "training_time_seconds": training_time,
        }

        if metadata:
            final_metadata.update(metadata)

        # Register model
        info = SurrogateInfo(
            name=name,
            model_type=model_type,
            input_dim=input_dim,
            output_dim=output_dim,
            path=str(model_path),
            metadata=final_metadata,
        )

        self.registry.register(info)

        logger.info(
            f"Model '{name}' trained: "
            f"best_loss={best_val_loss:.6f}, time={training_time:.1f}s"
        )

        return info

    def _get_normalization(self, dataset: "Dataset") -> Optional[Dict]:
        """Extract normalization parameters from dataset."""
        if hasattr(dataset, "get_normalization_params"):
            params = dataset.get_normalization_params()
            # Convert numpy arrays to lists for JSON serialization
            return {k: v.tolist() for k, v in params.items()}
        return None

    def load_model(
        self,
        name: str,
    ) -> Tuple["nn.Module", Dict[str, Any]]:
        """
        Load a trained model by name.

        Args:
            name: Model name

        Returns:
            Tuple of (model, checkpoint_dict)

        Raises:
            ValueError: If model not found
        """
        info = self.registry.get(name)
        if info is None:
            raise ValueError(f"Model '{name}' not found in registry")

        model_path = Path(info.path)
        if not model_path.exists():
            raise ValueError(f"Model file not found: {model_path}")

        checkpoint = torch.load(model_path, map_location=self.device)

        # Recreate model
        from aero.surrogate.models import create_model

        model = create_model(
            model_type=checkpoint["model_type"],
            input_dim=checkpoint["input_dim"],
            output_dim=checkpoint["output_dim"],
            hidden_dims=checkpoint["hidden_dims"],
        )

        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(self.device)
        model.eval()

        return model, checkpoint

    def retrain(
        self,
        name: str,
        dataset: "Dataset",
        epochs: int = 100,
        batch_size: int = 32,
        lr: float = 0.001,
        **kwargs,
    ) -> SurrogateInfo:
        """
        Retrain an existing model with new data.

        Args:
            name: Model name to retrain
            dataset: New training dataset
            epochs: Number of epochs
            batch_size: Batch size
            lr: Learning rate
            **kwargs: Additional training arguments

        Returns:
            Updated SurrogateInfo
        """
        info = self.registry.get(name)
        if info is None:
            raise ValueError(f"Model '{name}' not found")

        # Get original parameters
        model_type = info.model_type
        hidden_dims = info.metadata.get("hidden_dims", [64, 64])

        if model_type == "mlp":
            return self.train_mlp(
                name=name,
                dataset=dataset,
                input_dim=info.input_dim,
                output_dim=info.output_dim,
                hidden_dims=hidden_dims,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                **kwargs,
            )
        elif model_type == "pinn":
            return self.train_pinn(
                name=name,
                dataset=dataset,
                input_dim=info.input_dim,
                output_dim=info.output_dim,
                hidden_dims=hidden_dims,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                **kwargs,
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
