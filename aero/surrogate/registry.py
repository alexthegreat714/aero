"""
Surrogate Model Registry for Aero Agent.

Tracks registered surrogate models with metadata and file paths.
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class SurrogateInfo:
    """
    Information about a registered surrogate model.

    Attributes:
        name: Unique name for the model
        model_type: Type of model (e.g., "mlp", "pinn")
        input_dim: Input dimension
        output_dim: Output dimension
        path: Path to the saved model file
        metadata: Additional metadata (training info, metrics, etc.)
    """

    name: str
    model_type: str
    input_dim: int
    output_dim: int
    path: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SurrogateInfo":
        """Create from dictionary."""
        return cls(
            name=d["name"],
            model_type=d["model_type"],
            input_dim=d["input_dim"],
            output_dim=d["output_dim"],
            path=d["path"],
            metadata=d.get("metadata", {}),
        )


class SurrogateRegistry:
    """
    Registry for tracking surrogate models.

    Persists model information to a JSON file and provides
    CRUD operations for model registration.

    Example:
        registry = SurrogateRegistry("./data/models")

        # Register a model
        info = SurrogateInfo(
            name="heat1d_v1",
            model_type="mlp",
            input_dim=2,
            output_dim=1,
            path="./data/models/heat1d_v1.pt",
            metadata={"epochs": 100, "mse": 0.001}
        )
        registry.register(info)

        # List all models
        models = registry.list_models()

        # Get a specific model
        model_info = registry.get("heat1d_v1")
    """

    REGISTRY_FILENAME = "models_registry.json"

    def __init__(self, models_dir: str):
        """
        Initialize the registry.

        Args:
            models_dir: Directory for storing models and registry file
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self._registry_path = self.models_dir / self.REGISTRY_FILENAME
        self._models: Dict[str, SurrogateInfo] = {}

        self._load_registry()
        logger.info(
            f"SurrogateRegistry initialized: {len(self._models)} models "
            f"(dir: {self.models_dir})"
        )

    def _load_registry(self) -> None:
        """Load registry from disk."""
        if not self._registry_path.exists():
            self._models = {}
            return

        try:
            with open(self._registry_path, "r") as f:
                data = json.load(f)

            self._models = {}
            for name, info_dict in data.items():
                self._models[name] = SurrogateInfo.from_dict(info_dict)

            logger.debug(f"Loaded {len(self._models)} models from registry")

        except Exception as e:
            logger.warning(f"Failed to load registry: {e}")
            self._models = {}

    def _save_registry(self) -> None:
        """Save registry to disk."""
        try:
            data = {name: info.to_dict() for name, info in self._models.items()}

            with open(self._registry_path, "w") as f:
                json.dump(data, f, indent=2, default=str)

            logger.debug(f"Saved registry with {len(self._models)} models")

        except Exception as e:
            logger.error(f"Failed to save registry: {e}")

    def register(self, info: SurrogateInfo) -> None:
        """
        Register a surrogate model.

        Args:
            info: Model information to register
        """
        # Add registration timestamp if not present
        if "registered_at" not in info.metadata:
            info.metadata["registered_at"] = datetime.now().isoformat()

        self._models[info.name] = info
        self._save_registry()
        logger.info(f"Registered model: {info.name} ({info.model_type})")

    def list_models(self) -> List[SurrogateInfo]:
        """
        List all registered models.

        Returns:
            List of SurrogateInfo objects
        """
        return list(self._models.values())

    def get(self, name: str) -> Optional[SurrogateInfo]:
        """
        Get a model by name.

        Args:
            name: Model name

        Returns:
            SurrogateInfo or None if not found
        """
        return self._models.get(name)

    def remove(self, name: str) -> bool:
        """
        Remove a model from the registry.

        Note: This does not delete the model file.

        Args:
            name: Model name

        Returns:
            True if removed, False if not found
        """
        if name not in self._models:
            return False

        del self._models[name]
        self._save_registry()
        logger.info(f"Removed model from registry: {name}")
        return True

    def update_metadata(self, name: str, metadata: Dict[str, Any]) -> bool:
        """
        Update metadata for a registered model.

        Args:
            name: Model name
            metadata: New metadata (merged with existing)

        Returns:
            True if updated, False if not found
        """
        if name not in self._models:
            return False

        self._models[name].metadata.update(metadata)
        self._save_registry()
        return True

    def get_models_by_type(self, model_type: str) -> List[SurrogateInfo]:
        """
        Get all models of a specific type.

        Args:
            model_type: Model type (e.g., "mlp", "pinn")

        Returns:
            List of matching models
        """
        return [
            info for info in self._models.values()
            if info.model_type == model_type
        ]

    def get_models_for_sim_type(self, sim_type: str) -> List[SurrogateInfo]:
        """
        Get all models trained for a specific simulation type.

        Args:
            sim_type: Simulation type (e.g., "heat_1d", "laplace_2d")

        Returns:
            List of matching models
        """
        return [
            info for info in self._models.values()
            if info.metadata.get("sim_type") == sim_type
        ]

    def model_exists(self, name: str) -> bool:
        """Check if a model with the given name exists."""
        return name in self._models

    def get_model_path(self, name: str) -> Optional[Path]:
        """
        Get the file path for a model.

        Args:
            name: Model name

        Returns:
            Path to model file or None if not found
        """
        info = self.get(name)
        if info is None:
            return None
        return Path(info.path)

    def __len__(self) -> int:
        """Return number of registered models."""
        return len(self._models)

    def __contains__(self, name: str) -> bool:
        """Check if a model is registered."""
        return name in self._models
