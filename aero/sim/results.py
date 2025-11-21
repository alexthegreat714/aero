"""
Simulation results container for Aero Agent.

Provides structured storage and serialization of simulation outputs.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

import numpy as np

logger = logging.getLogger(__name__)


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy arrays."""

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return {
                "_type": "ndarray",
                "dtype": str(obj.dtype),
                "shape": obj.shape,
                "data": obj.tolist(),
            }
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


def numpy_decoder(dct):
    """JSON decoder hook for numpy arrays."""
    if "_type" in dct and dct["_type"] == "ndarray":
        return np.array(dct["data"], dtype=dct["dtype"])
    return dct


@dataclass
class SimulationResult:
    """
    Container for simulation results.

    Stores:
    - id: Unique identifier for the result (UUID)
    - fields: Dictionary of output arrays (velocity, pressure, temperature, etc.)
    - metadata: Information about the simulation run
    - tags: List of tags for categorization

    Example:
        result = SimulationResult(
            fields={"temperature": np.array([1, 2, 3])},
            metadata={"solver": "heat_1d", "steps": 1000}
        )
        result.save("output/heat_result")
        json_str = result.to_json()
    """

    fields: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    id: str = field(default_factory=lambda: str(uuid4()))

    def __post_init__(self):
        """Validate fields."""
        if self.fields is None:
            self.fields = {}
        if self.metadata is None:
            self.metadata = {}
        if self.tags is None:
            self.tags = []
        if not self.id:
            self.id = str(uuid4())

    def to_json(self, indent: int = 2) -> str:
        """
        Convert result to JSON string.

        Args:
            indent: JSON indentation level

        Returns:
            JSON string representation
        """
        data = {
            "id": self.id,
            "fields": self.fields,
            "metadata": self.metadata,
            "tags": self.tags,
        }
        return json.dumps(data, cls=NumpyEncoder, indent=indent)

    def to_record(self) -> Dict[str, Any]:
        """
        Convert to a record suitable for database storage.

        Returns:
            Dictionary with id, type, config, metadata, status, tags
        """
        return {
            "id": self.id,
            "type": self.metadata.get("simulation_type", "unknown"),
            "config": self.metadata.get("config", {}),
            "metadata": {
                k: v for k, v in self.metadata.items()
                if k not in ("config", "simulation_type")
            },
            "status": self.metadata.get("status", "completed"),
            "tags": self.tags,
            "fields": self.fields,
        }

    @classmethod
    def from_json(cls, json_str: str) -> "SimulationResult":
        """
        Create result from JSON string.

        Args:
            json_str: JSON string

        Returns:
            SimulationResult instance
        """
        data = json.loads(json_str, object_hook=numpy_decoder)
        result = cls(
            fields=data.get("fields", {}),
            metadata=data.get("metadata", {}),
            tags=data.get("tags", []),
        )
        # Preserve ID if present
        if "id" in data:
            result.id = data["id"]
        return result

    def save(self, path: Union[str, Path], format: str = "auto") -> None:
        """
        Save result to disk.

        Args:
            path: Output path (without extension)
            format: "auto", "npz", "json", or "both"

        The method saves:
        - .npz file for numpy arrays (efficient binary storage)
        - .json file for metadata and small data
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if format == "auto":
            # Save both formats
            format = "both"

        # Save numpy arrays
        if format in ("npz", "both"):
            npz_path = path.with_suffix(".npz")
            arrays = {}
            for key, value in self.fields.items():
                if isinstance(value, np.ndarray):
                    arrays[key] = value
                elif isinstance(value, (list, tuple)):
                    arrays[key] = np.array(value)

            if arrays:
                np.savez_compressed(npz_path, **arrays)
                logger.debug(f"Saved arrays to {npz_path}")

        # Save JSON
        if format in ("json", "both"):
            json_path = path.with_suffix(".json")

            # For JSON, convert arrays to lists for smaller fields
            json_fields = {}
            for key, value in self.fields.items():
                if isinstance(value, np.ndarray):
                    if value.size < 1000:  # Small arrays as lists
                        json_fields[key] = value.tolist()
                    else:
                        json_fields[key] = {
                            "_type": "ndarray_ref",
                            "shape": value.shape,
                            "dtype": str(value.dtype),
                            "file": str(npz_path.name) if format == "both" else None,
                        }
                else:
                    json_fields[key] = value

            data = {
                "fields": json_fields,
                "metadata": self.metadata,
            }

            with open(json_path, "w") as f:
                json.dump(data, f, cls=NumpyEncoder, indent=2)
            logger.debug(f"Saved metadata to {json_path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "SimulationResult":
        """
        Load result from disk.

        Args:
            path: Path to result file (with or without extension)

        Returns:
            SimulationResult instance
        """
        path = Path(path)

        # Try to find files
        json_path = path.with_suffix(".json")
        npz_path = path.with_suffix(".npz")

        fields = {}
        metadata = {}

        # Load JSON
        if json_path.exists():
            with open(json_path) as f:
                data = json.load(f, object_hook=numpy_decoder)
            fields = data.get("fields", {})
            metadata = data.get("metadata", {})

        # Load NPZ (overrides JSON arrays)
        if npz_path.exists():
            npz_data = np.load(npz_path)
            for key in npz_data.files:
                fields[key] = npz_data[key]

        return cls(fields=fields, metadata=metadata)

    def get_field(self, name: str, default: Any = None) -> Any:
        """Get a field by name."""
        return self.fields.get(name, default)

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get metadata by key."""
        return self.metadata.get(key, default)

    @property
    def field_names(self) -> list:
        """Get list of field names."""
        return list(self.fields.keys())

    @property
    def is_success(self) -> bool:
        """Check if simulation completed successfully."""
        status = self.metadata.get("status", "")
        return status == "completed" and "error" not in self.metadata

    @property
    def error(self) -> Optional[str]:
        """Get error message if any."""
        return self.metadata.get("error")

    @property
    def runtime(self) -> Optional[float]:
        """Get runtime in seconds."""
        return self.metadata.get("runtime_seconds")

    def summary(self) -> str:
        """Get a summary string of the result."""
        lines = [
            f"SimulationResult:",
            f"  ID: {self.id[:8]}...",
            f"  Status: {self.metadata.get('status', 'unknown')}",
            f"  Type: {self.metadata.get('simulation_type', 'unknown')}",
            f"  Runtime: {self.runtime:.3f}s" if self.runtime else "  Runtime: N/A",
            f"  Fields: {', '.join(self.field_names) or 'none'}",
        ]

        if self.tags:
            lines.append(f"  Tags: {', '.join(self.tags)}")

        if self.error:
            lines.append(f"  Error: {self.error}")

        # Field shapes
        for name, value in self.fields.items():
            if isinstance(value, np.ndarray):
                lines.append(f"    {name}: shape={value.shape}, dtype={value.dtype}")

        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"SimulationResult(id={self.id[:8]}..., fields={self.field_names}, status={self.metadata.get('status')})"
