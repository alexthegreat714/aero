"""
Registry system for Aero Agent framework.

Provides centralized registration and discovery of:
- OCR backends
- Models
- Simulation modules
- Web search modules
"""

import logging
from typing import Any, Callable, Optional, Type, TypeVar
from dataclasses import dataclass, field
import threading

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class RegistryEntry:
    """Represents a registered item."""

    name: str
    item: Any
    metadata: dict = field(default_factory=dict)
    enabled: bool = True


class Registry:
    """
    Generic registry for managing named items.

    Supports:
    - Registration with metadata
    - Lazy initialization
    - Enable/disable items
    - Category-based organization

    Example:
        registry = Registry("ocr_backends")
        registry.register("tesseract", TesseractBackend, {"version": "5.0"})

        backend_class = registry.get("tesseract")
        backend = backend_class()
    """

    def __init__(self, name: str):
        """
        Initialize the registry.

        Args:
            name: Name of this registry (e.g., "ocr_backends")
        """
        self.name = name
        self._entries: dict[str, RegistryEntry] = {}
        self._lock = threading.RLock()
        self._logger = logging.getLogger(f"aero.registry.{name}")

    def register(
        self,
        name: str,
        item: Any,
        metadata: Optional[dict] = None,
        replace: bool = False
    ) -> None:
        """
        Register an item.

        Args:
            name: Unique name for the item
            item: The item to register (class, function, object, etc.)
            metadata: Optional metadata about the item
            replace: If True, replace existing entry; if False, raise error
        """
        with self._lock:
            if name in self._entries and not replace:
                raise ValueError(f"Item '{name}' already registered in {self.name}")

            entry = RegistryEntry(
                name=name,
                item=item,
                metadata=metadata or {},
            )
            self._entries[name] = entry
            self._logger.debug(f"Registered '{name}' in {self.name}")

    def unregister(self, name: str) -> bool:
        """
        Unregister an item.

        Args:
            name: Name of item to unregister

        Returns:
            True if item was removed, False if not found
        """
        with self._lock:
            if name in self._entries:
                del self._entries[name]
                self._logger.debug(f"Unregistered '{name}' from {self.name}")
                return True
            return False

    def get(self, name: str) -> Any:
        """
        Get a registered item by name.

        Args:
            name: Name of the item

        Returns:
            The registered item

        Raises:
            KeyError: If item not found
        """
        with self._lock:
            if name not in self._entries:
                raise KeyError(f"Item '{name}' not found in {self.name}")
            return self._entries[name].item

    def get_or_none(self, name: str) -> Optional[Any]:
        """
        Get a registered item or None if not found.

        Args:
            name: Name of the item

        Returns:
            The registered item or None
        """
        try:
            return self.get(name)
        except KeyError:
            return None

    def get_entry(self, name: str) -> Optional[RegistryEntry]:
        """
        Get the full registry entry.

        Args:
            name: Name of the item

        Returns:
            RegistryEntry or None
        """
        with self._lock:
            return self._entries.get(name)

    def has(self, name: str) -> bool:
        """Check if an item is registered."""
        with self._lock:
            return name in self._entries

    def list_names(self, enabled_only: bool = True) -> list[str]:
        """
        List all registered item names.

        Args:
            enabled_only: If True, only return enabled items

        Returns:
            List of item names
        """
        with self._lock:
            if enabled_only:
                return [
                    name for name, entry in self._entries.items()
                    if entry.enabled
                ]
            return list(self._entries.keys())

    def list_entries(self, enabled_only: bool = True) -> list[RegistryEntry]:
        """
        List all registry entries.

        Args:
            enabled_only: If True, only return enabled entries

        Returns:
            List of RegistryEntry objects
        """
        with self._lock:
            if enabled_only:
                return [e for e in self._entries.values() if e.enabled]
            return list(self._entries.values())

    def enable(self, name: str) -> None:
        """Enable a registered item."""
        with self._lock:
            if name in self._entries:
                self._entries[name].enabled = True

    def disable(self, name: str) -> None:
        """Disable a registered item."""
        with self._lock:
            if name in self._entries:
                self._entries[name].enabled = False

    def clear(self) -> None:
        """Remove all registered items."""
        with self._lock:
            self._entries.clear()

    @property
    def count(self) -> int:
        """Get the number of registered items."""
        with self._lock:
            return len(self._entries)

    def __len__(self) -> int:
        return self.count

    def __contains__(self, name: str) -> bool:
        return self.has(name)

    def __iter__(self):
        return iter(self.list_names())


class RegistryManager:
    """
    Manages multiple registries for different component types.

    Provides centralized access to:
    - OCR backends
    - Models
    - Simulation modules
    - Web search modules
    """

    def __init__(self):
        """Initialize the registry manager."""
        self._registries: dict[str, Registry] = {}
        self._lock = threading.RLock()

        # Create default registries
        self._create_default_registries()

    def _create_default_registries(self) -> None:
        """Create the default component registries."""
        self.create_registry("ocr_backends")
        self.create_registry("models")
        self.create_registry("simulation_modules")
        self.create_registry("web_search_modules")
        self.create_registry("agents")
        self.create_registry("embedders")

    def create_registry(self, name: str) -> Registry:
        """
        Create a new registry.

        Args:
            name: Name for the registry

        Returns:
            The created Registry
        """
        with self._lock:
            if name not in self._registries:
                self._registries[name] = Registry(name)
            return self._registries[name]

    def get_registry(self, name: str) -> Registry:
        """
        Get a registry by name.

        Args:
            name: Registry name

        Returns:
            The Registry

        Raises:
            KeyError: If registry not found
        """
        with self._lock:
            if name not in self._registries:
                raise KeyError(f"Registry '{name}' not found")
            return self._registries[name]

    def list_registries(self) -> list[str]:
        """List all registry names."""
        with self._lock:
            return list(self._registries.keys())

    # -------------------------------------------------------------------------
    # Convenience Properties
    # -------------------------------------------------------------------------

    @property
    def ocr_backends(self) -> Registry:
        """Get the OCR backends registry."""
        return self.get_registry("ocr_backends")

    @property
    def models(self) -> Registry:
        """Get the models registry."""
        return self.get_registry("models")

    @property
    def simulation_modules(self) -> Registry:
        """Get the simulation modules registry."""
        return self.get_registry("simulation_modules")

    @property
    def web_search_modules(self) -> Registry:
        """Get the web search modules registry."""
        return self.get_registry("web_search_modules")

    @property
    def agents(self) -> Registry:
        """Get the agents registry."""
        return self.get_registry("agents")

    @property
    def embedders(self) -> Registry:
        """Get the embedders registry."""
        return self.get_registry("embedders")

    def get_summary(self) -> dict:
        """Get a summary of all registries."""
        with self._lock:
            return {
                name: {
                    "count": registry.count,
                    "items": registry.list_names(),
                }
                for name, registry in self._registries.items()
            }


# Global registry manager instance
_global_manager: Optional[RegistryManager] = None


def get_registry_manager() -> RegistryManager:
    """Get the global registry manager instance."""
    global _global_manager
    if _global_manager is None:
        _global_manager = RegistryManager()
    return _global_manager


def register_ocr_backend(name: str, backend: Any, metadata: Optional[dict] = None) -> None:
    """Convenience function to register an OCR backend."""
    get_registry_manager().ocr_backends.register(name, backend, metadata)


def register_model(name: str, model: Any, metadata: Optional[dict] = None) -> None:
    """Convenience function to register a model."""
    get_registry_manager().models.register(name, model, metadata)


def register_simulation(name: str, sim: Any, metadata: Optional[dict] = None) -> None:
    """Convenience function to register a simulation module."""
    get_registry_manager().simulation_modules.register(name, sim, metadata)


def register_web_search(name: str, search: Any, metadata: Optional[dict] = None) -> None:
    """Convenience function to register a web search module."""
    get_registry_manager().web_search_modules.register(name, search, metadata)
