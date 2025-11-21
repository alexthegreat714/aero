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


def register_embedder(name: str, embedder: Any, metadata: Optional[dict] = None) -> None:
    """Convenience function to register an embedder."""
    get_registry_manager().embedders.register(name, embedder, metadata)


def register_agent(name: str, agent: Any, metadata: Optional[dict] = None) -> None:
    """Convenience function to register an agent."""
    get_registry_manager().agents.register(name, agent, metadata)


# -----------------------------------------------------------------------------
# RAG Component Registration
# -----------------------------------------------------------------------------


def register_rag_components() -> None:
    """
    Register all RAG components with the registry.

    This includes:
    - Embedders (Ollama, SentenceTransformers, Hash)
    - Document readers
    - Vector stores
    """
    manager = get_registry_manager()

    # Create RAG-specific registries
    manager.create_registry("rag_stores")
    manager.create_registry("rag_readers")
    manager.create_registry("rag_chunkers")
    manager.create_registry("rag_preprocessors")

    # Register embedders
    try:
        from aero.rag.embedder import (
            OllamaEmbedder,
            SentenceTransformerEmbedder,
            HashEmbedder,
            AutoEmbedder,
        )

        manager.embedders.register("ollama", OllamaEmbedder, {
            "description": "Ollama-based embedder using llama.cpp",
            "default_model": "nomic-embed-text",
        })
        manager.embedders.register("sentence_transformers", SentenceTransformerEmbedder, {
            "description": "SentenceTransformers embedder",
            "default_model": "all-MiniLM-L6-v2",
        })
        manager.embedders.register("hash", HashEmbedder, {
            "description": "Hash-based fallback embedder",
        })
        manager.embedders.register("auto", AutoEmbedder, {
            "description": "Auto-selecting embedder",
        })

        logger.info("Registered RAG embedders")

    except ImportError as e:
        logger.warning(f"Could not register RAG embedders: {e}")

    # Register vector stores
    try:
        from aero.rag.store import (
            DuckDBVectorStore,
            SQLiteVectorStore,
            InMemoryVectorStore,
        )

        stores = manager.get_registry("rag_stores")
        stores.register("duckdb", DuckDBVectorStore, {
            "description": "DuckDB-based vector store",
        })
        stores.register("sqlite", SQLiteVectorStore, {
            "description": "SQLite-based vector store",
        })
        stores.register("memory", InMemoryVectorStore, {
            "description": "In-memory vector store",
        })

        logger.info("Registered RAG stores")

    except ImportError as e:
        logger.warning(f"Could not register RAG stores: {e}")

    # Register document readers
    try:
        from aero.rag.reader import (
            TextReader,
            PDFReader,
            DocxReader,
            HTMLReader,
            ImageReader,
            DocumentReader,
        )

        readers = manager.get_registry("rag_readers")
        readers.register("text", TextReader, {
            "formats": [".txt", ".md", ".rst", ".csv", ".json", ".yaml"],
        })
        readers.register("pdf", PDFReader, {
            "formats": [".pdf"],
        })
        readers.register("docx", DocxReader, {
            "formats": [".docx"],
        })
        readers.register("html", HTMLReader, {
            "formats": [".html", ".htm"],
        })
        readers.register("image", ImageReader, {
            "formats": [".jpg", ".jpeg", ".png", ".tiff", ".bmp", ".gif"],
        })
        readers.register("auto", DocumentReader, {
            "description": "Auto-selecting document reader",
        })

        logger.info("Registered RAG readers")

    except ImportError as e:
        logger.warning(f"Could not register RAG readers: {e}")


def get_rag_store(store_type: str = "auto", **kwargs) -> Any:
    """
    Get a RAG store instance.

    Args:
        store_type: Type of store ("duckdb", "sqlite", "memory", "auto")
        **kwargs: Arguments passed to store constructor

    Returns:
        Vector store instance
    """
    if store_type == "auto":
        from aero.rag.store import create_vector_store
        return create_vector_store(**kwargs)

    manager = get_registry_manager()
    stores = manager.get_registry("rag_stores")
    store_class = stores.get(store_type)
    return store_class(**kwargs)


def get_embedder(embedder_type: str = "auto", **kwargs) -> Any:
    """
    Get an embedder instance.

    Args:
        embedder_type: Type of embedder ("ollama", "sentence_transformers", "hash", "auto")
        **kwargs: Arguments passed to embedder constructor

    Returns:
        Embedder instance
    """
    manager = get_registry_manager()
    embedder_class = manager.embedders.get(embedder_type)
    return embedder_class(**kwargs)


def get_document_reader(reader_type: str = "auto", **kwargs) -> Any:
    """
    Get a document reader instance.

    Args:
        reader_type: Type of reader ("text", "pdf", "docx", "html", "image", "auto")
        **kwargs: Arguments passed to reader constructor

    Returns:
        Document reader instance
    """
    manager = get_registry_manager()
    readers = manager.get_registry("rag_readers")
    reader_class = readers.get(reader_type)
    return reader_class(**kwargs)


# -----------------------------------------------------------------------------
# Agent Bus and Registry Access
# -----------------------------------------------------------------------------


def get_agent_bus() -> "AgentBus":
    """
    Get the global AgentBus instance.

    Returns:
        AgentBus instance
    """
    from aero.agents.bus import get_default_bus
    return get_default_bus()


def get_agent_registry() -> "AgentRegistry":
    """
    Get the global AgentRegistry instance.

    Returns:
        AgentRegistry instance
    """
    from aero.agents.registry import get_default_registry
    return get_default_registry()


def initialize_agent_system(
    bus_enabled: bool = True,
    log_limit: int = 1000,
) -> tuple:
    """
    Initialize the agent system with bus and registry.

    Args:
        bus_enabled: Whether the message bus is enabled
        log_limit: Maximum messages to keep in bus log

    Returns:
        Tuple of (AgentBus, AgentRegistry)
    """
    from aero.agents.bus import AgentBus, set_default_bus
    from aero.agents.registry import AgentRegistry, set_default_registry

    # Create and set bus
    bus = AgentBus(enabled=bus_enabled, log_limit=log_limit)
    set_default_bus(bus)

    # Create and set registry
    registry = AgentRegistry()
    set_default_registry(registry)

    logger.info(f"Agent system initialized (bus_enabled={bus_enabled})")

    return bus, registry

