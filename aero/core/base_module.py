"""
Base module class for Aero Agent framework.

Provides a foundation for pluggable modules that can be registered
and managed by the system.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Optional

from aero.config.loader import get_config, AeroConfig


class BaseModule(ABC):
    """
    Base class for all Aero modules.

    Modules are pluggable components that provide specific functionality
    such as OCR, simulation, or web search capabilities.

    Example:
        class MyModule(BaseModule):
            @property
            def name(self) -> str:
                return "my_module"

            def initialize(self) -> None:
                # Setup code
                pass

            def shutdown(self) -> None:
                # Cleanup code
                pass
    """

    def __init__(self, config: Optional[AeroConfig] = None):
        """
        Initialize the module.

        Args:
            config: Configuration object (uses global config if not provided)
        """
        self.config = config or get_config()
        self._logger = logging.getLogger(f"aero.module.{self.name}")
        self._initialized = False

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Unique name for this module.

        Returns:
            Module name string
        """
        raise NotImplementedError

    @property
    def version(self) -> str:
        """
        Module version string.

        Returns:
            Version string (default: "0.1.0")
        """
        return "0.1.0"

    @property
    def description(self) -> str:
        """
        Human-readable description of the module.

        Returns:
            Description string
        """
        return f"Aero module: {self.name}"

    @property
    def is_initialized(self) -> bool:
        """Check if the module has been initialized."""
        return self._initialized

    # -------------------------------------------------------------------------
    # Lifecycle Methods
    # -------------------------------------------------------------------------

    @abstractmethod
    def initialize(self) -> None:
        """
        Initialize the module.

        Called once when the module is first loaded. Subclasses should
        perform any necessary setup here.
        """
        raise NotImplementedError

    @abstractmethod
    def shutdown(self) -> None:
        """
        Shutdown the module.

        Called when the module is being unloaded. Subclasses should
        perform any necessary cleanup here.
        """
        raise NotImplementedError

    def safe_initialize(self) -> bool:
        """
        Safely initialize the module, catching any errors.

        Returns:
            True if initialization succeeded, False otherwise
        """
        if self._initialized:
            return True

        try:
            self.initialize()
            self._initialized = True
            self._logger.info(f"Module '{self.name}' initialized successfully")
            return True
        except Exception as e:
            self._logger.error(f"Failed to initialize module '{self.name}': {e}")
            return False

    def safe_shutdown(self) -> bool:
        """
        Safely shutdown the module, catching any errors.

        Returns:
            True if shutdown succeeded, False otherwise
        """
        if not self._initialized:
            return True

        try:
            self.shutdown()
            self._initialized = False
            self._logger.info(f"Module '{self.name}' shutdown successfully")
            return True
        except Exception as e:
            self._logger.error(f"Failed to shutdown module '{self.name}': {e}")
            return False

    # -------------------------------------------------------------------------
    # Logging Helpers
    # -------------------------------------------------------------------------

    def log_debug(self, message: str) -> None:
        """Log a debug message."""
        self._logger.debug(message)

    def log_info(self, message: str) -> None:
        """Log an info message."""
        self._logger.info(message)

    def log_warning(self, message: str) -> None:
        """Log a warning message."""
        self._logger.warning(message)

    def log_error(self, message: str) -> None:
        """Log an error message."""
        self._logger.error(message)

    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------

    def get_info(self) -> dict:
        """
        Get module information as a dictionary.

        Returns:
            Dictionary with module info
        """
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "initialized": self._initialized,
        }

    def __repr__(self) -> str:
        status = "initialized" if self._initialized else "not initialized"
        return f"<{self.__class__.__name__}(name='{self.name}', {status})>"
