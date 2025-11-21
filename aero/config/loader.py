"""
Configuration loader for Aero Agent.

Loads YAML defaults and allows override from environment variables.
Environment variables should be prefixed with AERO_ and use underscores
for nested keys (e.g., AERO_SERVER_PORT for server.port).
"""

import os
import logging
from pathlib import Path
from typing import Any, Optional
from dataclasses import dataclass, field

import yaml

logger = logging.getLogger(__name__)

# Global configuration instance
_config: Optional["AeroConfig"] = None


@dataclass
class AeroConfig:
    """
    Main configuration container for Aero Agent.

    Provides dict-like access to configuration values with support
    for nested keys using dot notation.
    """

    _data: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self._data:
            self._data = {}

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value using dot notation.

        Args:
            key: Configuration key (e.g., "server.port")
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        keys = key.split(".")
        value = self._data

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value using dot notation.

        Args:
            key: Configuration key (e.g., "server.port")
            value: Value to set
        """
        keys = key.split(".")
        data = self._data

        for k in keys[:-1]:
            if k not in data:
                data[k] = {}
            data = data[k]

        data[keys[-1]] = value

    def __getitem__(self, key: str) -> Any:
        return self.get(key)

    def __setitem__(self, key: str, value: Any) -> None:
        self.set(key, value)

    def to_dict(self) -> dict:
        """Return the raw configuration dictionary."""
        return self._data.copy()

    @property
    def debug(self) -> bool:
        """Check if debug mode is enabled."""
        return self.get("app.debug", False)

    @property
    def log_level(self) -> str:
        """Get the logging level."""
        return self.get("app.log_level", "INFO")


def _get_defaults_path() -> Path:
    """Get the path to the defaults.yaml file."""
    return Path(__file__).parent / "defaults.yaml"


def _load_yaml(path: Path) -> dict:
    """Load a YAML file and return its contents."""
    if not path.exists():
        logger.warning(f"Configuration file not found: {path}")
        return {}

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _apply_env_overrides(config: dict, prefix: str = "AERO") -> dict:
    """
    Apply environment variable overrides to configuration.

    Environment variables should be prefixed with AERO_ and use
    underscores for nested keys. Values are automatically converted
    to appropriate types (int, float, bool, or str).

    Args:
        config: Configuration dictionary to update
        prefix: Environment variable prefix

    Returns:
        Updated configuration dictionary
    """
    prefix = f"{prefix}_"

    for key, value in os.environ.items():
        if not key.startswith(prefix):
            continue

        # Convert AERO_SERVER_PORT to server.port
        config_key = key[len(prefix):].lower().replace("_", ".")

        # Type conversion
        converted_value: Any = value

        # Try boolean
        if value.lower() in ("true", "yes", "1", "on"):
            converted_value = True
        elif value.lower() in ("false", "no", "0", "off"):
            converted_value = False
        else:
            # Try numeric
            try:
                if "." in value:
                    converted_value = float(value)
                else:
                    converted_value = int(value)
            except ValueError:
                pass  # Keep as string

        # Apply to config
        _set_nested(config, config_key, converted_value)
        logger.debug(f"Applied env override: {config_key} = {converted_value}")

    return config


def _set_nested(data: dict, key: str, value: Any) -> None:
    """Set a nested dictionary value using dot notation."""
    keys = key.split(".")

    for k in keys[:-1]:
        if k not in data:
            data[k] = {}
        data = data[k]

    data[keys[-1]] = value


def load_config(
    config_path: Optional[Path] = None,
    apply_env: bool = True
) -> AeroConfig:
    """
    Load configuration from YAML file with optional environment overrides.

    Args:
        config_path: Optional path to custom config file
        apply_env: Whether to apply environment variable overrides

    Returns:
        AeroConfig instance
    """
    global _config

    # Load defaults
    defaults = _load_yaml(_get_defaults_path())

    # Load custom config if provided
    if config_path:
        custom = _load_yaml(Path(config_path))
        defaults = _deep_merge(defaults, custom)

    # Apply environment overrides
    if apply_env:
        defaults = _apply_env_overrides(defaults)

    # Create and store config
    _config = AeroConfig(_data=defaults)

    # Configure logging
    _configure_logging(_config)

    logger.info("Configuration loaded successfully")
    return _config


def _deep_merge(base: dict, override: dict) -> dict:
    """Deep merge two dictionaries."""
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value

    return result


def _configure_logging(config: AeroConfig) -> None:
    """Configure logging based on configuration."""
    log_format = config.get("logging.format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    log_level = config.get("app.log_level", "INFO")

    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format=log_format,
    )

    # File logging if configured
    log_file = config.get("logging.file")
    if log_file:
        from logging.handlers import RotatingFileHandler

        handler = RotatingFileHandler(
            log_file,
            maxBytes=config.get("logging.max_bytes", 10485760),
            backupCount=config.get("logging.backup_count", 5),
        )
        handler.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(handler)


def get_config() -> AeroConfig:
    """
    Get the global configuration instance.

    Loads default configuration if not already loaded.

    Returns:
        AeroConfig instance
    """
    global _config

    if _config is None:
        _config = load_config()

    return _config
