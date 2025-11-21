"""
Aero Agent - An intelligent agent framework for aerodynamics research.

This package provides:
- Configuration management
- Core agent infrastructure
- RAG (Retrieval-Augmented Generation) capabilities
- Simulation modules (CFD, PINN, FD solvers)
- OCR backends (Tesseract, DeepSeek, EasyOCR)
- Web search and scraping utilities
- Experiment management (cameras, sensors)
- REST API for external integration
"""

__version__ = "0.1.0"
__author__ = "Aero Team"

from aero.config.loader import load_config, get_config
from aero.core.base_agent import BaseAgent
from aero.core.registry import Registry

__all__ = [
    "load_config",
    "get_config",
    "BaseAgent",
    "Registry",
    "__version__",
]
