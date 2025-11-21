"""
API module for Aero Agent.

Provides a FastAPI-based REST API server.
"""

from aero.api.server import create_app, run_server

__all__ = [
    "create_app",
    "run_server",
]
