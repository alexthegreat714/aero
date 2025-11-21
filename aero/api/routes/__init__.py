"""
API route modules for Aero Agent.
"""

from aero.api.routes import rag_routes
from aero.api.routes import sim_routes
from aero.api.routes import ocr_routes
from aero.api.routes import web_routes
from aero.api.routes import agent_routes

__all__ = [
    "rag_routes",
    "sim_routes",
    "ocr_routes",
    "web_routes",
    "agent_routes",
]
