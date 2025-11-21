"""
FastAPI server for Aero Agent.

Provides the main API server with all routes.
"""

import logging
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from aero.config.loader import get_config

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.

    Returns:
        Configured FastAPI app
    """
    config = get_config()

    app = FastAPI(
        title="Aero Agent API",
        description="REST API for Aero Agent - Aerodynamics Research Assistant",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure as needed for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register routes
    _register_routes(app)

    # Add startup/shutdown events
    @app.on_event("startup")
    async def startup_event():
        logger.info("Aero API server starting up")

    @app.on_event("shutdown")
    async def shutdown_event():
        logger.info("Aero API server shutting down")

    return app


def _register_routes(app: FastAPI) -> None:
    """Register all API routes."""
    from aero.api.routes import (
        rag_routes,
        sim_routes,
        ocr_routes,
        web_routes,
        agent_routes,
        experiment_routes,
    )

    # Include route modules
    app.include_router(rag_routes.router, prefix="/api/rag", tags=["RAG"])
    app.include_router(sim_routes.router, prefix="/api/sim", tags=["Simulation"])
    app.include_router(ocr_routes.router, prefix="/api/ocr", tags=["OCR"])
    app.include_router(web_routes.router, prefix="/api/web", tags=["Web"])
    app.include_router(agent_routes.router, prefix="/api/agent", tags=["Agent"])
    app.include_router(experiment_routes.router, prefix="/api/experiments", tags=["Experiments"])

    # Root endpoint
    @app.get("/", tags=["Health"])
    async def root():
        """Root endpoint - API health check."""
        return {
            "status": "ok",
            "service": "Aero Agent API",
            "version": "0.1.0",
        }

    # Health check
    @app.get("/health", tags=["Health"])
    async def health():
        """Health check endpoint."""
        return {"status": "healthy"}

    logger.debug("API routes registered")


def run_server(
    host: Optional[str] = None,
    port: Optional[int] = None,
    reload: bool = False,
) -> None:
    """
    Run the API server.

    Args:
        host: Server host (default from config)
        port: Server port (default from config)
        reload: Enable auto-reload for development
    """
    import uvicorn

    config = get_config()

    host = host or config.get("server.host", "127.0.0.1")
    port = port or config.get("server.port", 8000)

    logger.info(f"Starting Aero API server at http://{host}:{port}")

    uvicorn.run(
        "aero.api.server:create_app",
        host=host,
        port=port,
        reload=reload,
        factory=True,
    )
