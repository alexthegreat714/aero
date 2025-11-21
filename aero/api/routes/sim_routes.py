"""
Simulation API routes for Aero Agent.

Provides endpoints for running simulations.
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter()


class SimulationConfig(BaseModel):
    """Configuration for a simulation run."""

    solver: str = "fd"  # fd, ns, pinn
    nx: int = 50
    ny: int = 50
    max_iterations: int = 1000
    convergence_threshold: float = 1e-6
    parameters: Optional[dict] = None


class LaplaceConfig(BaseModel):
    """Configuration for Laplace equation simulation."""

    nx: int = 50
    ny: int = 50
    left_bc: float = 0.0
    right_bc: float = 100.0
    top_bc: float = 0.0
    bottom_bc: float = 0.0


@router.get("/")
async def sim_status():
    """Get simulation module status."""
    return {
        "status": "ok",
        "module": "simulation",
        "available_solvers": ["fd", "laplace", "heat", "ns", "pinn"],
        "message": "Simulation module is operational (stub)",
    }


@router.get("/solvers")
async def list_solvers():
    """List available simulation solvers."""
    return {
        "status": "ok",
        "solvers": [
            {
                "name": "fd",
                "description": "Generic finite difference solver",
                "status": "available",
            },
            {
                "name": "laplace",
                "description": "Laplace equation solver",
                "status": "available",
            },
            {
                "name": "heat",
                "description": "Heat equation solver",
                "status": "available",
            },
            {
                "name": "ns",
                "description": "Navier-Stokes solver",
                "status": "stub",
            },
            {
                "name": "pinn",
                "description": "Physics-Informed Neural Network",
                "status": "stub",
            },
        ],
    }


@router.post("/run")
async def run_simulation(config: SimulationConfig):
    """
    Run a simulation with the specified configuration.

    Returns simulation results.
    """
    logger.info(f"Running simulation: solver={config.solver}")

    return {
        "status": "completed",
        "solver": config.solver,
        "iterations": 0,
        "converged": True,
        "residual": 0.0,
        "elapsed_time": 0.0,
        "message": "Simulation completed (stub)",
    }


@router.post("/laplace")
async def run_laplace(config: LaplaceConfig):
    """
    Run a Laplace equation simulation.

    Returns the solution field.
    """
    logger.info(f"Running Laplace simulation: {config.nx}x{config.ny}")

    return {
        "status": "completed",
        "solver": "laplace",
        "grid_size": [config.nx, config.ny],
        "iterations": 0,
        "converged": True,
        "solution": None,  # Would contain actual solution array
        "message": "Laplace simulation completed (stub)",
    }


@router.get("/jobs")
async def list_jobs():
    """List active and recent simulation jobs."""
    return {
        "status": "ok",
        "active_jobs": [],
        "recent_jobs": [],
        "message": "Job listing (stub)",
    }


@router.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    """Get status of a specific simulation job."""
    return {
        "status": "ok",
        "job_id": job_id,
        "state": "unknown",
        "progress": 0.0,
        "message": "Job status (stub)",
    }


@router.delete("/jobs/{job_id}")
async def cancel_job(job_id: str):
    """Cancel a running simulation job."""
    return {
        "status": "cancelled",
        "job_id": job_id,
        "message": "Job cancelled (stub)",
    }
