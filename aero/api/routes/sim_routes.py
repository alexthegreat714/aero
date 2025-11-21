"""
Simulation API routes for Aero Agent.

Provides endpoints for running simulations:
- POST /sim/run - Run generic simulation
- POST /sim/heat_1d - 1D heat equation
- POST /sim/laplace_2d - 2D Laplace equation
- POST /sim/ns_stub - Navier-Stokes stub
- POST /sim/pinn/train - PINN training
- GET /sim/jobs - List jobs
- GET /sim/job/{id} - Get job status
"""

import logging
from typing import List, Optional

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter()


# =============================================================================
# Request/Response Models
# =============================================================================


class SimulationRunConfig(BaseModel):
    """Configuration for a simulation run."""

    solver: str = Field(default="fd", description="Solver type: fd, laplace, heat, ns, pinn")
    nx: int = Field(default=64, ge=2, le=1000)
    ny: int = Field(default=64, ge=2, le=1000)
    max_iterations: int = Field(default=1000, ge=1, le=100000)
    convergence_threshold: float = Field(default=1e-6, gt=0)
    parameters: Optional[dict] = None


class Heat1DConfig(BaseModel):
    """Configuration for 1D heat equation."""

    nx: int = Field(default=101, ge=3, le=10000)
    alpha: float = Field(default=0.01, gt=0, description="Thermal diffusivity")
    dx: float = Field(default=0.01, gt=0)
    dt: float = Field(default=0.0001, gt=0)
    steps: int = Field(default=1000, ge=1, le=1000000)
    boundary_left: Optional[float] = Field(default=0.0)
    boundary_right: Optional[float] = Field(default=0.0)
    initial_condition: str = Field(default="gaussian", description="gaussian, step, or custom")


class Laplace2DConfig(BaseModel):
    """Configuration for 2D Laplace equation."""

    nx: int = Field(default=50, ge=3, le=500)
    ny: int = Field(default=50, ge=3, le=500)
    left_bc: float = Field(default=0.0)
    right_bc: float = Field(default=0.0)
    top_bc: float = Field(default=100.0)
    bottom_bc: float = Field(default=0.0)
    tolerance: float = Field(default=1e-6, gt=0)
    max_iterations: int = Field(default=10000, ge=1)
    omega: float = Field(default=1.0, gt=0, le=2.0, description="SOR relaxation factor")


class NSStubConfig(BaseModel):
    """Configuration for Navier-Stokes stub."""

    nx: int = Field(default=100, ge=10, le=1000)
    ny: int = Field(default=100, ge=10, le=1000)
    Re: float = Field(default=100.0, gt=0, description="Reynolds number")
    lx: float = Field(default=1.0, gt=0)
    ly: float = Field(default=1.0, gt=0)


class PINNTrainConfig(BaseModel):
    """Configuration for PINN training."""

    layers: List[int] = Field(default=[2, 64, 64, 64, 1])
    epochs: int = Field(default=1000, ge=1, le=100000)
    learning_rate: float = Field(default=0.001, gt=0)
    activation: str = Field(default="tanh")
    n_collocation: int = Field(default=1000, ge=10)
    n_boundary: int = Field(default=100, ge=10)


# =============================================================================
# Helper Functions
# =============================================================================


def _get_scheduler():
    """Get the simulation scheduler."""
    from aero.sim.scheduler import get_scheduler
    return get_scheduler()


# =============================================================================
# Routes
# =============================================================================


@router.get("/")
async def sim_status():
    """Get simulation module status."""
    try:
        from aero.sim.pinn.pinn_trainer import check_pytorch_cuda

        cuda_info = check_pytorch_cuda()

        return {
            "status": "ok",
            "module": "simulation",
            "available_solvers": ["heat_1d", "laplace_2d", "ns_stub", "pinn"],
            "pytorch_available": cuda_info["pytorch_available"],
            "cuda_available": cuda_info["cuda_available"],
            "cuda_device": cuda_info.get("cuda_device"),
            "numerics_backend": "numpy",
        }
    except Exception as e:
        return {
            "status": "ok",
            "module": "simulation",
            "available_solvers": ["heat_1d", "laplace_2d", "ns_stub"],
            "error": str(e),
        }


@router.get("/solvers")
async def list_solvers():
    """List available simulation solvers."""
    return {
        "status": "ok",
        "solvers": [
            {
                "name": "heat_1d",
                "description": "1D Heat equation (explicit finite difference)",
                "status": "available",
            },
            {
                "name": "laplace_2d",
                "description": "2D Laplace equation (Gauss-Seidel)",
                "status": "available",
            },
            {
                "name": "ns_stub",
                "description": "2D Navier-Stokes (projection method stub)",
                "status": "stub",
            },
            {
                "name": "pinn",
                "description": "Physics-Informed Neural Network",
                "status": "available",
            },
        ],
    }


@router.post("/run")
async def run_simulation(config: SimulationRunConfig):
    """
    Run a simulation with the specified configuration.

    Returns simulation results.
    """
    logger.info(f"Running simulation: solver={config.solver}")

    try:
        if config.solver in ("laplace", "laplace_2d"):
            return await run_laplace_2d(Laplace2DConfig(
                nx=config.nx,
                ny=config.ny,
                tolerance=config.convergence_threshold,
                max_iterations=config.max_iterations,
            ))
        elif config.solver in ("heat", "heat_1d"):
            return await run_heat_1d(Heat1DConfig(
                nx=config.nx,
                steps=config.max_iterations,
            ))
        else:
            raise HTTPException(status_code=400, detail=f"Unknown solver: {config.solver}")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/heat_1d")
async def run_heat_1d(config: Heat1DConfig):
    """
    Solve 1D heat equation: du/dt = alpha * d²u/dx²

    Returns the final temperature distribution.
    """
    logger.info(f"Running heat_1d: nx={config.nx}, steps={config.steps}")

    try:
        from aero.sim.numerics.fd_solver import solve_heat_1d

        # Create initial condition
        x = np.linspace(0, (config.nx - 1) * config.dx, config.nx)

        if config.initial_condition == "gaussian":
            x_mid = x[-1] / 2
            u0 = np.exp(-100 * (x - x_mid) ** 2)
        elif config.initial_condition == "step":
            u0 = np.where(x < x[-1] / 2, 1.0, 0.0)
        else:
            u0 = np.zeros(config.nx)
            u0[config.nx // 2] = 1.0

        # Solve
        u_final, metadata = solve_heat_1d(
            u0=u0,
            alpha=config.alpha,
            dx=config.dx,
            dt=config.dt,
            steps=config.steps,
            boundary_left=config.boundary_left,
            boundary_right=config.boundary_right,
        )

        return {
            "status": "completed",
            "solver": "heat_1d",
            "solution": u_final.tolist(),
            "x": x.tolist(),
            **metadata,
        }

    except Exception as e:
        logger.error(f"Heat 1D failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/laplace_2d")
async def run_laplace_2d(config: Laplace2DConfig):
    """
    Solve 2D Laplace equation: d²u/dx² + d²u/dy² = 0

    Returns the steady-state solution.
    """
    logger.info(f"Running laplace_2d: {config.nx}x{config.ny}")

    try:
        from aero.sim.numerics.fd_solver import solve_laplace_2d

        # Set up initial grid with boundary conditions
        u0 = np.zeros((config.nx, config.ny))
        u0[0, :] = config.left_bc
        u0[-1, :] = config.right_bc
        u0[:, 0] = config.bottom_bc
        u0[:, -1] = config.top_bc

        # Solve
        solution, metadata = solve_laplace_2d(
            initial_grid=u0,
            tol=config.tolerance,
            max_iterations=config.max_iterations,
            omega=config.omega,
        )

        return {
            "status": "completed",
            "solver": "laplace_2d",
            "solution": solution.tolist(),
            **metadata,
        }

    except Exception as e:
        logger.error(f"Laplace 2D failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ns_stub")
async def run_ns_stub(config: NSStubConfig):
    """
    Run Navier-Stokes stub simulation.

    Note: This is a placeholder - returns zero fields.
    """
    logger.info(f"Running NS stub: {config.nx}x{config.ny}, Re={config.Re}")

    try:
        from aero.sim.cfd.ns_solver_stub import NavierStokes2DStub

        solver = NavierStokes2DStub(
            nx=config.nx,
            ny=config.ny,
            lx=config.lx,
            ly=config.ly,
            Re=config.Re,
        )

        result = solver.run()

        return {
            "status": "completed",
            "solver": "navier_stokes_2d_stub",
            "u": result.data["u"].tolist() if "u" in result.data else None,
            "v": result.data["v"].tolist() if "v" in result.data else None,
            "p": result.data["p"].tolist() if "p" in result.data else None,
            **result.metadata,
        }

    except Exception as e:
        logger.error(f"NS stub failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/pinn/train")
async def train_pinn(config: PINNTrainConfig):
    """
    Train a Physics-Informed Neural Network.

    Returns training results and model info.
    """
    logger.info(f"Training PINN: layers={config.layers}, epochs={config.epochs}")

    try:
        from aero.sim.pinn.pinn_model import PINNModel, TORCH_AVAILABLE
        from aero.sim.pinn.pinn_trainer import PINNTrainer, TrainingConfig

        if not TORCH_AVAILABLE:
            raise HTTPException(status_code=503, detail="PyTorch not available")

        # Create model
        model = PINNModel(layers=config.layers, activation=config.activation)

        # Create trainer
        train_config = TrainingConfig(
            epochs=config.epochs,
            learning_rate=config.learning_rate,
            print_every=max(1, config.epochs // 10),
        )
        trainer = PINNTrainer(model, config=train_config)

        # Generate collocation points (for demo - 2D unit square)
        x_colloc = np.random.rand(config.n_collocation, 2)
        trainer.set_collocation_points(x_colloc)

        # Generate boundary points
        x_bc = np.vstack([
            np.column_stack([np.zeros(config.n_boundary // 4), np.random.rand(config.n_boundary // 4)]),
            np.column_stack([np.ones(config.n_boundary // 4), np.random.rand(config.n_boundary // 4)]),
            np.column_stack([np.random.rand(config.n_boundary // 4), np.zeros(config.n_boundary // 4)]),
            np.column_stack([np.random.rand(config.n_boundary // 4), np.ones(config.n_boundary // 4)]),
        ])
        y_bc = np.zeros((len(x_bc), 1))
        trainer.set_boundary_data(x_bc, y_bc)

        # Train (note: without physics loss defined, this is just BC fitting)
        result = trainer.train()

        return {
            "status": "completed",
            "solver": "pinn",
            "epochs_trained": result.epochs_trained,
            "final_loss": result.final_loss,
            "training_time": result.training_time,
            "model_parameters": model.num_parameters,
            "converged": result.converged,
            "loss_history_length": len(result.loss_history),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"PINN training failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/jobs")
async def list_jobs():
    """List all simulation jobs."""
    try:
        scheduler = _get_scheduler()
        jobs = scheduler.list_jobs()
        stats = scheduler.get_stats()

        return {
            "status": "ok",
            "jobs": jobs,
            "stats": stats,
        }

    except Exception as e:
        logger.error(f"Failed to list jobs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/job/{job_id}")
async def get_job_status(job_id: str):
    """Get status of a specific simulation job."""
    try:
        scheduler = _get_scheduler()
        job = scheduler.get_job(job_id)

        if job is None:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

        return {
            "status": "ok",
            **job.to_dict(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get job: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/job/{job_id}")
async def cancel_job(job_id: str):
    """Cancel a queued simulation job."""
    try:
        scheduler = _get_scheduler()

        if scheduler.cancel_job(job_id):
            return {
                "status": "cancelled",
                "job_id": job_id,
            }
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Job {job_id} not found or not cancellable",
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to cancel job: {e}")
        raise HTTPException(status_code=500, detail=str(e))
