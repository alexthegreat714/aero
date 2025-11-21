"""
Simulation module for Aero Agent.

Provides numerical simulation capabilities including:
- Finite difference solvers (Heat equation, Laplace equation)
- CFD (Computational Fluid Dynamics) solvers
- PINN (Physics-Informed Neural Networks)
- Simulation job scheduling
"""

from aero.sim.base_simulation import BaseSimulation, SimulationConfig, SimulationResult
from aero.sim.numerics.fd_solver import (
    FDSolver,
    LaplaceEquation,
    HeatEquation,
    solve_heat_1d,
    solve_laplace_2d,
)
from aero.sim.cfd.ns_solver_stub import NavierStokesSolver, NavierStokes2DStub
from aero.sim.pinn.pinn_stub import PINNSolver
from aero.sim.job import SimulationJob, JobStatus
from aero.sim.scheduler import SimulationScheduler, get_scheduler

__all__ = [
    # Base classes
    "BaseSimulation",
    "SimulationConfig",
    "SimulationResult",
    # Finite Difference
    "FDSolver",
    "LaplaceEquation",
    "HeatEquation",
    "solve_heat_1d",
    "solve_laplace_2d",
    # CFD
    "NavierStokesSolver",
    "NavierStokes2DStub",
    # PINN
    "PINNSolver",
    # Job System
    "SimulationJob",
    "JobStatus",
    "SimulationScheduler",
    "get_scheduler",
]
