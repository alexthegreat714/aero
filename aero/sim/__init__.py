"""
Simulation module for Aero Agent.

Provides numerical simulation capabilities including:
- Finite difference solvers
- CFD (Computational Fluid Dynamics) solvers
- PINN (Physics-Informed Neural Networks)
"""

from aero.sim.base_simulation import BaseSimulation, SimulationResult
from aero.sim.numerics.fd_solver import FDSolver, LaplaceEquation, HeatEquation
from aero.sim.cfd.ns_solver_stub import NavierStokesSolver
from aero.sim.pinn.pinn_stub import PINNSolver

__all__ = [
    "BaseSimulation",
    "SimulationResult",
    "FDSolver",
    "LaplaceEquation",
    "HeatEquation",
    "NavierStokesSolver",
    "PINNSolver",
]
