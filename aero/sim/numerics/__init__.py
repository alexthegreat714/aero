"""
Numerical methods for Aero simulations.

Provides finite difference and other numerical solvers.
"""

from aero.sim.numerics.fd_solver import FDSolver, LaplaceEquation, HeatEquation

__all__ = [
    "FDSolver",
    "LaplaceEquation",
    "HeatEquation",
]
