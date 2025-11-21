"""
Finite Difference solvers for Aero Agent.

Provides basic FD implementations for common PDEs.
"""

import logging
from typing import Callable, Optional

import numpy as np

from aero.sim.base_simulation import BaseSimulation, SimulationConfig, SimulationResult

logger = logging.getLogger(__name__)


class FDSolver(BaseSimulation):
    """
    Generic finite difference solver base class.

    Provides grid management and iteration framework.
    """

    def __init__(
        self,
        nx: int = 50,
        ny: int = 50,
        dx: float = 1.0,
        dy: float = 1.0,
        config: Optional[SimulationConfig] = None,
    ):
        """
        Initialize the FD solver.

        Args:
            nx: Number of grid points in x direction
            ny: Number of grid points in y direction
            dx: Grid spacing in x
            dy: Grid spacing in y
            config: Simulation configuration
        """
        super().__init__(config)
        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dy = dy

        # Solution arrays
        self.u: np.ndarray = np.zeros((nx, ny))
        self.u_new: np.ndarray = np.zeros((nx, ny))

    @property
    def name(self) -> str:
        return "fd_solver"

    def setup(self) -> None:
        """Initialize the solution field."""
        self.u = np.zeros((self.nx, self.ny))
        self.u_new = np.zeros((self.nx, self.ny))

    def set_boundary_conditions(
        self,
        left: float = 0.0,
        right: float = 0.0,
        top: float = 0.0,
        bottom: float = 0.0,
    ) -> None:
        """
        Set Dirichlet boundary conditions.

        Args:
            left: Value at left boundary (x=0)
            right: Value at right boundary (x=nx-1)
            top: Value at top boundary (y=ny-1)
            bottom: Value at bottom boundary (y=0)
        """
        self.u[0, :] = left
        self.u[-1, :] = right
        self.u[:, 0] = bottom
        self.u[:, -1] = top

    def step(self) -> float:
        """Execute one iteration - to be implemented by subclasses."""
        raise NotImplementedError

    def run(self) -> SimulationResult:
        """Run the simulation using the standard loop."""
        return self.run_loop()

    def get_solution(self) -> np.ndarray:
        """Get the current solution field."""
        return self.u.copy()


class LaplaceEquation(FDSolver):
    """
    2D Laplace equation solver using Jacobi iteration.

    Solves: ∇²u = 0

    Uses the 5-point stencil:
    u(i,j) = 0.25 * (u(i+1,j) + u(i-1,j) + u(i,j+1) + u(i,j-1))
    """

    @property
    def name(self) -> str:
        return "laplace_equation"

    def step(self) -> float:
        """
        Execute one Jacobi iteration.

        Returns:
            Maximum residual (change from previous iteration)
        """
        # Store old values for residual calculation
        old_u = self.u.copy()

        # Update interior points
        for i in range(1, self.nx - 1):
            for j in range(1, self.ny - 1):
                self.u_new[i, j] = 0.25 * (
                    self.u[i + 1, j]
                    + self.u[i - 1, j]
                    + self.u[i, j + 1]
                    + self.u[i, j - 1]
                )

        # Copy boundary conditions
        self.u_new[0, :] = self.u[0, :]
        self.u_new[-1, :] = self.u[-1, :]
        self.u_new[:, 0] = self.u[:, 0]
        self.u_new[:, -1] = self.u[:, -1]

        # Swap arrays
        self.u, self.u_new = self.u_new, self.u

        # Calculate residual
        residual = np.max(np.abs(self.u - old_u))
        return residual


class HeatEquation(FDSolver):
    """
    2D Heat equation solver using explicit time stepping.

    Solves: ∂u/∂t = α * ∇²u

    Uses forward Euler time integration with stability constraint:
    dt <= dx² * dy² / (2 * α * (dx² + dy²))
    """

    def __init__(
        self,
        nx: int = 50,
        ny: int = 50,
        dx: float = 0.1,
        dy: float = 0.1,
        alpha: float = 0.01,
        dt: Optional[float] = None,
        config: Optional[SimulationConfig] = None,
    ):
        """
        Initialize the heat equation solver.

        Args:
            nx: Number of grid points in x
            ny: Number of grid points in y
            dx: Grid spacing in x
            dy: Grid spacing in y
            alpha: Thermal diffusivity
            dt: Time step (auto-calculated for stability if not provided)
            config: Simulation configuration
        """
        super().__init__(nx, ny, dx, dy, config)
        self.alpha = alpha

        # Calculate stable time step if not provided
        if dt is None:
            self.dt = 0.4 * dx * dx * dy * dy / (2 * alpha * (dx * dx + dy * dy))
        else:
            self.dt = dt

        self._time = 0.0

    @property
    def name(self) -> str:
        return "heat_equation"

    def setup(self) -> None:
        """Initialize the temperature field."""
        super().setup()
        self._time = 0.0

    def step(self) -> float:
        """
        Execute one time step.

        Returns:
            Maximum change in temperature
        """
        # Precompute coefficients
        rx = self.alpha * self.dt / (self.dx * self.dx)
        ry = self.alpha * self.dt / (self.dy * self.dy)

        # Store for residual
        old_u = self.u.copy()

        # Update interior points (explicit Euler)
        for i in range(1, self.nx - 1):
            for j in range(1, self.ny - 1):
                laplacian = (
                    (self.u[i + 1, j] - 2 * self.u[i, j] + self.u[i - 1, j]) / (self.dx * self.dx)
                    + (self.u[i, j + 1] - 2 * self.u[i, j] + self.u[i, j - 1]) / (self.dy * self.dy)
                )
                self.u_new[i, j] = self.u[i, j] + self.alpha * self.dt * laplacian

        # Copy boundary conditions
        self.u_new[0, :] = self.u[0, :]
        self.u_new[-1, :] = self.u[-1, :]
        self.u_new[:, 0] = self.u[:, 0]
        self.u_new[:, -1] = self.u[:, -1]

        # Swap arrays
        self.u, self.u_new = self.u_new, self.u

        # Update time
        self._time += self.dt

        # Calculate change
        residual = np.max(np.abs(self.u - old_u))
        return residual

    @property
    def current_time(self) -> float:
        """Get the current simulation time."""
        return self._time

    def set_initial_condition(self, func: Callable[[float, float], float]) -> None:
        """
        Set initial temperature distribution.

        Args:
            func: Function f(x, y) returning temperature
        """
        for i in range(self.nx):
            for j in range(self.ny):
                x = i * self.dx
                y = j * self.dy
                self.u[i, j] = func(x, y)
