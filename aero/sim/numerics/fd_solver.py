"""
Finite Difference solvers for Aero Agent.

Provides basic FD implementations for common PDEs:
- 1D Heat equation (explicit time-stepping)
- 2D Laplace equation (Gauss-Seidel iteration)

Both class-based solvers and standalone functions are provided.
"""

import logging
from typing import Callable, Optional, Tuple

import numpy as np

from aero.sim.base_simulation import BaseSimulation, SimulationConfig, SimulationResult

logger = logging.getLogger(__name__)


# =============================================================================
# Standalone Solver Functions
# =============================================================================


def solve_heat_1d(
    u0: np.ndarray,
    alpha: float,
    dx: float,
    dt: float,
    steps: int,
    boundary_left: Optional[float] = None,
    boundary_right: Optional[float] = None,
) -> Tuple[np.ndarray, dict]:
    """
    Solve 1D heat equation using explicit finite difference.

    Solves: du/dt = alpha * d²u/dx²

    Uses forward Euler time stepping with central difference for spatial derivative.
    Stability requires: alpha * dt / dx² <= 0.5

    Args:
        u0: Initial temperature distribution (1D array)
        alpha: Thermal diffusivity
        dx: Grid spacing
        dt: Time step
        steps: Number of time steps to take

    Returns:
        Tuple of (final_u, metadata_dict)

    Example:
        # Create initial condition with hot spot in center
        x = np.linspace(0, 1, 101)
        u0 = np.exp(-100 * (x - 0.5)**2)

        # Solve for 1000 steps
        u_final, info = solve_heat_1d(u0, alpha=0.01, dx=0.01, dt=0.0001, steps=1000)
    """
    # Check stability
    r = alpha * dt / (dx * dx)
    if r > 0.5:
        logger.warning(f"Heat equation may be unstable: r = {r:.4f} > 0.5")

    # Initialize
    nx = len(u0)
    u = u0.copy()
    u_new = np.zeros_like(u)

    # Apply boundary conditions if specified
    if boundary_left is not None:
        u[0] = boundary_left
    if boundary_right is not None:
        u[-1] = boundary_right

    # Time stepping
    for step in range(steps):
        # Interior points using explicit scheme
        for i in range(1, nx - 1):
            u_new[i] = u[i] + r * (u[i + 1] - 2 * u[i] + u[i - 1])

        # Boundary conditions
        if boundary_left is not None:
            u_new[0] = boundary_left
        else:
            u_new[0] = u[0]  # Keep initial

        if boundary_right is not None:
            u_new[-1] = boundary_right
        else:
            u_new[-1] = u[-1]  # Keep initial

        # Swap arrays
        u, u_new = u_new, u

    # Metadata
    metadata = {
        "solver": "heat_1d_explicit",
        "nx": nx,
        "dx": dx,
        "dt": dt,
        "alpha": alpha,
        "r": r,
        "steps": steps,
        "final_time": steps * dt,
        "stable": r <= 0.5,
    }

    logger.info(f"Heat 1D solved: {steps} steps, r={r:.4f}, time={steps*dt:.4f}")

    return u, metadata


def solve_laplace_2d(
    initial_grid: np.ndarray,
    tol: float = 1e-6,
    max_iterations: int = 10000,
    omega: float = 1.0,
) -> Tuple[np.ndarray, dict]:
    """
    Solve 2D Laplace equation using Gauss-Seidel iteration.

    Solves: d²u/dx² + d²u/dy² = 0

    Boundary conditions are taken from the edges of initial_grid.
    Uses successive over-relaxation (SOR) if omega > 1.

    Args:
        initial_grid: 2D array with boundary values set, interior can be any guess
        tol: Convergence tolerance (max change between iterations)
        max_iterations: Maximum number of iterations
        omega: Relaxation factor (1.0 = Gauss-Seidel, >1 = SOR, <1 = under-relaxation)

    Returns:
        Tuple of (solution_grid, metadata_dict)

    Example:
        # Create grid with boundary conditions
        u = np.zeros((50, 50))
        u[0, :] = 100   # Top boundary = 100
        u[-1, :] = 0    # Bottom = 0
        u[:, 0] = 50    # Left = 50
        u[:, -1] = 50   # Right = 50

        # Solve
        solution, info = solve_laplace_2d(u, tol=1e-6)
        print(f"Converged in {info['iterations']} iterations")
    """
    nx, ny = initial_grid.shape
    u = initial_grid.copy()

    # Store boundary values
    bc_left = u[0, :].copy()
    bc_right = u[-1, :].copy()
    bc_bottom = u[:, 0].copy()
    bc_top = u[:, -1].copy()

    converged = False
    iterations = 0
    final_residual = float('inf')

    for iteration in range(max_iterations):
        max_change = 0.0

        # Gauss-Seidel iteration over interior points
        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                # Standard 5-point stencil average
                u_gs = 0.25 * (u[i + 1, j] + u[i - 1, j] + u[i, j + 1] + u[i, j - 1])

                # SOR update
                u_new = omega * u_gs + (1 - omega) * u[i, j]

                # Track convergence
                change = abs(u_new - u[i, j])
                if change > max_change:
                    max_change = change

                u[i, j] = u_new

        # Enforce boundary conditions
        u[0, :] = bc_left
        u[-1, :] = bc_right
        u[:, 0] = bc_bottom
        u[:, -1] = bc_top

        iterations = iteration + 1
        final_residual = max_change

        if max_change < tol:
            converged = True
            break

    # Metadata
    metadata = {
        "solver": "laplace_2d_gauss_seidel",
        "nx": nx,
        "ny": ny,
        "iterations": iterations,
        "final_residual": final_residual,
        "converged": converged,
        "tolerance": tol,
        "omega": omega,
    }

    if converged:
        logger.info(f"Laplace 2D converged in {iterations} iterations, residual={final_residual:.2e}")
    else:
        logger.warning(f"Laplace 2D did not converge after {iterations} iterations, residual={final_residual:.2e}")

    return u, metadata


# =============================================================================
# Class-Based Solvers
# =============================================================================


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
