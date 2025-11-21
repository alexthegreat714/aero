"""
Numerical utilities for Aero Agent simulations.

Provides:
- Grid creation
- Stability checks (CFL condition)
- Boundary condition helpers
- Array padding and safe indexing
"""

import logging
from typing import Callable, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Grid Creation
# =============================================================================


def create_grid_1d(
    x_min: float,
    x_max: float,
    nx: int,
    include_endpoints: bool = True,
) -> np.ndarray:
    """
    Create a 1D uniform grid.

    Args:
        x_min: Minimum coordinate
        x_max: Maximum coordinate
        nx: Number of grid points
        include_endpoints: If True, include x_min and x_max

    Returns:
        1D array of grid coordinates
    """
    if include_endpoints:
        return np.linspace(x_min, x_max, nx)
    else:
        dx = (x_max - x_min) / nx
        return np.linspace(x_min + dx / 2, x_max - dx / 2, nx)


def create_grid_2d(
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    nx: int,
    ny: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a 2D uniform grid.

    Args:
        x_range: (x_min, x_max)
        y_range: (y_min, y_max)
        nx: Number of points in x
        ny: Number of points in y

    Returns:
        Tuple of (X, Y) meshgrid arrays
    """
    x = np.linspace(x_range[0], x_range[1], nx)
    y = np.linspace(y_range[0], y_range[1], ny)
    return np.meshgrid(x, y, indexing='ij')


def get_grid_spacing(
    x_range: Tuple[float, float],
    nx: int,
) -> float:
    """
    Calculate uniform grid spacing.

    Args:
        x_range: (x_min, x_max)
        nx: Number of grid points

    Returns:
        Grid spacing dx
    """
    return (x_range[1] - x_range[0]) / (nx - 1)


# =============================================================================
# Stability Checks
# =============================================================================


def check_cfl_1d(
    u_max: float,
    dx: float,
    dt: float,
    cfl_limit: float = 1.0,
) -> Tuple[bool, float]:
    """
    Check 1D CFL (Courant-Friedrichs-Lewy) condition.

    The CFL condition for advection: C = u * dt / dx <= CFL_limit

    Args:
        u_max: Maximum velocity
        dx: Grid spacing
        dt: Time step
        cfl_limit: Maximum allowed CFL number (default 1.0)

    Returns:
        Tuple of (is_stable, cfl_number)
    """
    if dx <= 0 or dt <= 0:
        raise ValueError("dx and dt must be positive")

    cfl = abs(u_max) * dt / dx
    is_stable = cfl <= cfl_limit

    if not is_stable:
        logger.warning(f"CFL condition violated: {cfl:.4f} > {cfl_limit}")

    return is_stable, cfl


def check_cfl_2d(
    u_max: float,
    v_max: float,
    dx: float,
    dy: float,
    dt: float,
    cfl_limit: float = 1.0,
) -> Tuple[bool, float]:
    """
    Check 2D CFL condition.

    CFL for 2D: C = (|u|/dx + |v|/dy) * dt <= CFL_limit

    Args:
        u_max: Maximum x-velocity
        v_max: Maximum y-velocity
        dx: Grid spacing in x
        dy: Grid spacing in y
        dt: Time step
        cfl_limit: Maximum allowed CFL number

    Returns:
        Tuple of (is_stable, cfl_number)
    """
    cfl = (abs(u_max) / dx + abs(v_max) / dy) * dt
    is_stable = cfl <= cfl_limit

    if not is_stable:
        logger.warning(f"2D CFL condition violated: {cfl:.4f} > {cfl_limit}")

    return is_stable, cfl


def check_diffusion_stability(
    alpha: float,
    dx: float,
    dt: float,
    dim: int = 1,
) -> Tuple[bool, float]:
    """
    Check stability for explicit diffusion schemes.

    For 1D: alpha * dt / dx^2 <= 0.5
    For 2D: alpha * dt * (1/dx^2 + 1/dy^2) <= 0.5

    Args:
        alpha: Diffusion coefficient
        dx: Grid spacing
        dt: Time step
        dim: Spatial dimension (1 or 2, assumes dx=dy for 2D)

    Returns:
        Tuple of (is_stable, stability_number)
    """
    if dim == 1:
        r = alpha * dt / (dx * dx)
        limit = 0.5
    else:
        r = 2 * alpha * dt / (dx * dx)  # Assumes dx = dy
        limit = 0.5

    is_stable = r <= limit

    if not is_stable:
        logger.warning(f"Diffusion stability violated: {r:.4f} > {limit}")

    return is_stable, r


def calculate_stable_dt(
    dx: float,
    alpha: float = 0.0,
    u_max: float = 0.0,
    safety_factor: float = 0.8,
) -> float:
    """
    Calculate stable time step for combined advection-diffusion.

    Args:
        dx: Grid spacing
        alpha: Diffusion coefficient
        u_max: Maximum velocity
        safety_factor: Safety factor (< 1.0)

    Returns:
        Stable time step dt
    """
    dt_adv = float('inf')
    dt_diff = float('inf')

    if abs(u_max) > 1e-10:
        dt_adv = dx / abs(u_max)

    if alpha > 1e-10:
        dt_diff = 0.5 * dx * dx / alpha

    dt = safety_factor * min(dt_adv, dt_diff)
    return dt


# =============================================================================
# Boundary Condition Helpers
# =============================================================================


def apply_dirichlet_1d(
    u: np.ndarray,
    left: float,
    right: float,
) -> np.ndarray:
    """
    Apply Dirichlet boundary conditions in 1D.

    Args:
        u: Solution array
        left: Value at left boundary
        right: Value at right boundary

    Returns:
        Array with boundary conditions applied
    """
    u_new = u.copy()
    u_new[0] = left
    u_new[-1] = right
    return u_new


def apply_dirichlet_2d(
    u: np.ndarray,
    left: float = 0.0,
    right: float = 0.0,
    bottom: float = 0.0,
    top: float = 0.0,
) -> np.ndarray:
    """
    Apply Dirichlet boundary conditions in 2D.

    Args:
        u: Solution array (nx, ny)
        left, right, bottom, top: Boundary values

    Returns:
        Array with boundary conditions applied
    """
    u_new = u.copy()
    u_new[0, :] = left
    u_new[-1, :] = right
    u_new[:, 0] = bottom
    u_new[:, -1] = top
    return u_new


def apply_neumann_1d(
    u: np.ndarray,
    left_flux: float,
    right_flux: float,
    dx: float,
) -> np.ndarray:
    """
    Apply Neumann (derivative) boundary conditions in 1D.

    Args:
        u: Solution array
        left_flux: du/dx at left boundary
        right_flux: du/dx at right boundary
        dx: Grid spacing

    Returns:
        Array with boundary conditions applied
    """
    u_new = u.copy()
    u_new[0] = u_new[1] - left_flux * dx
    u_new[-1] = u_new[-2] + right_flux * dx
    return u_new


def apply_periodic_1d(u: np.ndarray) -> np.ndarray:
    """
    Apply periodic boundary conditions in 1D.

    Args:
        u: Solution array

    Returns:
        Array with periodic boundaries
    """
    u_new = u.copy()
    u_new[0] = u_new[-2]
    u_new[-1] = u_new[1]
    return u_new


# =============================================================================
# Array Utilities
# =============================================================================


def pad_array(
    u: np.ndarray,
    pad_width: int = 1,
    mode: str = "constant",
    constant_value: float = 0.0,
) -> np.ndarray:
    """
    Pad array with ghost cells.

    Args:
        u: Input array
        pad_width: Number of cells to pad
        mode: Padding mode ('constant', 'edge', 'reflect')
        constant_value: Value for constant padding

    Returns:
        Padded array
    """
    if mode == "constant":
        return np.pad(u, pad_width, mode='constant', constant_values=constant_value)
    elif mode == "edge":
        return np.pad(u, pad_width, mode='edge')
    elif mode == "reflect":
        return np.pad(u, pad_width, mode='reflect')
    else:
        raise ValueError(f"Unknown padding mode: {mode}")


def safe_index(
    u: np.ndarray,
    i: int,
    j: Optional[int] = None,
    default: float = 0.0,
) -> float:
    """
    Safely index an array with bounds checking.

    Args:
        u: Input array
        i: First index
        j: Second index (for 2D arrays)
        default: Value to return if out of bounds

    Returns:
        Array value or default
    """
    try:
        if j is None:
            if 0 <= i < len(u):
                return float(u[i])
        else:
            if 0 <= i < u.shape[0] and 0 <= j < u.shape[1]:
                return float(u[i, j])
    except (IndexError, TypeError):
        pass

    return default


def compute_residual(
    u_new: np.ndarray,
    u_old: np.ndarray,
    norm_type: str = "max",
) -> float:
    """
    Compute residual between iterations.

    Args:
        u_new: New solution
        u_old: Previous solution
        norm_type: 'max', 'l2', or 'l1'

    Returns:
        Residual value
    """
    diff = u_new - u_old

    if norm_type == "max":
        return float(np.max(np.abs(diff)))
    elif norm_type == "l2":
        return float(np.sqrt(np.sum(diff ** 2) / diff.size))
    elif norm_type == "l1":
        return float(np.sum(np.abs(diff)) / diff.size)
    else:
        raise ValueError(f"Unknown norm type: {norm_type}")


# =============================================================================
# Initialization Helpers
# =============================================================================


def initialize_gaussian(
    X: np.ndarray,
    Y: Optional[np.ndarray] = None,
    center: Union[float, Tuple[float, float]] = 0.5,
    sigma: float = 0.1,
    amplitude: float = 1.0,
) -> np.ndarray:
    """
    Initialize a Gaussian distribution.

    Args:
        X: X coordinates (1D or 2D meshgrid)
        Y: Y coordinates (for 2D)
        center: Center of Gaussian (scalar for 1D, tuple for 2D)
        sigma: Standard deviation
        amplitude: Peak amplitude

    Returns:
        Gaussian distribution array
    """
    if Y is None:
        # 1D case
        cx = center if isinstance(center, (int, float)) else center[0]
        return amplitude * np.exp(-((X - cx) ** 2) / (2 * sigma ** 2))
    else:
        # 2D case
        cx, cy = center if isinstance(center, tuple) else (center, center)
        return amplitude * np.exp(-(
            (X - cx) ** 2 + (Y - cy) ** 2
        ) / (2 * sigma ** 2))


def initialize_step(
    X: np.ndarray,
    threshold: float = 0.5,
    left_value: float = 1.0,
    right_value: float = 0.0,
) -> np.ndarray:
    """
    Initialize a step function.

    Args:
        X: Coordinates
        threshold: Step location
        left_value: Value for X < threshold
        right_value: Value for X >= threshold

    Returns:
        Step function array
    """
    return np.where(X < threshold, left_value, right_value)
