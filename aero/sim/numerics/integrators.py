"""
Time integration methods for Aero Agent simulations.

Provides general-purpose ODE integrators:
- Forward Euler (1st order)
- Runge-Kutta 4 (4th order)
- Midpoint method (2nd order)
- Heun's method (2nd order)

All integrators work with user-supplied derivative functions
of the form: f(t, y) -> dy/dt
"""

import logging
from typing import Callable, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# Type alias for derivative function: f(t, y) -> dy/dt
DerivativeFunc = Callable[[float, np.ndarray], np.ndarray]


def euler_step(
    f: DerivativeFunc,
    t: float,
    y: np.ndarray,
    dt: float,
) -> np.ndarray:
    """
    Forward Euler integration step.

    First-order method: y_{n+1} = y_n + dt * f(t_n, y_n)

    Args:
        f: Derivative function f(t, y) -> dy/dt
        t: Current time
        y: Current state
        dt: Time step

    Returns:
        New state y_{n+1}
    """
    return y + dt * f(t, y)


def rk4_step(
    f: DerivativeFunc,
    t: float,
    y: np.ndarray,
    dt: float,
) -> np.ndarray:
    """
    4th-order Runge-Kutta integration step.

    Classic RK4 method with four function evaluations per step.

    Args:
        f: Derivative function f(t, y) -> dy/dt
        t: Current time
        y: Current state
        dt: Time step

    Returns:
        New state y_{n+1}
    """
    k1 = f(t, y)
    k2 = f(t + dt / 2, y + dt * k1 / 2)
    k3 = f(t + dt / 2, y + dt * k2 / 2)
    k4 = f(t + dt, y + dt * k3)

    return y + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


def midpoint_step(
    f: DerivativeFunc,
    t: float,
    y: np.ndarray,
    dt: float,
) -> np.ndarray:
    """
    Midpoint method (2nd order Runge-Kutta).

    Also known as modified Euler method.

    Args:
        f: Derivative function f(t, y) -> dy/dt
        t: Current time
        y: Current state
        dt: Time step

    Returns:
        New state y_{n+1}
    """
    k1 = f(t, y)
    k2 = f(t + dt / 2, y + dt * k1 / 2)

    return y + dt * k2


def heun_step(
    f: DerivativeFunc,
    t: float,
    y: np.ndarray,
    dt: float,
) -> np.ndarray:
    """
    Heun's method (2nd order, improved Euler).

    Predictor-corrector method using average of derivatives.

    Args:
        f: Derivative function f(t, y) -> dy/dt
        t: Current time
        y: Current state
        dt: Time step

    Returns:
        New state y_{n+1}
    """
    k1 = f(t, y)
    y_predict = y + dt * k1
    k2 = f(t + dt, y_predict)

    return y + (dt / 2) * (k1 + k2)


def integrate(
    f: DerivativeFunc,
    y0: np.ndarray,
    t_span: Tuple[float, float],
    dt: float,
    method: str = "rk4",
    return_history: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Integrate an ODE over a time span.

    Args:
        f: Derivative function f(t, y) -> dy/dt
        y0: Initial condition
        t_span: (t_start, t_end)
        dt: Time step
        method: Integration method ('euler', 'rk4', 'midpoint', 'heun')
        return_history: If True, return full time history

    Returns:
        If return_history is False: Final state y(t_end)
        If return_history is True: (t_array, y_history, final_y)
    """
    # Select step function
    step_funcs = {
        "euler": euler_step,
        "rk4": rk4_step,
        "midpoint": midpoint_step,
        "heun": heun_step,
    }

    if method not in step_funcs:
        raise ValueError(f"Unknown method: {method}. Available: {list(step_funcs.keys())}")

    step_fn = step_funcs[method]

    # Initialize
    t_start, t_end = t_span
    t = t_start
    y = np.array(y0, dtype=np.float64)

    # Storage for history
    if return_history:
        t_history = [t]
        y_history = [y.copy()]

    # Integration loop
    n_steps = int(np.ceil((t_end - t_start) / dt))

    for _ in range(n_steps):
        # Adjust last step if needed
        if t + dt > t_end:
            dt_actual = t_end - t
        else:
            dt_actual = dt

        # Take step
        y = step_fn(f, t, y, dt_actual)
        t += dt_actual

        if return_history:
            t_history.append(t)
            y_history.append(y.copy())

        # Check for completion
        if t >= t_end - 1e-10:
            break

    if return_history:
        return np.array(t_history), np.array(y_history), y
    else:
        return y


def integrate_adaptive(
    f: DerivativeFunc,
    y0: np.ndarray,
    t_span: Tuple[float, float],
    dt_initial: float = 0.01,
    rtol: float = 1e-6,
    atol: float = 1e-9,
    max_steps: int = 100000,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Integrate ODE with adaptive step size control.

    Uses embedded RK4(5) method for error estimation.

    Args:
        f: Derivative function f(t, y) -> dy/dt
        y0: Initial condition
        t_span: (t_start, t_end)
        dt_initial: Initial time step
        rtol: Relative tolerance
        atol: Absolute tolerance
        max_steps: Maximum number of steps

    Returns:
        Tuple of (t_array, y_array)
    """
    t_start, t_end = t_span
    t = t_start
    y = np.array(y0, dtype=np.float64)
    dt = dt_initial

    t_history = [t]
    y_history = [y.copy()]

    for step in range(max_steps):
        if t >= t_end:
            break

        # Don't overshoot
        if t + dt > t_end:
            dt = t_end - t

        # RK4 step
        y_rk4 = rk4_step(f, t, y, dt)

        # Two half-steps for error estimate
        y_half = rk4_step(f, t, y, dt / 2)
        y_double = rk4_step(f, t + dt / 2, y_half, dt / 2)

        # Error estimate
        error = np.max(np.abs(y_double - y_rk4))
        scale = atol + rtol * np.max(np.abs(y))

        if error < scale or dt < 1e-15:
            # Accept step
            t += dt
            y = y_double  # Use more accurate result

            t_history.append(t)
            y_history.append(y.copy())

        # Adjust step size
        if error > 0:
            factor = 0.9 * (scale / error) ** 0.2
            factor = min(max(factor, 0.1), 10.0)
            dt = dt * factor
        else:
            dt = dt * 2

    if step >= max_steps - 1:
        logger.warning(f"Maximum steps ({max_steps}) reached")

    return np.array(t_history), np.array(y_history)


# =============================================================================
# Convenience Functions for Common ODEs
# =============================================================================


def solve_harmonic_oscillator(
    omega: float,
    y0: np.ndarray,
    t_span: Tuple[float, float],
    dt: float,
    method: str = "rk4",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve simple harmonic oscillator: y'' + omega^2 * y = 0

    Args:
        omega: Angular frequency
        y0: Initial [position, velocity]
        t_span: Time span
        dt: Time step
        method: Integration method

    Returns:
        Tuple of (t_array, y_array)
    """
    def f(t, y):
        return np.array([y[1], -omega**2 * y[0]])

    t, y, _ = integrate(f, y0, t_span, dt, method, return_history=True)
    return t, y


def solve_exponential_decay(
    k: float,
    y0: float,
    t_span: Tuple[float, float],
    dt: float,
    method: str = "rk4",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve exponential decay: dy/dt = -k * y

    Args:
        k: Decay constant
        y0: Initial value
        t_span: Time span
        dt: Time step
        method: Integration method

    Returns:
        Tuple of (t_array, y_array)
    """
    def f(t, y):
        return -k * y

    t, y, _ = integrate(f, np.array([y0]), t_span, dt, method, return_history=True)
    return t, y[:, 0]
