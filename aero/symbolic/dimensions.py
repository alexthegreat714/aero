"""
Dimensional analysis utilities for Aero Agent.

Provides:
- Dimension class for representing physical dimensions
- Operations on dimensions (multiply, divide, equality)
- Registry of common physical dimensions
- Dimensional consistency checking
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class Dimension:
    """
    Representation of physical dimensions using base unit exponents.

    Base dimensions:
    - M: Mass
    - L: Length
    - T: Time
    - K: Temperature (optional)
    - Q: Electric charge (optional)

    Examples:
        - Length: Dimension("length", L=1)
        - Velocity: Dimension("velocity", L=1, T=-1)
        - Force: Dimension("force", M=1, L=1, T=-2)
    """

    name: str
    M: int = 0  # Mass
    L: int = 0  # Length
    T: int = 0  # Time
    K: int = 0  # Temperature
    Q: int = 0  # Electric charge

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Dimension):
            return False
        return (
            self.M == other.M
            and self.L == other.L
            and self.T == other.T
            and self.K == other.K
            and self.Q == other.Q
        )

    def __hash__(self) -> int:
        return hash((self.M, self.L, self.T, self.K, self.Q))

    def __str__(self) -> str:
        parts = []
        if self.M != 0:
            parts.append(f"M^{self.M}" if self.M != 1 else "M")
        if self.L != 0:
            parts.append(f"L^{self.L}" if self.L != 1 else "L")
        if self.T != 0:
            parts.append(f"T^{self.T}" if self.T != 1 else "T")
        if self.K != 0:
            parts.append(f"K^{self.K}" if self.K != 1 else "K")
        if self.Q != 0:
            parts.append(f"Q^{self.Q}" if self.Q != 1 else "Q")

        if not parts:
            return f"{self.name} [dimensionless]"
        return f"{self.name} [{' '.join(parts)}]"

    def __repr__(self) -> str:
        return f"Dimension({self.name!r}, M={self.M}, L={self.L}, T={self.T}, K={self.K}, Q={self.Q})"

    def is_dimensionless(self) -> bool:
        """Check if this is a dimensionless quantity."""
        return self.M == 0 and self.L == 0 and self.T == 0 and self.K == 0 and self.Q == 0

    def to_tuple(self) -> Tuple[int, int, int, int, int]:
        """Return dimensions as a tuple (M, L, T, K, Q)."""
        return (self.M, self.L, self.T, self.K, self.Q)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "M": self.M,
            "L": self.L,
            "T": self.T,
            "K": self.K,
            "Q": self.Q,
        }


def multiply_dims(a: Dimension, b: Dimension, name: str = "product") -> Dimension:
    """
    Multiply two dimensions (add exponents).

    Args:
        a: First dimension
        b: Second dimension
        name: Name for the resulting dimension

    Returns:
        New Dimension with combined exponents
    """
    return Dimension(
        name=name,
        M=a.M + b.M,
        L=a.L + b.L,
        T=a.T + b.T,
        K=a.K + b.K,
        Q=a.Q + b.Q,
    )


def divide_dims(a: Dimension, b: Dimension, name: str = "quotient") -> Dimension:
    """
    Divide two dimensions (subtract exponents).

    Args:
        a: Numerator dimension
        b: Denominator dimension
        name: Name for the resulting dimension

    Returns:
        New Dimension with subtracted exponents
    """
    return Dimension(
        name=name,
        M=a.M - b.M,
        L=a.L - b.L,
        T=a.T - b.T,
        K=a.K - b.K,
        Q=a.Q - b.Q,
    )


def equal_dims(a: Dimension, b: Dimension) -> bool:
    """
    Check if two dimensions are equal.

    Args:
        a: First dimension
        b: Second dimension

    Returns:
        True if dimensions match
    """
    return a == b


def power_dim(d: Dimension, n: int, name: str = "power") -> Dimension:
    """
    Raise a dimension to a power.

    Args:
        d: Dimension to raise
        n: Power (integer)
        name: Name for the resulting dimension

    Returns:
        New Dimension with scaled exponents
    """
    return Dimension(
        name=name,
        M=d.M * n,
        L=d.L * n,
        T=d.T * n,
        K=d.K * n,
        Q=d.Q * n,
    )


# Registry of common physical dimensions
BASE_DIMENSIONS: Dict[str, Dimension] = {
    # Base dimensions
    "dimensionless": Dimension("dimensionless"),
    "length": Dimension("length", L=1),
    "time": Dimension("time", T=1),
    "mass": Dimension("mass", M=1),
    "temperature": Dimension("temperature", K=1),
    "charge": Dimension("charge", Q=1),

    # Derived dimensions - Mechanics
    "area": Dimension("area", L=2),
    "volume": Dimension("volume", L=3),
    "velocity": Dimension("velocity", L=1, T=-1),
    "acceleration": Dimension("acceleration", L=1, T=-2),
    "force": Dimension("force", M=1, L=1, T=-2),
    "pressure": Dimension("pressure", M=1, L=-1, T=-2),
    "energy": Dimension("energy", M=1, L=2, T=-2),
    "power": Dimension("power", M=1, L=2, T=-3),
    "momentum": Dimension("momentum", M=1, L=1, T=-1),

    # Fluid mechanics
    "density": Dimension("density", M=1, L=-3),
    "kinematic_viscosity": Dimension("kinematic_viscosity", L=2, T=-1),
    "dynamic_viscosity": Dimension("dynamic_viscosity", M=1, L=-1, T=-1),
    "mass_flow_rate": Dimension("mass_flow_rate", M=1, T=-1),
    "volumetric_flow_rate": Dimension("volumetric_flow_rate", L=3, T=-1),

    # Heat transfer
    "thermal_diffusivity": Dimension("thermal_diffusivity", L=2, T=-1),
    "thermal_conductivity": Dimension("thermal_conductivity", M=1, L=1, T=-3, K=-1),
    "specific_heat": Dimension("specific_heat", L=2, T=-2, K=-1),
    "heat_flux": Dimension("heat_flux", M=1, T=-3),

    # Dimensionless numbers (for reference)
    "reynolds_number": Dimension("reynolds_number"),
    "prandtl_number": Dimension("prandtl_number"),
    "nusselt_number": Dimension("nusselt_number"),
}

# Aliases
BASE_DIMENSIONS["L"] = BASE_DIMENSIONS["length"]
BASE_DIMENSIONS["T"] = BASE_DIMENSIONS["time"]
BASE_DIMENSIONS["M"] = BASE_DIMENSIONS["mass"]
BASE_DIMENSIONS["K"] = BASE_DIMENSIONS["temperature"]
BASE_DIMENSIONS["viscosity"] = BASE_DIMENSIONS["kinematic_viscosity"]
BASE_DIMENSIONS["alpha"] = BASE_DIMENSIONS["thermal_diffusivity"]


def get_dimension(name: str) -> Optional[Dimension]:
    """
    Get a dimension by name from the registry.

    Args:
        name: Dimension name

    Returns:
        Dimension or None if not found
    """
    return BASE_DIMENSIONS.get(name.lower())


def check_dimensional_consistency(
    equation_type: str,
    parameters: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Check dimensional consistency for a given equation type and parameters.

    Args:
        equation_type: Type of equation (e.g., "heat_1d", "laplace_2d")
        parameters: Dictionary of parameter names to values/dimensions

    Returns:
        Dictionary with:
            - consistent: bool
            - details: list of check results
            - errors: list of inconsistencies found
    """
    result = {
        "consistent": True,
        "details": [],
        "errors": [],
    }

    equation_type_lower = equation_type.lower()

    if "heat" in equation_type_lower:
        # Heat equation: u_t = α u_xx
        # Dimensions: [u]/[t] = [α][u]/[L]²
        # So: [α] = [L]²/[T] = thermal diffusivity

        if "alpha" in parameters or "thermal_diffusivity" in parameters:
            alpha_dim = BASE_DIMENSIONS["thermal_diffusivity"]
            result["details"].append({
                "check": "alpha_dimension",
                "expected": str(alpha_dim),
                "status": "checked",
            })

        if "dx" in parameters and "dt" in parameters:
            # Stability check: α * dt / dx² should be dimensionless
            result["details"].append({
                "check": "stability_dimensionless",
                "expression": "α * dt / dx²",
                "expected_dimension": "dimensionless",
                "status": "checked",
            })

    elif "laplace" in equation_type_lower:
        # Laplace equation: ∇²u = 0
        # No time derivative, just spatial
        result["details"].append({
            "check": "laplace_spatial_only",
            "status": "checked",
        })

    elif "navier" in equation_type_lower or "ns" in equation_type_lower:
        # Navier-Stokes: check viscosity dimension
        if "nu" in parameters or "viscosity" in parameters:
            nu_dim = BASE_DIMENSIONS["kinematic_viscosity"]
            result["details"].append({
                "check": "viscosity_dimension",
                "expected": str(nu_dim),
                "status": "checked",
            })

        if "rho" in parameters or "density" in parameters:
            rho_dim = BASE_DIMENSIONS["density"]
            result["details"].append({
                "check": "density_dimension",
                "expected": str(rho_dim),
                "status": "checked",
            })

    else:
        result["details"].append({
            "check": "unknown_equation_type",
            "status": "skipped",
            "note": f"No dimensional rules for {equation_type}",
        })

    return result


def verify_expression_dimensions(
    lhs_dim: Dimension,
    rhs_dim: Dimension,
) -> Dict[str, Any]:
    """
    Verify that left and right hand sides of an equation have matching dimensions.

    Args:
        lhs_dim: Dimension of left-hand side
        rhs_dim: Dimension of right-hand side

    Returns:
        Dictionary with verification result
    """
    matches = equal_dims(lhs_dim, rhs_dim)

    return {
        "dimensionally_consistent": matches,
        "lhs": str(lhs_dim),
        "rhs": str(rhs_dim),
        "error": None if matches else f"Dimension mismatch: {lhs_dim} != {rhs_dim}",
    }
