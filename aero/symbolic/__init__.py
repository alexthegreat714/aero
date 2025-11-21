"""
Symbolic / Constraint Validator Layer for Aero Agent.

Provides:
- Symbolic expression representation for PDEs
- Dimensional analysis utilities
- Constraint definitions and evaluation
- Integration with validation pipeline

SymPy is optional - the module degrades gracefully without it.
"""

import logging

logger = logging.getLogger(__name__)

# Check for SymPy availability
try:
    import sympy as sp
    SYM_AVAILABLE = True
    logger.debug("SymPy available for symbolic computations")
except ImportError:  # pragma: no cover
    sp = None
    SYM_AVAILABLE = False
    logger.debug("SymPy not available - symbolic features disabled")

# Import submodules
from aero.symbolic.expressions import (
    SymbolicExpression,
    make_heat_equation_symbolic,
    make_laplace_equation_symbolic,
    make_navier_stokes_stub_symbolic,
    make_wave_equation_symbolic,
    get_expression_for_sim_type,
)

from aero.symbolic.dimensions import (
    Dimension,
    multiply_dims,
    divide_dims,
    equal_dims,
    BASE_DIMENSIONS,
    check_dimensional_consistency,
)

from aero.symbolic.constraints import (
    Constraint,
    ConstraintResult,
    make_mass_conservation_constraint,
    make_energy_like_constraint,
    make_dimensional_consistency_constraint,
    make_boundedness_constraint,
    make_monotonicity_constraint,
    get_default_simulation_constraints,
    get_default_experiment_constraints,
)

from aero.symbolic.checks import (
    check_simulation_constraints,
    check_experiment_constraints,
    get_symbolic_status,
)

__all__ = [
    # Availability flag
    "SYM_AVAILABLE",
    # Expressions
    "SymbolicExpression",
    "make_heat_equation_symbolic",
    "make_laplace_equation_symbolic",
    "make_navier_stokes_stub_symbolic",
    "make_wave_equation_symbolic",
    "get_expression_for_sim_type",
    # Dimensions
    "Dimension",
    "multiply_dims",
    "divide_dims",
    "equal_dims",
    "BASE_DIMENSIONS",
    "check_dimensional_consistency",
    # Constraints
    "Constraint",
    "ConstraintResult",
    "make_mass_conservation_constraint",
    "make_energy_like_constraint",
    "make_dimensional_consistency_constraint",
    "make_boundedness_constraint",
    "make_monotonicity_constraint",
    "get_default_simulation_constraints",
    "get_default_experiment_constraints",
    # Checks
    "check_simulation_constraints",
    "check_experiment_constraints",
    "get_symbolic_status",
]
