"""
Tests for the aero.symbolic module.

Tests symbolic expressions, dimensional analysis, constraints, and checks.
"""

import pytest
import numpy as np
from typing import Dict, Any


# =============================================================================
# Dimension Tests
# =============================================================================


class TestDimension:
    """Tests for Dimension class and operations."""

    def test_dimension_creation(self):
        """Test basic dimension creation."""
        from aero.symbolic.dimensions import Dimension

        # Dimensionless
        d = Dimension("dimensionless")
        assert d.M == 0
        assert d.L == 0
        assert d.T == 0
        assert d.is_dimensionless()

        # Length
        d = Dimension("length", L=1)
        assert d.L == 1
        assert not d.is_dimensionless()

        # Velocity
        d = Dimension("velocity", L=1, T=-1)
        assert d.L == 1
        assert d.T == -1

    def test_dimension_equality(self):
        """Test dimension equality comparison."""
        from aero.symbolic.dimensions import Dimension

        d1 = Dimension("length", L=1)
        d2 = Dimension("other_length", L=1)
        d3 = Dimension("area", L=2)

        assert d1 == d2  # Same dimensions, different names
        assert d1 != d3

    def test_dimension_str(self):
        """Test dimension string representation."""
        from aero.symbolic.dimensions import Dimension

        d = Dimension("velocity", L=1, T=-1)
        s = str(d)
        assert "L" in s
        assert "T^-1" in s
        assert "velocity" in s

    def test_multiply_dims(self):
        """Test dimension multiplication."""
        from aero.symbolic.dimensions import Dimension, multiply_dims

        length = Dimension("length", L=1)
        time = Dimension("time", T=1)

        # L * T
        result = multiply_dims(length, time, "LT")
        assert result.L == 1
        assert result.T == 1

        # velocity * time = length
        velocity = Dimension("velocity", L=1, T=-1)
        result = multiply_dims(velocity, time, "length")
        assert result.L == 1
        assert result.T == 0

    def test_divide_dims(self):
        """Test dimension division."""
        from aero.symbolic.dimensions import Dimension, divide_dims

        length = Dimension("length", L=1)
        time = Dimension("time", T=1)

        # L / T = velocity
        result = divide_dims(length, time, "velocity")
        assert result.L == 1
        assert result.T == -1

    def test_power_dim(self):
        """Test dimension power."""
        from aero.symbolic.dimensions import Dimension, power_dim

        length = Dimension("length", L=1)

        # L^2 = area
        area = power_dim(length, 2, "area")
        assert area.L == 2

        # L^3 = volume
        volume = power_dim(length, 3, "volume")
        assert volume.L == 3

    def test_base_dimensions_registry(self):
        """Test BASE_DIMENSIONS registry."""
        from aero.symbolic.dimensions import BASE_DIMENSIONS, get_dimension

        # Check some common dimensions exist
        assert "length" in BASE_DIMENSIONS
        assert "time" in BASE_DIMENSIONS
        assert "mass" in BASE_DIMENSIONS
        assert "velocity" in BASE_DIMENSIONS
        assert "pressure" in BASE_DIMENSIONS

        # Test get_dimension
        length = get_dimension("length")
        assert length is not None
        assert length.L == 1

        # Test case insensitivity
        velocity = get_dimension("VELOCITY")
        assert velocity is not None
        assert velocity.L == 1
        assert velocity.T == -1

    def test_dimensional_consistency_check(self):
        """Test dimensional consistency checking."""
        from aero.symbolic.dimensions import check_dimensional_consistency

        # Heat equation check
        result = check_dimensional_consistency("heat_1d", {"alpha": 0.01, "dx": 0.01, "dt": 0.001})
        assert result["consistent"] is True
        assert "details" in result

        # Laplace equation check
        result = check_dimensional_consistency("laplace_2d", {})
        assert result["consistent"] is True

        # Unknown equation type
        result = check_dimensional_consistency("unknown_equation", {})
        assert "details" in result


# =============================================================================
# Expression Tests
# =============================================================================


class TestExpressions:
    """Tests for symbolic expressions."""

    def test_symbolic_expression_creation(self):
        """Test SymbolicExpression dataclass."""
        from aero.symbolic.expressions import SymbolicExpression

        expr = SymbolicExpression(
            name="test_expr",
            expr=None,
            symbols={},
            metadata={"description": "Test expression"},
        )

        assert expr.name == "test_expr"
        assert not expr.is_available()

    def test_heat_equation_creation(self):
        """Test heat equation creation."""
        from aero.symbolic.expressions import make_heat_equation_symbolic, SYM_AVAILABLE

        heat_eq = make_heat_equation_symbolic()
        assert heat_eq.name == "heat_equation"
        assert "description" in heat_eq.metadata

        if SYM_AVAILABLE:
            assert heat_eq.is_available()
            assert "alpha" in heat_eq.symbols
            assert "x" in heat_eq.symbols
            assert "t" in heat_eq.symbols

    def test_laplace_equation_creation(self):
        """Test Laplace equation creation."""
        from aero.symbolic.expressions import make_laplace_equation_symbolic, SYM_AVAILABLE

        laplace_eq = make_laplace_equation_symbolic()
        assert laplace_eq.name == "laplace_equation"

        if SYM_AVAILABLE:
            assert laplace_eq.is_available()
            assert "x" in laplace_eq.symbols
            assert "y" in laplace_eq.symbols

    def test_wave_equation_creation(self):
        """Test wave equation creation."""
        from aero.symbolic.expressions import make_wave_equation_symbolic, SYM_AVAILABLE

        wave_eq = make_wave_equation_symbolic()
        assert wave_eq.name == "wave_equation"

        if SYM_AVAILABLE:
            assert wave_eq.is_available()
            assert "c" in wave_eq.symbols

    def test_navier_stokes_creation(self):
        """Test Navier-Stokes stub creation."""
        from aero.symbolic.expressions import make_navier_stokes_stub_symbolic, SYM_AVAILABLE

        ns_eq = make_navier_stokes_stub_symbolic()
        assert ns_eq.name == "navier_stokes"
        assert ns_eq.metadata.get("is_placeholder") is True or not SYM_AVAILABLE

    def test_get_expression_for_sim_type(self):
        """Test getting expression by simulation type."""
        from aero.symbolic.expressions import get_expression_for_sim_type

        # Heat
        expr = get_expression_for_sim_type("heat_1d")
        assert expr is not None
        assert "heat" in expr.name

        # Laplace
        expr = get_expression_for_sim_type("laplace_2d")
        assert expr is not None
        assert "laplace" in expr.name

        # Unknown
        expr = get_expression_for_sim_type("unknown_type")
        assert expr is None

    def test_list_available_expressions(self):
        """Test listing all expressions."""
        from aero.symbolic.expressions import list_available_expressions

        expressions = list_available_expressions()
        assert isinstance(expressions, dict)
        assert "heat_equation" in expressions
        assert "laplace_equation" in expressions
        assert "wave_equation" in expressions

    def test_expression_to_dict(self):
        """Test expression serialization."""
        from aero.symbolic.expressions import make_heat_equation_symbolic

        heat_eq = make_heat_equation_symbolic()
        d = heat_eq.to_dict()

        assert "name" in d
        assert "symbols" in d
        assert "metadata" in d
        assert "available" in d


# =============================================================================
# Constraint Tests
# =============================================================================


class TestConstraints:
    """Tests for constraint definitions and evaluation."""

    def test_constraint_result(self):
        """Test ConstraintResult dataclass."""
        from aero.symbolic.constraints import ConstraintResult

        result = ConstraintResult(
            passed=True,
            score=0.9,
            details="Test passed",
            data={"key": "value"},
        )

        assert result.passed is True
        assert result.score == 0.9
        assert "Test passed" in result.details

        d = result.to_dict()
        assert "passed" in d
        assert "score" in d
        assert "details" in d

    def test_constraint_class(self):
        """Test Constraint class."""
        from aero.symbolic.constraints import Constraint, ConstraintResult

        def check_fn(data):
            return ConstraintResult(
                passed=True,
                score=1.0,
                details="Always passes",
            )

        constraint = Constraint(
            name="test_constraint",
            description="A test constraint",
            check_fn=check_fn,
            weight=1.0,
            required=False,
        )

        assert constraint.name == "test_constraint"

        result = constraint.evaluate({"fields": {}})
        assert result.passed is True
        assert result.score == 1.0

    def test_mass_conservation_constraint(self):
        """Test mass conservation constraint."""
        from aero.symbolic.constraints import make_mass_conservation_constraint

        constraint = make_mass_conservation_constraint(tolerance=0.05)
        assert constraint.name == "mass_conservation"

        # Test with conserved mass
        u0 = np.ones(10)
        u_final = np.ones(10) * 0.98  # 2% loss
        data = {
            "fields": {"solution": u_final, "initial": u0},
        }
        result = constraint.evaluate(data)
        assert result.passed is True

        # Test with large mass change
        u_final = np.ones(10) * 0.5  # 50% loss
        data = {
            "fields": {"solution": u_final, "initial": u0},
        }
        result = constraint.evaluate(data)
        assert result.passed is False

    def test_energy_like_constraint(self):
        """Test energy-like constraint."""
        from aero.symbolic.constraints import make_energy_like_constraint

        constraint = make_energy_like_constraint(tolerance=0.1)
        assert constraint.name == "energy_conservation"

        # Test with stable energy
        u0 = np.array([1.0, 2.0, 3.0])
        u_final = u0 * 0.95  # Energy decreased (dissipative)
        data = {
            "fields": {"solution": u_final, "initial": u0},
        }
        result = constraint.evaluate(data)
        assert result.passed is True

    def test_boundedness_constraint(self):
        """Test boundedness constraint."""
        from aero.symbolic.constraints import make_boundedness_constraint

        constraint = make_boundedness_constraint(lower=0.0, upper=100.0)
        assert constraint.name == "boundedness"

        # Within bounds
        data = {"fields": {"solution": np.array([10, 50, 90])}}
        result = constraint.evaluate(data)
        assert result.passed is True

        # Out of bounds
        data = {"fields": {"solution": np.array([-10, 50, 150])}}
        result = constraint.evaluate(data)
        assert result.passed is False

    def test_monotonicity_constraint(self):
        """Test monotonicity constraint."""
        from aero.symbolic.constraints import make_monotonicity_constraint

        # Increasing
        constraint = make_monotonicity_constraint(direction="increasing")
        data = {"fields": {"solution": np.array([1, 2, 3, 4, 5])}}
        result = constraint.evaluate(data)
        assert result.passed is True

        data = {"fields": {"solution": np.array([1, 2, 1, 4, 5])}}
        result = constraint.evaluate(data)
        assert result.passed is False

        # Decreasing
        constraint = make_monotonicity_constraint(direction="decreasing")
        data = {"fields": {"solution": np.array([5, 4, 3, 2, 1])}}
        result = constraint.evaluate(data)
        assert result.passed is True

    def test_convergence_constraint(self):
        """Test convergence constraint."""
        from aero.symbolic.constraints import make_convergence_constraint

        constraint = make_convergence_constraint(threshold=1e-6)
        assert constraint.required is True

        # Converged
        data = {"metadata": {"converged": True, "final_residual": 1e-8}}
        result = constraint.evaluate(data)
        assert result.passed is True

        # Not converged
        data = {"metadata": {"converged": False}}
        result = constraint.evaluate(data)
        assert result.passed is False

    def test_noise_level_constraint(self):
        """Test noise level constraint."""
        from aero.symbolic.constraints import make_noise_level_constraint

        constraint = make_noise_level_constraint(max_noise_ratio=0.1)

        # Low noise signal
        t = np.linspace(0, 1, 100)
        clean_signal = np.sin(2 * np.pi * t)
        data = {"fields": {"values": clean_signal}}
        result = constraint.evaluate(data)
        assert result.score > 0.5

    def test_get_default_simulation_constraints(self):
        """Test default simulation constraints."""
        from aero.symbolic.constraints import get_default_simulation_constraints

        constraints = get_default_simulation_constraints()
        assert len(constraints) > 0
        assert any(c.name == "convergence" for c in constraints)
        assert any(c.name == "mass_conservation" for c in constraints)

    def test_get_default_experiment_constraints(self):
        """Test default experiment constraints."""
        from aero.symbolic.constraints import get_default_experiment_constraints

        constraints = get_default_experiment_constraints()
        assert len(constraints) > 0
        assert any(c.name == "boundedness" for c in constraints)
        assert any(c.name == "noise_level" for c in constraints)


# =============================================================================
# Checks Tests
# =============================================================================


class TestChecks:
    """Tests for high-level constraint checking functions."""

    def test_check_simulation_constraints_basic(self):
        """Test basic simulation constraint checking."""
        from aero.symbolic.checks import check_simulation_constraints

        # Create a simple simulation result
        result = {
            "fields": {
                "solution": np.array([1.0, 2.0, 3.0, 2.0, 1.0]),
                "initial": np.array([1.0, 2.0, 3.0, 2.0, 1.0]),
            },
            "metadata": {
                "converged": True,
                "final_residual": 1e-8,
                "type": "heat_1d",
            },
        }

        check_result = check_simulation_constraints(result)

        assert "constraints" in check_result
        assert "overall_score" in check_result
        assert "passed" in check_result
        assert "sympy_used" in check_result
        assert "num_constraints" in check_result
        assert isinstance(check_result["overall_score"], float)
        assert 0 <= check_result["overall_score"] <= 1

    def test_check_simulation_constraints_with_failures(self):
        """Test simulation constraints with failures."""
        from aero.symbolic.checks import check_simulation_constraints

        # Simulation that didn't converge
        result = {
            "fields": {
                "solution": np.array([float("nan"), 1.0, 2.0]),
            },
            "metadata": {
                "converged": False,
                "final_residual": 1.0,
            },
        }

        check_result = check_simulation_constraints(result)

        # Should have lower score due to non-convergence
        assert check_result["overall_score"] < 1.0

    def test_check_experiment_constraints_basic(self):
        """Test basic experiment constraint checking."""
        from aero.symbolic.checks import check_experiment_constraints

        # Create a simple experiment result
        result = {
            "data": {
                "values": np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            },
            "metadata": {
                "type": "synthetic_flow",
            },
        }

        check_result = check_experiment_constraints(result)

        assert "constraints" in check_result
        assert "overall_score" in check_result
        assert "passed" in check_result
        assert check_result["sympy_used"] is False  # Experiments don't use symbolic

    def test_get_symbolic_status(self):
        """Test getting symbolic status."""
        from aero.symbolic.checks import get_symbolic_status

        status = get_symbolic_status()

        assert "sympy_available" in status
        assert "enabled" in status
        assert "expressions" in status
        assert isinstance(status["sympy_available"], bool)

    def test_evaluate_constraints_batch(self):
        """Test batch constraint evaluation."""
        from aero.symbolic.checks import evaluate_constraints_batch

        results = [
            {
                "fields": {"solution": np.array([1, 2, 3])},
                "metadata": {"converged": True},
            },
            {
                "fields": {"solution": np.array([1, 2, 3])},
                "metadata": {"converged": True},
            },
        ]

        batch_result = evaluate_constraints_batch(results, result_type="simulation")

        assert "results" in batch_result
        assert "num_results" in batch_result
        assert "average_score" in batch_result
        assert batch_result["num_results"] == 2

    def test_compute_constraint_penalty(self):
        """Test constraint penalty computation."""
        from aero.symbolic.checks import compute_constraint_penalty

        # No failures
        constraint_results = {
            "constraints": [
                {"passed": True, "score": 1.0, "weight": 1.0, "required": False},
                {"passed": True, "score": 1.0, "weight": 1.0, "required": False},
            ],
        }
        penalty = compute_constraint_penalty(constraint_results)
        assert penalty == 0.0

        # Some failures
        constraint_results = {
            "constraints": [
                {"passed": False, "score": 0.5, "weight": 1.0, "required": False},
            ],
        }
        penalty = compute_constraint_penalty(constraint_results)
        assert penalty > 0.0

        # Required constraint failure
        constraint_results = {
            "constraints": [
                {"passed": False, "score": 0.0, "weight": 1.0, "required": True},
            ],
        }
        penalty = compute_constraint_penalty(constraint_results)
        assert penalty > 0.2  # Should have extra penalty


# =============================================================================
# Integration Tests
# =============================================================================


class TestSymbolicIntegration:
    """Integration tests for the symbolic module."""

    def test_full_simulation_validation_flow(self):
        """Test complete simulation validation flow."""
        from aero.symbolic.checks import check_simulation_constraints
        from aero.symbolic.expressions import get_expression_for_sim_type

        # Get symbolic expression
        expr = get_expression_for_sim_type("heat_1d")

        # Create realistic simulation result
        nx = 50
        x = np.linspace(0, 1, nx)
        u0 = np.sin(np.pi * x)
        u_final = u0 * np.exp(-0.1)  # Decayed sine (heat diffusion)

        result = {
            "fields": {
                "solution": u_final,
                "initial": u0,
            },
            "metadata": {
                "simulation_type": "heat_1d",
                "converged": True,
                "final_residual": 1e-10,
                "config": {
                    "alpha": 0.01,
                    "dx": 1.0 / (nx - 1),
                    "dt": 0.0001,
                },
            },
        }

        check_result = check_simulation_constraints(
            result,
            symbolic_expr=expr,
        )

        # Should pass most constraints
        assert check_result["passed"] is True
        assert check_result["overall_score"] > 0.7
        assert check_result["num_passed"] > 0

    def test_module_exports(self):
        """Test that all expected items are exported from __init__."""
        import aero.symbolic as sym

        # Check availability flag
        assert hasattr(sym, "SYM_AVAILABLE")

        # Check expressions
        assert hasattr(sym, "SymbolicExpression")
        assert hasattr(sym, "make_heat_equation_symbolic")
        assert hasattr(sym, "make_laplace_equation_symbolic")
        assert hasattr(sym, "get_expression_for_sim_type")

        # Check dimensions
        assert hasattr(sym, "Dimension")
        assert hasattr(sym, "multiply_dims")
        assert hasattr(sym, "divide_dims")
        assert hasattr(sym, "BASE_DIMENSIONS")
        assert hasattr(sym, "check_dimensional_consistency")

        # Check constraints
        assert hasattr(sym, "Constraint")
        assert hasattr(sym, "ConstraintResult")
        assert hasattr(sym, "make_mass_conservation_constraint")
        assert hasattr(sym, "make_boundedness_constraint")
        assert hasattr(sym, "get_default_simulation_constraints")

        # Check checks
        assert hasattr(sym, "check_simulation_constraints")
        assert hasattr(sym, "check_experiment_constraints")
        assert hasattr(sym, "get_symbolic_status")

    def test_graceful_degradation_without_sympy(self):
        """Test that module works without SymPy (if applicable)."""
        from aero.symbolic import SYM_AVAILABLE
        from aero.symbolic.expressions import make_heat_equation_symbolic

        heat_eq = make_heat_equation_symbolic()

        # Should always have name and metadata
        assert heat_eq.name == "heat_equation"
        assert "description" in heat_eq.metadata

        # If no SymPy, expr should be None but to_dict should work
        d = heat_eq.to_dict()
        assert "name" in d
        assert "available" in d

        if not SYM_AVAILABLE:
            assert heat_eq.expr is None
            assert not heat_eq.is_available()
