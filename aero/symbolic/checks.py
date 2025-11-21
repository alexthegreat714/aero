"""
High-level constraint checking functions for Aero Agent.

Provides unified interface for running symbolic and numeric
constraints on simulation and experiment results.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from aero.symbolic import SYM_AVAILABLE
from aero.symbolic.expressions import SymbolicExpression, get_expression_for_sim_type
from aero.symbolic.constraints import (
    Constraint,
    ConstraintResult,
    get_default_simulation_constraints,
    get_default_experiment_constraints,
)

logger = logging.getLogger(__name__)


def check_simulation_constraints(
    sim_result: Any,
    symbolic_expr: Optional[SymbolicExpression] = None,
    constraints: Optional[List[Constraint]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Run constraint checks on a simulation result.

    Args:
        sim_result: SimulationResult object or dict with fields/metadata
        symbolic_expr: Optional symbolic expression for the PDE
        constraints: Optional list of constraints (uses defaults if None)
        config: Optional configuration overrides

    Returns:
        Dictionary with:
            - constraints: list of individual constraint results
            - overall_score: weighted average score
            - passed: whether all required constraints passed
            - sympy_used: whether symbolic checks were used
    """
    config = config or {}

    # Extract data from sim_result
    if hasattr(sim_result, "fields") and hasattr(sim_result, "metadata"):
        data = {
            "fields": sim_result.fields,
            "metadata": sim_result.metadata,
        }
        if hasattr(sim_result, "tags"):
            data["tags"] = sim_result.tags
    elif isinstance(sim_result, dict):
        data = sim_result
    else:
        data = {"fields": {}, "metadata": {}}

    # Get symbolic expression if not provided
    sim_type = data.get("metadata", {}).get("simulation_type") or data.get("metadata", {}).get("type", "")
    if symbolic_expr is None and sim_type:
        symbolic_expr = get_expression_for_sim_type(sim_type)

    # Get constraints
    if constraints is None:
        mass_tol = config.get("mass_conservation_tol", 0.05)
        energy_tol = config.get("energy_like_tol", 0.1)
        constraints = get_default_simulation_constraints(
            mass_tol=mass_tol,
            energy_tol=energy_tol,
        )

    # Run constraints
    results = []
    total_weight = 0.0
    weighted_score = 0.0
    all_required_passed = True

    for constraint in constraints:
        try:
            result = constraint.evaluate(data)
            results.append({
                "name": constraint.name,
                "description": constraint.description,
                "weight": constraint.weight,
                "required": constraint.required,
                **result.to_dict(),
            })

            total_weight += constraint.weight
            weighted_score += result.score * constraint.weight

            if constraint.required and not result.passed:
                all_required_passed = False

        except Exception as e:
            logger.warning(f"Error evaluating constraint {constraint.name}: {e}")
            results.append({
                "name": constraint.name,
                "description": constraint.description,
                "weight": constraint.weight,
                "required": constraint.required,
                "passed": False,
                "score": 0.0,
                "details": f"Error: {e}",
            })

    # Compute overall score
    overall_score = weighted_score / total_weight if total_weight > 0 else 1.0

    # Build symbolic info
    symbolic_info = {
        "sympy_available": SYM_AVAILABLE,
        "used_symbolic_form": None,
    }

    if symbolic_expr is not None:
        symbolic_info["used_symbolic_form"] = symbolic_expr.name
        if symbolic_expr.is_available():
            symbolic_info["expression"] = str(symbolic_expr.expr)

    return {
        "constraints": results,
        "overall_score": overall_score,
        "passed": all_required_passed and overall_score >= 0.5,
        "sympy_used": SYM_AVAILABLE and symbolic_expr is not None and symbolic_expr.is_available(),
        "symbolic": symbolic_info,
        "num_constraints": len(results),
        "num_passed": sum(1 for r in results if r.get("passed", False)),
    }


def check_experiment_constraints(
    exp_result: Any,
    constraints: Optional[List[Constraint]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Run constraint checks on an experiment result.

    Args:
        exp_result: ExperimentResult object or dict with data/metadata
        constraints: Optional list of constraints (uses defaults if None)
        config: Optional configuration overrides

    Returns:
        Dictionary with constraint results
    """
    config = config or {}

    # Extract data from exp_result
    if hasattr(exp_result, "data") and hasattr(exp_result, "metadata"):
        data = {
            "fields": exp_result.data,
            "data": exp_result.data,
            "metadata": exp_result.metadata,
        }
        if hasattr(exp_result, "experiment_type"):
            data["metadata"]["type"] = exp_result.experiment_type
    elif isinstance(exp_result, dict):
        data = exp_result
        if "data" in data and "fields" not in data:
            data["fields"] = data["data"]
    else:
        data = {"fields": {}, "data": {}, "metadata": {}}

    # Get constraints
    if constraints is None:
        constraints = get_default_experiment_constraints()

    # Run constraints
    results = []
    total_weight = 0.0
    weighted_score = 0.0
    all_required_passed = True

    for constraint in constraints:
        try:
            result = constraint.evaluate(data)
            results.append({
                "name": constraint.name,
                "description": constraint.description,
                "weight": constraint.weight,
                "required": constraint.required,
                **result.to_dict(),
            })

            total_weight += constraint.weight
            weighted_score += result.score * constraint.weight

            if constraint.required and not result.passed:
                all_required_passed = False

        except Exception as e:
            logger.warning(f"Error evaluating constraint {constraint.name}: {e}")
            results.append({
                "name": constraint.name,
                "passed": False,
                "score": 0.0,
                "details": f"Error: {e}",
            })

    overall_score = weighted_score / total_weight if total_weight > 0 else 1.0

    return {
        "constraints": results,
        "overall_score": overall_score,
        "passed": all_required_passed and overall_score >= 0.5,
        "sympy_used": False,  # Experiments don't use symbolic PDEs
        "num_constraints": len(results),
        "num_passed": sum(1 for r in results if r.get("passed", False)),
    }


def get_symbolic_status() -> Dict[str, Any]:
    """
    Get status of the symbolic engine.

    Returns:
        Dictionary with symbolic system status
    """
    from aero.symbolic.expressions import list_available_expressions

    status = {
        "sympy_available": SYM_AVAILABLE,
        "enabled": True,  # Will be overridden by config check
        "expressions": {},
    }

    # Try to get config
    try:
        from aero.config.loader import get_config
        config = get_config()
        status["enabled"] = config.get("symbolic.enable", True)
    except Exception:
        pass

    # Get available expressions
    if SYM_AVAILABLE:
        expressions = list_available_expressions()
        for name, expr in expressions.items():
            status["expressions"][name] = {
                "available": expr.is_available(),
                "description": expr.metadata.get("description", ""),
                "type": expr.metadata.get("type", ""),
            }
    else:
        # List expressions even without SymPy
        status["expressions"] = {
            "heat_equation": {"available": False, "description": "Heat equation: u_t = α u_xx"},
            "laplace_equation": {"available": False, "description": "Laplace equation: ∇²u = 0"},
            "wave_equation": {"available": False, "description": "Wave equation: u_tt = c² u_xx"},
            "navier_stokes": {"available": False, "description": "Navier-Stokes (placeholder)"},
        }

    return status


def evaluate_constraints_batch(
    results: List[Any],
    result_type: str = "simulation",
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Evaluate constraints on a batch of results.

    Args:
        results: List of simulation or experiment results
        result_type: "simulation" or "experiment"
        config: Optional configuration

    Returns:
        Aggregated constraint results
    """
    config = config or {}

    all_results = []
    total_score = 0.0
    num_passed = 0

    check_fn = check_simulation_constraints if result_type == "simulation" else check_experiment_constraints

    for result in results:
        check_result = check_fn(result, config=config)
        all_results.append(check_result)
        total_score += check_result["overall_score"]
        if check_result["passed"]:
            num_passed += 1

    avg_score = total_score / len(results) if results else 0.0

    return {
        "results": all_results,
        "num_results": len(results),
        "num_passed": num_passed,
        "pass_rate": num_passed / len(results) if results else 0.0,
        "average_score": avg_score,
    }


def compute_constraint_penalty(
    constraint_results: Dict[str, Any],
    base_penalty: float = 0.2,
) -> float:
    """
    Compute a penalty score based on constraint failures.

    Used for hypothesis refinement to penalize hypotheses
    whose simulations violate constraints.

    Args:
        constraint_results: Output from check_simulation_constraints
        base_penalty: Base penalty per failed constraint

    Returns:
        Total penalty (0 to 1)
    """
    if not constraint_results:
        return 0.0

    constraints = constraint_results.get("constraints", [])
    if not constraints:
        return 0.0

    total_penalty = 0.0

    for c in constraints:
        if not c.get("passed", True):
            weight = c.get("weight", 1.0)
            score = c.get("score", 0.0)

            # Higher penalty for lower scores and higher weights
            penalty = base_penalty * weight * (1.0 - score)
            total_penalty += penalty

            # Extra penalty for required constraints
            if c.get("required", False):
                total_penalty += base_penalty * 0.5

    # Cap at 1.0
    return min(1.0, total_penalty)
