"""
Constraint definitions for simulation and experiment validation.

Provides:
- Constraint class for defining validation checks
- Factory functions for common physics constraints
- Mass conservation, energy, boundedness, monotonicity checks
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ConstraintResult:
    """
    Result from evaluating a constraint.

    Attributes:
        passed: Whether the constraint was satisfied
        score: Score from 0 to 1 (1 = fully satisfied)
        details: Human-readable explanation
        data: Optional additional data
    """

    passed: bool
    score: float
    details: str
    data: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "passed": self.passed,
            "score": self.score,
            "details": self.details,
            "data": self.data,
        }


@dataclass
class Constraint:
    """
    Definition of a constraint that can be evaluated on data.

    Attributes:
        name: Unique identifier for the constraint
        description: Human-readable description
        check_fn: Function that evaluates the constraint
        weight: Weight for combining multiple constraints
        required: Whether failure of this constraint is critical
    """

    name: str
    description: str
    check_fn: Callable[[Dict[str, Any]], ConstraintResult]
    weight: float = 1.0
    required: bool = False

    def evaluate(self, data: Dict[str, Any]) -> ConstraintResult:
        """
        Evaluate the constraint on given data.

        Args:
            data: Dictionary containing fields, metadata, etc.

        Returns:
            ConstraintResult
        """
        try:
            return self.check_fn(data)
        except Exception as e:
            logger.warning(f"Constraint '{self.name}' evaluation failed: {e}")
            return ConstraintResult(
                passed=False,
                score=0.0,
                details=f"Evaluation error: {e}",
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (without check_fn)."""
        return {
            "name": self.name,
            "description": self.description,
            "weight": self.weight,
            "required": self.required,
        }


# =============================================================================
# Constraint Factory Functions
# =============================================================================


def make_mass_conservation_constraint(tolerance: float = 0.05) -> Constraint:
    """
    Create a mass conservation constraint.

    Checks that total "mass" (sum of field values) is approximately conserved.

    Args:
        tolerance: Allowed relative change in total mass

    Returns:
        Constraint for mass conservation
    """

    def check_fn(data: Dict[str, Any]) -> ConstraintResult:
        # Get solution field
        fields = data.get("fields", {})
        solution = fields.get("solution") or fields.get("u") or fields.get("temperature")

        if solution is None:
            return ConstraintResult(
                passed=True,
                score=1.0,
                details="No solution field to check",
            )

        solution = np.asarray(solution)

        # Get initial condition if available
        initial = fields.get("initial") or fields.get("u0")
        if initial is not None:
            initial = np.asarray(initial)
            initial_mass = np.sum(initial)
        else:
            # Assume initial mass from metadata or estimate
            initial_mass = data.get("metadata", {}).get("initial_mass")
            if initial_mass is None:
                # Can't check without reference
                return ConstraintResult(
                    passed=True,
                    score=0.8,
                    details="No initial condition for mass comparison",
                )

        final_mass = np.sum(solution)

        if abs(initial_mass) < 1e-10:
            # Avoid division by zero
            relative_change = abs(final_mass)
        else:
            relative_change = abs(final_mass - initial_mass) / abs(initial_mass)

        passed = relative_change <= tolerance
        score = max(0, 1.0 - relative_change / tolerance) if not passed else 1.0

        return ConstraintResult(
            passed=passed,
            score=score,
            details=f"Mass change: {relative_change*100:.2f}% (tolerance: {tolerance*100:.1f}%)",
            data={"initial_mass": float(initial_mass), "final_mass": float(final_mass)},
        )

    return Constraint(
        name="mass_conservation",
        description="Check that total mass is approximately conserved",
        check_fn=check_fn,
        weight=1.0,
    )


def make_energy_like_constraint(tolerance: float = 0.1) -> Constraint:
    """
    Create an energy-like conservation constraint.

    Checks that L2 norm (energy proxy) doesn't grow unboundedly.

    Args:
        tolerance: Allowed relative growth in energy

    Returns:
        Constraint for energy-like conservation
    """

    def check_fn(data: Dict[str, Any]) -> ConstraintResult:
        fields = data.get("fields", {})
        solution = fields.get("solution") or fields.get("u")

        if solution is None:
            return ConstraintResult(
                passed=True,
                score=1.0,
                details="No solution field to check",
            )

        solution = np.asarray(solution)

        # Get initial condition
        initial = fields.get("initial") or fields.get("u0")

        if initial is not None:
            initial = np.asarray(initial)
            initial_energy = np.sum(initial ** 2)
        else:
            initial_energy = data.get("metadata", {}).get("initial_energy")
            if initial_energy is None:
                return ConstraintResult(
                    passed=True,
                    score=0.8,
                    details="No initial condition for energy comparison",
                )

        final_energy = np.sum(solution ** 2)

        if abs(initial_energy) < 1e-10:
            relative_change = abs(final_energy)
        else:
            relative_change = (final_energy - initial_energy) / abs(initial_energy)

        # For dissipative systems, energy should decrease or stay same
        # Allow small growth due to numerical error
        passed = relative_change <= tolerance
        score = max(0, 1.0 - max(0, relative_change) / tolerance) if not passed else 1.0

        return ConstraintResult(
            passed=passed,
            score=score,
            details=f"Energy change: {relative_change*100:.2f}% (tolerance: {tolerance*100:.1f}%)",
            data={"initial_energy": float(initial_energy), "final_energy": float(final_energy)},
        )

    return Constraint(
        name="energy_conservation",
        description="Check that energy (L2 norm) doesn't grow unboundedly",
        check_fn=check_fn,
        weight=1.0,
    )


def make_boundedness_constraint(
    lower: Optional[float] = None,
    upper: Optional[float] = None,
) -> Constraint:
    """
    Create a boundedness constraint.

    Checks that solution values stay within specified bounds.

    Args:
        lower: Minimum allowed value (None = no lower bound)
        upper: Maximum allowed value (None = no upper bound)

    Returns:
        Constraint for boundedness
    """

    def check_fn(data: Dict[str, Any]) -> ConstraintResult:
        fields = data.get("fields", {})
        solution = fields.get("solution") or fields.get("u")

        if solution is None:
            return ConstraintResult(
                passed=True,
                score=1.0,
                details="No solution field to check",
            )

        solution = np.asarray(solution)
        min_val = float(np.min(solution))
        max_val = float(np.max(solution))

        violations = []
        if lower is not None and min_val < lower:
            violations.append(f"min={min_val:.4f} < lower={lower}")
        if upper is not None and max_val > upper:
            violations.append(f"max={max_val:.4f} > upper={upper}")

        passed = len(violations) == 0

        # Compute score based on how far out of bounds
        score = 1.0
        if lower is not None and min_val < lower:
            score *= max(0, 1.0 - abs(min_val - lower) / max(abs(lower), 1))
        if upper is not None and max_val > upper:
            score *= max(0, 1.0 - abs(max_val - upper) / max(abs(upper), 1))

        details = "Within bounds" if passed else f"Violations: {', '.join(violations)}"

        return ConstraintResult(
            passed=passed,
            score=score,
            details=details,
            data={"min": min_val, "max": max_val},
        )

    bounds_str = f"[{lower}, {upper}]"
    return Constraint(
        name="boundedness",
        description=f"Check solution values are within {bounds_str}",
        check_fn=check_fn,
        weight=0.8,
    )


def make_monotonicity_constraint(direction: str = "any") -> Constraint:
    """
    Create a monotonicity constraint.

    Checks if solution exhibits expected monotonic behavior.

    Args:
        direction: "increasing", "decreasing", or "any" (just no wild oscillations)

    Returns:
        Constraint for monotonicity
    """

    def check_fn(data: Dict[str, Any]) -> ConstraintResult:
        fields = data.get("fields", {})
        solution = fields.get("solution") or fields.get("u")

        if solution is None:
            return ConstraintResult(
                passed=True,
                score=1.0,
                details="No solution field to check",
            )

        solution = np.asarray(solution).flatten()

        if len(solution) < 3:
            return ConstraintResult(
                passed=True,
                score=1.0,
                details="Solution too short for monotonicity check",
            )

        # Compute differences
        diffs = np.diff(solution)

        if direction == "increasing":
            violations = np.sum(diffs < -1e-10)
            passed = violations == 0
        elif direction == "decreasing":
            violations = np.sum(diffs > 1e-10)
            passed = violations == 0
        else:  # "any" - check for excessive oscillations
            sign_changes = np.sum(np.diff(np.sign(diffs)) != 0)
            max_expected = len(solution) * 0.3  # Allow some oscillation
            passed = sign_changes < max_expected
            violations = sign_changes

        total = len(diffs)
        score = max(0, 1.0 - violations / total)

        return ConstraintResult(
            passed=passed,
            score=score,
            details=f"Monotonicity ({direction}): {violations} violations in {total} points",
            data={"violations": int(violations), "total_points": total},
        )

    return Constraint(
        name="monotonicity",
        description=f"Check for {direction} monotonicity",
        check_fn=check_fn,
        weight=0.5,
    )


def make_dimensional_consistency_constraint() -> Constraint:
    """
    Create a dimensional consistency constraint.

    Checks that parameters have correct dimensions for the equation type.

    Returns:
        Constraint for dimensional consistency
    """

    def check_fn(data: Dict[str, Any]) -> ConstraintResult:
        from aero.symbolic.dimensions import check_dimensional_consistency

        metadata = data.get("metadata", {})
        sim_type = metadata.get("simulation_type") or metadata.get("type", "unknown")
        config = metadata.get("config", {})

        # Extract parameters
        parameters = {}
        for key in ["alpha", "nu", "viscosity", "dx", "dt", "rho", "c"]:
            if key in config:
                parameters[key] = config[key]
            elif key in metadata:
                parameters[key] = metadata[key]

        result = check_dimensional_consistency(sim_type, parameters)

        passed = result["consistent"]
        errors = result.get("errors", [])

        score = 1.0 if passed else max(0, 1.0 - len(errors) * 0.2)
        details = "Dimensions consistent" if passed else f"Issues: {errors}"

        return ConstraintResult(
            passed=passed,
            score=score,
            details=details,
            data=result,
        )

    return Constraint(
        name="dimensional_consistency",
        description="Check that parameters have correct physical dimensions",
        check_fn=check_fn,
        weight=0.7,
    )


def make_convergence_constraint(threshold: float = 1e-6) -> Constraint:
    """
    Create a convergence constraint.

    Checks that the solution converged (for iterative solvers).

    Args:
        threshold: Convergence threshold

    Returns:
        Constraint for convergence
    """

    def check_fn(data: Dict[str, Any]) -> ConstraintResult:
        metadata = data.get("metadata", {})

        converged = metadata.get("converged")
        final_residual = metadata.get("final_residual")

        if converged is None and final_residual is None:
            return ConstraintResult(
                passed=True,
                score=0.9,
                details="No convergence info available",
            )

        if converged is not None:
            passed = converged
            score = 1.0 if converged else 0.3
            details = "Converged" if converged else "Did not converge"
        elif final_residual is not None:
            passed = final_residual < threshold
            score = max(0, 1.0 - np.log10(final_residual / threshold) / 6) if final_residual > 0 else 1.0
            details = f"Final residual: {final_residual:.2e} (threshold: {threshold:.2e})"
        else:
            passed = True
            score = 0.9
            details = "No convergence info"

        return ConstraintResult(
            passed=passed,
            score=score,
            details=details,
            data={"converged": converged, "final_residual": final_residual},
        )

    return Constraint(
        name="convergence",
        description="Check that iterative solver converged",
        check_fn=check_fn,
        weight=1.0,
        required=True,
    )


def make_noise_level_constraint(max_noise_ratio: float = 0.1) -> Constraint:
    """
    Create a noise level constraint for experiments.

    Checks that signal-to-noise ratio is acceptable.

    Args:
        max_noise_ratio: Maximum allowed noise/signal ratio

    Returns:
        Constraint for noise level
    """

    def check_fn(data: Dict[str, Any]) -> ConstraintResult:
        fields = data.get("fields", data.get("data", {}))

        # Try different field names
        signal = None
        for key in ["values", "signal", "measurements", "velocity"]:
            if key in fields:
                signal = np.asarray(fields[key])
                break

        if signal is None:
            return ConstraintResult(
                passed=True,
                score=0.9,
                details="No signal data to check noise",
            )

        signal = signal.flatten()

        if len(signal) < 10:
            return ConstraintResult(
                passed=True,
                score=0.9,
                details="Signal too short for noise estimation",
            )

        # Estimate noise using high-frequency content
        # Simple approach: difference of consecutive values
        diff = np.diff(signal)
        noise_estimate = np.std(diff) / np.sqrt(2)  # Approximate noise std
        signal_range = np.max(signal) - np.min(signal)

        if signal_range < 1e-10:
            noise_ratio = 0  # Constant signal
        else:
            noise_ratio = noise_estimate / signal_range

        passed = noise_ratio <= max_noise_ratio
        score = max(0, 1.0 - noise_ratio / max_noise_ratio) if not passed else 1.0

        return ConstraintResult(
            passed=passed,
            score=score,
            details=f"Noise ratio: {noise_ratio:.2%} (max: {max_noise_ratio:.2%})",
            data={"noise_ratio": float(noise_ratio), "noise_estimate": float(noise_estimate)},
        )

    return Constraint(
        name="noise_level",
        description="Check that noise level is acceptable",
        check_fn=check_fn,
        weight=0.6,
    )


# =============================================================================
# Default Constraint Sets
# =============================================================================


def get_default_simulation_constraints(
    mass_tol: float = 0.05,
    energy_tol: float = 0.1,
) -> List[Constraint]:
    """
    Get default constraints for simulation validation.

    Args:
        mass_tol: Tolerance for mass conservation
        energy_tol: Tolerance for energy conservation

    Returns:
        List of Constraint objects
    """
    return [
        make_convergence_constraint(),
        make_mass_conservation_constraint(tolerance=mass_tol),
        make_energy_like_constraint(tolerance=energy_tol),
        make_dimensional_consistency_constraint(),
        make_boundedness_constraint(lower=None, upper=None),  # No fixed bounds
    ]


def get_default_experiment_constraints() -> List[Constraint]:
    """
    Get default constraints for experiment validation.

    Returns:
        List of Constraint objects
    """
    return [
        make_boundedness_constraint(lower=None, upper=None),
        make_noise_level_constraint(max_noise_ratio=0.2),
        make_monotonicity_constraint(direction="any"),
    ]
