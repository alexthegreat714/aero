"""
Validation utilities for Aero Agent.

Provides data validation and result checking for the scientific reasoning loop.
Includes simulation validation, experiment validation, and hypothesis checking.
"""

import logging
from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of a validation check."""

    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def add_error(self, message: str) -> None:
        """Add an error message."""
        self.errors.append(message)
        self.valid = False

    def add_warning(self, message: str) -> None:
        """Add a warning message."""
        self.warnings.append(message)

    def merge(self, other: "ValidationResult") -> None:
        """Merge another result into this one."""
        if not other.valid:
            self.valid = False
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "valid": self.valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "metadata": self.metadata,
        }


class Validator:
    """
    Data validator for Aero Agent.

    Provides chainable validation rules for various data types.

    Example:
        validator = Validator()

        result = validator.validate_dict(data, {
            "name": {"type": str, "required": True, "min_length": 1},
            "value": {"type": (int, float), "min": 0, "max": 100},
        })

        if not result.valid:
            print("Validation errors:", result.errors)
    """

    def __init__(self):
        """Initialize the validator."""
        self._custom_validators: dict[str, Callable] = {}

    def validate_dict(
        self,
        data: dict,
        schema: dict,
        strict: bool = False,
    ) -> ValidationResult:
        """
        Validate a dictionary against a schema.

        Args:
            data: Dictionary to validate
            schema: Validation schema
            strict: If True, reject unknown keys

        Schema format:
            {
                "field_name": {
                    "type": type or tuple of types,
                    "required": bool,
                    "default": value,
                    "min": number (for numbers),
                    "max": number (for numbers),
                    "min_length": int (for strings/lists),
                    "max_length": int (for strings/lists),
                    "choices": list of valid values,
                    "validator": callable,
                }
            }

        Returns:
            ValidationResult
        """
        result = ValidationResult(valid=True)

        # Check for required fields
        for field_name, rules in schema.items():
            required = rules.get("required", False)

            if required and field_name not in data:
                result.add_error(f"Missing required field: {field_name}")
                continue

            if field_name not in data:
                continue

            value = data[field_name]

            # Type check
            expected_type = rules.get("type")
            if expected_type and not isinstance(value, expected_type):
                result.add_error(
                    f"Field '{field_name}' has wrong type: "
                    f"expected {expected_type}, got {type(value)}"
                )
                continue

            # Numeric range
            if isinstance(value, (int, float)):
                min_val = rules.get("min")
                max_val = rules.get("max")

                if min_val is not None and value < min_val:
                    result.add_error(f"Field '{field_name}' below minimum: {value} < {min_val}")

                if max_val is not None and value > max_val:
                    result.add_error(f"Field '{field_name}' above maximum: {value} > {max_val}")

            # String/list length
            if isinstance(value, (str, list)):
                min_len = rules.get("min_length")
                max_len = rules.get("max_length")

                if min_len is not None and len(value) < min_len:
                    result.add_error(f"Field '{field_name}' too short: {len(value)} < {min_len}")

                if max_len is not None and len(value) > max_len:
                    result.add_error(f"Field '{field_name}' too long: {len(value)} > {max_len}")

            # Choices
            choices = rules.get("choices")
            if choices is not None and value not in choices:
                result.add_error(f"Field '{field_name}' not in allowed choices: {value}")

            # Custom validator
            custom_validator = rules.get("validator")
            if custom_validator:
                try:
                    if not custom_validator(value):
                        result.add_error(f"Field '{field_name}' failed custom validation")
                except Exception as e:
                    result.add_error(f"Field '{field_name}' validator error: {e}")

        # Check for unknown fields in strict mode
        if strict:
            unknown = set(data.keys()) - set(schema.keys())
            for field_name in unknown:
                result.add_warning(f"Unknown field: {field_name}")

        return result

    def validate_type(
        self,
        value: Any,
        expected: type | tuple,
        name: str = "value",
    ) -> ValidationResult:
        """
        Validate that a value has the expected type.

        Args:
            value: Value to check
            expected: Expected type or tuple of types
            name: Name for error messages

        Returns:
            ValidationResult
        """
        result = ValidationResult(valid=True)

        if not isinstance(value, expected):
            result.add_error(f"{name} has wrong type: expected {expected}, got {type(value)}")

        return result

    def validate_range(
        self,
        value: float | int,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
        name: str = "value",
    ) -> ValidationResult:
        """
        Validate that a number is within range.

        Args:
            value: Value to check
            min_val: Minimum value (inclusive)
            max_val: Maximum value (inclusive)
            name: Name for error messages

        Returns:
            ValidationResult
        """
        result = ValidationResult(valid=True)

        if min_val is not None and value < min_val:
            result.add_error(f"{name} below minimum: {value} < {min_val}")

        if max_val is not None and value > max_val:
            result.add_error(f"{name} above maximum: {value} > {max_val}")

        return result

    def validate_string(
        self,
        value: str,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        pattern: Optional[str] = None,
        name: str = "string",
    ) -> ValidationResult:
        """
        Validate a string.

        Args:
            value: String to check
            min_length: Minimum length
            max_length: Maximum length
            pattern: Regex pattern to match
            name: Name for error messages

        Returns:
            ValidationResult
        """
        import re

        result = ValidationResult(valid=True)

        if not isinstance(value, str):
            result.add_error(f"{name} is not a string")
            return result

        if min_length is not None and len(value) < min_length:
            result.add_error(f"{name} too short: {len(value)} < {min_length}")

        if max_length is not None and len(value) > max_length:
            result.add_error(f"{name} too long: {len(value)} > {max_length}")

        if pattern and not re.match(pattern, value):
            result.add_error(f"{name} does not match pattern: {pattern}")

        return result

    def register_validator(
        self,
        name: str,
        validator: Callable[[Any], bool],
    ) -> None:
        """
        Register a custom validator.

        Args:
            name: Validator name
            validator: Validation function
        """
        self._custom_validators[name] = validator

    def run_custom(self, name: str, value: Any) -> ValidationResult:
        """
        Run a registered custom validator.

        Args:
            name: Validator name
            value: Value to validate

        Returns:
            ValidationResult
        """
        result = ValidationResult(valid=True)

        if name not in self._custom_validators:
            result.add_error(f"Unknown validator: {name}")
            return result

        try:
            if not self._custom_validators[name](value):
                result.add_error(f"Custom validation '{name}' failed")
        except Exception as e:
            result.add_error(f"Validator '{name}' error: {e}")

        return result


# =============================================================================
# Simulation Validation
# =============================================================================


@dataclass
class SimulationValidationResult:
    """Result of simulation validation."""

    valid: bool
    score: float  # 0.0 to 1.0
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    notes: str = ""
    metrics: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "valid": self.valid,
            "score": self.score,
            "issues": self.issues,
            "warnings": self.warnings,
            "notes": self.notes,
            "metrics": self.metrics,
            "metadata": self.metadata,
        }


def validate_simulation(
    hypothesis: Any,
    result: dict,
) -> SimulationValidationResult:
    """
    Validate simulation result against a hypothesis.

    Checks:
    - Convergence
    - Stability
    - Physical plausibility
    - Error metrics (L2 norm)
    - Conservation properties

    Args:
        hypothesis: Hypothesis object being tested
        result: Simulation result dictionary

    Returns:
        SimulationValidationResult with validation details
    """
    logger.info("Validating simulation result")

    issues = []
    warnings = []
    metrics = {}
    score = 1.0

    # Extract result data
    solution = result.get("solution")
    metadata = result.get("metadata", {})
    converged = result.get("converged", metadata.get("converged", True))
    iterations = result.get("iterations", metadata.get("iterations", 0))
    final_residual = result.get("final_residual", metadata.get("final_residual", 0.0))

    # Check convergence
    if not converged:
        issues.append("Simulation did not converge")
        score -= 0.3

    # Check residual
    if final_residual is not None:
        metrics["final_residual"] = final_residual
        if final_residual > 1e-3:
            warnings.append(f"High residual: {final_residual:.2e}")
            score -= 0.1
        elif final_residual > 0.1:
            issues.append(f"Very high residual: {final_residual:.2e}")
            score -= 0.2

    # Check for NaN/Inf in solution
    if solution is not None:
        if isinstance(solution, (list, np.ndarray)):
            solution_array = np.array(solution)
            if np.any(np.isnan(solution_array)):
                issues.append("Solution contains NaN values")
                score -= 0.4
            elif np.any(np.isinf(solution_array)):
                issues.append("Solution contains Inf values")
                score -= 0.4

            # Compute L2 norm
            l2_norm = float(np.linalg.norm(solution_array))
            metrics["l2_norm"] = l2_norm

            # Check for unreasonable magnitudes
            max_val = float(np.max(np.abs(solution_array)))
            metrics["max_value"] = max_val
            if max_val > 1e10:
                issues.append(f"Solution magnitude too large: {max_val:.2e}")
                score -= 0.3

    # Check stability indicator
    stable = result.get("stable", metadata.get("stable", True))
    if not stable:
        issues.append("Unstable timestep detected")
        score -= 0.2

    # Check iteration count
    max_iterations = result.get("max_iterations", metadata.get("max_iterations", 10000))
    if iterations > 0 and iterations >= max_iterations * 0.95:
        warnings.append("Reached near maximum iterations")
        score -= 0.1

    # Conservation checks (if applicable)
    if "conservation_error" in result or "conservation_error" in metadata:
        cons_error = result.get("conservation_error", metadata.get("conservation_error", 0))
        metrics["conservation_error"] = cons_error
        if cons_error > 1e-6:
            warnings.append(f"Conservation error: {cons_error:.2e}")
            score -= 0.05

    # Ensure score is bounded
    score = max(0.0, min(1.0, score))

    # Generate notes
    notes = _generate_validation_notes(hypothesis, result, issues, score)

    validation = SimulationValidationResult(
        valid=len(issues) == 0,
        score=score,
        issues=issues,
        warnings=warnings,
        notes=notes,
        metrics=metrics,
        metadata={
            "hypothesis_id": getattr(hypothesis, "id", None),
            "converged": converged,
            "iterations": iterations,
        },
    )

    logger.info(f"Validation complete: valid={validation.valid}, score={validation.score:.2f}")
    return validation


def validate_experiment(
    hypothesis: Any,
    data: dict,
) -> SimulationValidationResult:
    """
    Validate experiment data against a hypothesis.

    Checks:
    - Data completeness
    - Value ranges
    - Data quality
    - Consistency

    Args:
        hypothesis: Hypothesis object being tested
        data: Experiment data dictionary

    Returns:
        SimulationValidationResult with validation details
    """
    logger.info("Validating experiment data")

    issues = []
    warnings = []
    metrics = {}
    score = 1.0

    # Check if data is present
    if not data:
        return SimulationValidationResult(
            valid=False,
            score=0.0,
            issues=["No experiment data provided"],
            notes="Experiment returned no data.",
        )

    # Check for errors in experiment
    if "error" in data:
        issues.append(f"Experiment error: {data['error']}")
        score -= 0.5

    # Check data completeness
    expected_fields = data.get("expected_fields", [])
    for field_name in expected_fields:
        if field_name not in data:
            warnings.append(f"Missing expected field: {field_name}")
            score -= 0.05

    # Check numerical data quality
    values = data.get("values", data.get("measurements", []))
    if values:
        if isinstance(values, (list, np.ndarray)):
            values_array = np.array(values)

            # Check for NaN
            if np.any(np.isnan(values_array)):
                warnings.append("Data contains NaN values")
                score -= 0.1

            # Basic statistics
            if len(values_array) > 0:
                metrics["mean"] = float(np.nanmean(values_array))
                metrics["std"] = float(np.nanstd(values_array))
                metrics["min"] = float(np.nanmin(values_array))
                metrics["max"] = float(np.nanmax(values_array))
                metrics["count"] = len(values_array)

                # Check for outliers (simple Z-score)
                if metrics["std"] > 0:
                    z_scores = np.abs((values_array - metrics["mean"]) / metrics["std"])
                    outlier_count = int(np.sum(z_scores > 3))
                    if outlier_count > 0:
                        warnings.append(f"Data has {outlier_count} potential outliers")

    # Check OCR confidence (if OCR experiment)
    if "ocr_confidence" in data:
        conf = data["ocr_confidence"]
        metrics["ocr_confidence"] = conf
        if conf < 0.6:
            warnings.append(f"Low OCR confidence: {conf:.2f}")
            score -= 0.15
        elif conf < 0.8:
            warnings.append(f"Moderate OCR confidence: {conf:.2f}")
            score -= 0.05

    # Ensure score is bounded
    score = max(0.0, min(1.0, score))

    # Generate notes
    notes = _generate_experiment_notes(hypothesis, data, issues, score)

    return SimulationValidationResult(
        valid=len(issues) == 0,
        score=score,
        issues=issues,
        warnings=warnings,
        notes=notes,
        metrics=metrics,
        metadata={
            "hypothesis_id": getattr(hypothesis, "id", None),
            "experiment_type": data.get("type", "unknown"),
        },
    )


def _generate_validation_notes(
    hypothesis: Any,
    result: dict,
    issues: List[str],
    score: float,
) -> str:
    """Generate human-readable validation notes."""
    if score >= 0.9:
        return "Simulation results strongly support the hypothesis."
    elif score >= 0.7:
        return "Simulation results are consistent with the hypothesis."
    elif score >= 0.5:
        return "Simulation results partially support the hypothesis with some concerns."
    elif score >= 0.3:
        return "Simulation results show limited support for the hypothesis."
    else:
        return "Simulation results are inconsistent with the hypothesis."


def _generate_experiment_notes(
    hypothesis: Any,
    data: dict,
    issues: List[str],
    score: float,
) -> str:
    """Generate human-readable experiment notes."""
    if score >= 0.9:
        return "Experiment data strongly validates the hypothesis."
    elif score >= 0.7:
        return "Experiment data supports the hypothesis."
    elif score >= 0.5:
        return "Experiment data provides partial support."
    elif score >= 0.3:
        return "Experiment data is inconclusive."
    else:
        return "Experiment data does not support the hypothesis."


def check_physical_plausibility(
    result: dict,
    expected_range: Optional[tuple] = None,
    check_positive: bool = False,
    check_bounded: bool = False,
) -> Dict[str, Any]:
    """
    Check physical plausibility of simulation results.

    Args:
        result: Simulation result dictionary
        expected_range: Optional (min, max) expected value range
        check_positive: Whether values should be positive
        check_bounded: Whether values should be bounded

    Returns:
        Dictionary with plausibility checks
    """
    checks = {
        "plausible": True,
        "issues": [],
    }

    solution = result.get("solution")
    if solution is None:
        return checks

    solution_array = np.array(solution)

    # Check for NaN/Inf
    if np.any(np.isnan(solution_array)) or np.any(np.isinf(solution_array)):
        checks["plausible"] = False
        checks["issues"].append("Solution contains NaN or Inf")

    # Check positivity
    if check_positive and np.any(solution_array < 0):
        checks["plausible"] = False
        checks["issues"].append("Negative values in solution (expected positive)")

    # Check expected range
    if expected_range is not None:
        min_val, max_val = expected_range
        actual_min = float(np.min(solution_array))
        actual_max = float(np.max(solution_array))

        if actual_min < min_val or actual_max > max_val:
            checks["plausible"] = False
            checks["issues"].append(
                f"Values outside expected range [{min_val}, {max_val}]: "
                f"actual [{actual_min:.4f}, {actual_max:.4f}]"
            )

    # Check boundedness
    if check_bounded:
        max_abs = float(np.max(np.abs(solution_array)))
        if max_abs > 1e12:
            checks["plausible"] = False
            checks["issues"].append(f"Unbounded solution: max|u| = {max_abs:.2e}")

    return checks


def compute_error_metrics(
    computed: np.ndarray,
    reference: np.ndarray,
) -> Dict[str, float]:
    """
    Compute error metrics between computed and reference solutions.

    Args:
        computed: Computed solution array
        reference: Reference/expected solution array

    Returns:
        Dictionary of error metrics
    """
    computed = np.array(computed).flatten()
    reference = np.array(reference).flatten()

    if computed.shape != reference.shape:
        return {"error": "Shape mismatch"}

    diff = computed - reference

    return {
        "l2_error": float(np.linalg.norm(diff)),
        "l2_relative": float(np.linalg.norm(diff) / (np.linalg.norm(reference) + 1e-12)),
        "linf_error": float(np.max(np.abs(diff))),
        "mean_error": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff ** 2))),
    }
