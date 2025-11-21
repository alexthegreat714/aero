"""
Validation utilities for Aero Agent.

Provides data validation and result checking.
"""

import logging
from typing import Any, Callable, Optional
from dataclasses import dataclass, field

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
