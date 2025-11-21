"""
Pattern matching utilities for Aero Agent.

Provides pattern recognition and matching for agent workflows,
including detection of scientific patterns in simulation results.
"""

import logging
import re
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MatchResult:
    """Result of a pattern match."""

    matched: bool
    pattern_name: str
    captures: dict = field(default_factory=dict)
    score: float = 1.0
    metadata: dict = field(default_factory=dict)


class Pattern(ABC):
    """Base class for patterns."""

    def __init__(self, name: str):
        """
        Initialize the pattern.

        Args:
            name: Pattern name
        """
        self.name = name

    @abstractmethod
    def match(self, input_data: Any) -> MatchResult:
        """
        Attempt to match the pattern against input.

        Args:
            input_data: Data to match against

        Returns:
            MatchResult
        """
        raise NotImplementedError


class RegexPattern(Pattern):
    """Pattern based on regular expression."""

    def __init__(
        self,
        name: str,
        pattern: str,
        flags: int = 0,
        groups: Optional[list[str]] = None,
    ):
        """
        Initialize a regex pattern.

        Args:
            name: Pattern name
            pattern: Regex pattern string
            flags: Regex flags
            groups: Names for capture groups
        """
        super().__init__(name)
        self.pattern = pattern
        self.regex = re.compile(pattern, flags)
        self.groups = groups or []

    def match(self, input_data: Any) -> MatchResult:
        """Match against string input."""
        if not isinstance(input_data, str):
            return MatchResult(matched=False, pattern_name=self.name)

        match = self.regex.search(input_data)

        if not match:
            return MatchResult(matched=False, pattern_name=self.name)

        # Extract captures
        captures = {}

        # Named groups
        captures.update(match.groupdict())

        # Numbered groups with optional names
        for i, group in enumerate(match.groups(), 1):
            if i <= len(self.groups):
                captures[self.groups[i - 1]] = group
            else:
                captures[f"group_{i}"] = group

        return MatchResult(
            matched=True,
            pattern_name=self.name,
            captures=captures,
        )


class KeywordPattern(Pattern):
    """Pattern based on keyword matching."""

    def __init__(
        self,
        name: str,
        keywords: list[str],
        require_all: bool = False,
        case_sensitive: bool = False,
    ):
        """
        Initialize a keyword pattern.

        Args:
            name: Pattern name
            keywords: Keywords to match
            require_all: If True, all keywords must be present
            case_sensitive: Case sensitive matching
        """
        super().__init__(name)
        self.keywords = keywords
        self.require_all = require_all
        self.case_sensitive = case_sensitive

    def match(self, input_data: Any) -> MatchResult:
        """Match against string input."""
        if not isinstance(input_data, str):
            return MatchResult(matched=False, pattern_name=self.name)

        text = input_data if self.case_sensitive else input_data.lower()
        keywords = (
            self.keywords if self.case_sensitive
            else [k.lower() for k in self.keywords]
        )

        found = [k for k in keywords if k in text]

        if self.require_all:
            matched = len(found) == len(keywords)
        else:
            matched = len(found) > 0

        score = len(found) / len(keywords) if keywords else 0

        return MatchResult(
            matched=matched,
            pattern_name=self.name,
            captures={"found_keywords": found},
            score=score,
        )


class DictPattern(Pattern):
    """Pattern based on dictionary structure."""

    def __init__(
        self,
        name: str,
        required_keys: Optional[list[str]] = None,
        optional_keys: Optional[list[str]] = None,
        key_types: Optional[dict[str, type]] = None,
    ):
        """
        Initialize a dictionary pattern.

        Args:
            name: Pattern name
            required_keys: Keys that must be present
            optional_keys: Keys that may be present
            key_types: Expected types for keys
        """
        super().__init__(name)
        self.required_keys = required_keys or []
        self.optional_keys = optional_keys or []
        self.key_types = key_types or {}

    def match(self, input_data: Any) -> MatchResult:
        """Match against dictionary input."""
        if not isinstance(input_data, dict):
            return MatchResult(matched=False, pattern_name=self.name)

        # Check required keys
        missing = [k for k in self.required_keys if k not in input_data]
        if missing:
            return MatchResult(
                matched=False,
                pattern_name=self.name,
                captures={"missing_keys": missing},
            )

        # Check types
        type_errors = []
        for key, expected_type in self.key_types.items():
            if key in input_data and not isinstance(input_data[key], expected_type):
                type_errors.append(key)

        if type_errors:
            return MatchResult(
                matched=False,
                pattern_name=self.name,
                captures={"type_errors": type_errors},
            )

        return MatchResult(
            matched=True,
            pattern_name=self.name,
            captures={"matched_keys": list(input_data.keys())},
        )


class PatternMatcher:
    """
    Pattern matching engine for Aero Agent.

    Manages multiple patterns and finds matches.

    Example:
        matcher = PatternMatcher()
        matcher.add_pattern(RegexPattern("email", r"[\w.]+@[\w.]+"))
        matcher.add_pattern(KeywordPattern("greeting", ["hello", "hi", "hey"]))

        results = matcher.match_all("Hello, my email is test@example.com")
    """

    def __init__(self):
        """Initialize the pattern matcher."""
        self._patterns: dict[str, Pattern] = {}

    def add_pattern(self, pattern: Pattern) -> None:
        """
        Add a pattern to the matcher.

        Args:
            pattern: Pattern to add
        """
        self._patterns[pattern.name] = pattern
        logger.debug(f"Added pattern: {pattern.name}")

    def remove_pattern(self, name: str) -> bool:
        """
        Remove a pattern.

        Args:
            name: Pattern name

        Returns:
            True if removed
        """
        if name in self._patterns:
            del self._patterns[name]
            return True
        return False

    def match(self, input_data: Any, pattern_name: str) -> MatchResult:
        """
        Match against a specific pattern.

        Args:
            input_data: Data to match
            pattern_name: Pattern to use

        Returns:
            MatchResult
        """
        if pattern_name not in self._patterns:
            return MatchResult(matched=False, pattern_name=pattern_name)

        return self._patterns[pattern_name].match(input_data)

    def match_all(self, input_data: Any) -> list[MatchResult]:
        """
        Match against all patterns.

        Args:
            input_data: Data to match

        Returns:
            List of all match results (including non-matches)
        """
        return [
            pattern.match(input_data)
            for pattern in self._patterns.values()
        ]

    def find_matches(self, input_data: Any) -> list[MatchResult]:
        """
        Find all matching patterns.

        Args:
            input_data: Data to match

        Returns:
            List of successful matches only
        """
        return [
            result for result in self.match_all(input_data)
            if result.matched
        ]

    def best_match(self, input_data: Any) -> Optional[MatchResult]:
        """
        Find the best matching pattern (highest score).

        Args:
            input_data: Data to match

        Returns:
            Best match or None
        """
        matches = self.find_matches(input_data)

        if not matches:
            return None

        return max(matches, key=lambda m: m.score)

    def list_patterns(self) -> list[str]:
        """List all pattern names."""
        return list(self._patterns.keys())


# =============================================================================
# Scientific Pattern Detection
# =============================================================================


@dataclass
class ScientificPattern:
    """
    Represents a detected scientific pattern in simulation results.

    Attributes:
        type: Pattern type (symmetry, periodicity, convergence, etc.)
        confidence: Detection confidence (0.0 to 1.0)
        location: Where the pattern was detected
        parameters: Pattern-specific parameters
        description: Human-readable description
    """

    type: str
    confidence: float = 0.5
    location: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    description: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "type": self.type,
            "confidence": self.confidence,
            "location": self.location,
            "parameters": self.parameters,
            "description": self.description,
        }


def detect_patterns(result: dict) -> List[ScientificPattern]:
    """
    Detect scientific patterns in simulation results.

    Analyzes the solution field for common physical patterns:
    - Spatial symmetry
    - Periodicity
    - Convergence/divergence
    - Shock-like gradients
    - Diffusion profiles
    - Oscillations
    - Steady-state

    Args:
        result: Simulation result dictionary containing solution array

    Returns:
        List of detected ScientificPattern objects
    """
    logger.info("Detecting patterns in simulation result")

    patterns = []
    solution = result.get("solution")

    if solution is None:
        logger.warning("No solution data in result")
        return patterns

    solution = np.array(solution)

    # Handle 1D and 2D cases
    if solution.ndim == 1:
        patterns.extend(_detect_1d_patterns(solution))
    elif solution.ndim == 2:
        patterns.extend(_detect_2d_patterns(solution))
    else:
        logger.warning(f"Unsupported solution dimension: {solution.ndim}")

    # Check convergence metadata
    if result.get("converged", result.get("metadata", {}).get("converged")):
        patterns.append(ScientificPattern(
            type="convergence",
            confidence=0.9,
            description="Solution has converged to steady state",
            parameters={"iterations": result.get("iterations", 0)},
        ))

    # Check for steady state
    steady_pattern = _detect_steady_state(result)
    if steady_pattern:
        patterns.append(steady_pattern)

    logger.info(f"Detected {len(patterns)} patterns")
    return patterns


def _detect_1d_patterns(solution: np.ndarray) -> List[ScientificPattern]:
    """Detect patterns in 1D solution."""
    patterns = []
    n = len(solution)

    # Check for symmetry about midpoint
    mid = n // 2
    left = solution[:mid]
    right = solution[mid:][::-1][:len(left)]

    if len(left) > 0 and len(right) > 0:
        sym_error = np.mean(np.abs(left - right)) / (np.max(np.abs(solution)) + 1e-12)
        if sym_error < 0.05:
            patterns.append(ScientificPattern(
                type="symmetry",
                confidence=max(0.5, 1.0 - sym_error * 10),
                location="midpoint",
                description="Solution is symmetric about the midpoint",
                parameters={"symmetry_error": float(sym_error)},
            ))

    # Check for monotonicity
    diff = np.diff(solution)
    if np.all(diff >= -1e-10):
        patterns.append(ScientificPattern(
            type="monotonic_increasing",
            confidence=0.9,
            description="Solution is monotonically increasing",
        ))
    elif np.all(diff <= 1e-10):
        patterns.append(ScientificPattern(
            type="monotonic_decreasing",
            confidence=0.9,
            description="Solution is monotonically decreasing",
        ))

    # Check for periodicity (FFT-based)
    periodic_pattern = _detect_periodicity_1d(solution)
    if periodic_pattern:
        patterns.append(periodic_pattern)

    # Check for diffusion profile (bell curve-like)
    diffusion_pattern = _detect_diffusion_profile(solution)
    if diffusion_pattern:
        patterns.append(diffusion_pattern)

    # Check for oscillations
    osc_pattern = _detect_oscillations_1d(solution)
    if osc_pattern:
        patterns.append(osc_pattern)

    # Check for shock-like gradients
    shock_pattern = _detect_shock_1d(solution)
    if shock_pattern:
        patterns.append(shock_pattern)

    return patterns


def _detect_2d_patterns(solution: np.ndarray) -> List[ScientificPattern]:
    """Detect patterns in 2D solution."""
    patterns = []
    nx, ny = solution.shape

    # Check for x-symmetry
    left = solution[:nx//2, :]
    right = solution[nx//2:, :][::-1, :]
    if left.shape == right.shape:
        x_sym_error = np.mean(np.abs(left - right)) / (np.max(np.abs(solution)) + 1e-12)
        if x_sym_error < 0.05:
            patterns.append(ScientificPattern(
                type="x_symmetry",
                confidence=max(0.5, 1.0 - x_sym_error * 10),
                location="x=center",
                description="Solution is symmetric about x-midline",
                parameters={"symmetry_error": float(x_sym_error)},
            ))

    # Check for y-symmetry
    bottom = solution[:, :ny//2]
    top = solution[:, ny//2:][:, ::-1]
    if bottom.shape == top.shape:
        y_sym_error = np.mean(np.abs(bottom - top)) / (np.max(np.abs(solution)) + 1e-12)
        if y_sym_error < 0.05:
            patterns.append(ScientificPattern(
                type="y_symmetry",
                confidence=max(0.5, 1.0 - y_sym_error * 10),
                location="y=center",
                description="Solution is symmetric about y-midline",
                parameters={"symmetry_error": float(y_sym_error)},
            ))

    # Check for uniformity in one direction
    row_std = np.mean(np.std(solution, axis=0))
    col_std = np.mean(np.std(solution, axis=1))
    total_std = np.std(solution)

    if row_std < 0.1 * total_std:
        patterns.append(ScientificPattern(
            type="x_uniform",
            confidence=0.8,
            description="Solution is approximately uniform in x-direction",
        ))

    if col_std < 0.1 * total_std:
        patterns.append(ScientificPattern(
            type="y_uniform",
            confidence=0.8,
            description="Solution is approximately uniform in y-direction",
        ))

    # Check for gradient patterns
    grad_x = np.gradient(solution, axis=0)
    grad_y = np.gradient(solution, axis=1)

    if np.std(grad_x) < 0.1 * np.max(np.abs(grad_x) + 1e-12):
        patterns.append(ScientificPattern(
            type="linear_gradient_x",
            confidence=0.7,
            description="Solution has approximately linear gradient in x",
        ))

    if np.std(grad_y) < 0.1 * np.max(np.abs(grad_y) + 1e-12):
        patterns.append(ScientificPattern(
            type="linear_gradient_y",
            confidence=0.7,
            description="Solution has approximately linear gradient in y",
        ))

    return patterns


def _detect_periodicity_1d(solution: np.ndarray) -> Optional[ScientificPattern]:
    """Detect periodic patterns using FFT."""
    n = len(solution)
    if n < 10:
        return None

    # Remove mean
    centered = solution - np.mean(solution)

    # FFT
    fft = np.fft.rfft(centered)
    power = np.abs(fft) ** 2

    # Find dominant frequency (excluding DC)
    if len(power) < 2:
        return None

    dominant_idx = np.argmax(power[1:]) + 1
    dominant_power = power[dominant_idx]
    total_power = np.sum(power[1:])

    if total_power < 1e-12:
        return None

    # Check if dominant frequency is significant
    ratio = dominant_power / total_power
    if ratio > 0.5:
        period = n / dominant_idx
        return ScientificPattern(
            type="periodicity",
            confidence=min(0.95, ratio),
            description=f"Solution exhibits periodicity with period ~{period:.1f} points",
            parameters={
                "period": float(period),
                "dominant_frequency": int(dominant_idx),
                "power_ratio": float(ratio),
            },
        )

    return None


def _detect_diffusion_profile(solution: np.ndarray) -> Optional[ScientificPattern]:
    """Detect Gaussian/diffusion-like profiles."""
    n = len(solution)
    if n < 10:
        return None

    # Check if solution has a single peak
    max_idx = np.argmax(solution)
    max_val = solution[max_idx]
    min_val = np.min(solution)

    if max_val - min_val < 1e-10:
        return None

    # Check if values decrease away from peak
    left_monotonic = all(
        solution[i] >= solution[i-1]
        for i in range(1, max_idx)
    ) if max_idx > 0 else True

    right_monotonic = all(
        solution[i] <= solution[i-1]
        for i in range(max_idx + 1, n)
    ) if max_idx < n - 1 else True

    if left_monotonic and right_monotonic:
        # Estimate width (FWHM-like)
        half_max = (max_val + min_val) / 2
        above_half = np.where(solution > half_max)[0]
        if len(above_half) > 1:
            width = above_half[-1] - above_half[0]
            return ScientificPattern(
                type="diffusion_profile",
                confidence=0.75,
                location=f"peak at index {max_idx}",
                description="Solution shows diffusion-like bell curve profile",
                parameters={
                    "peak_location": int(max_idx),
                    "peak_value": float(max_val),
                    "width": int(width),
                },
            )

    return None


def _detect_oscillations_1d(solution: np.ndarray) -> Optional[ScientificPattern]:
    """Detect oscillatory behavior."""
    n = len(solution)
    if n < 10:
        return None

    # Count zero crossings (relative to mean)
    centered = solution - np.mean(solution)
    zero_crossings = np.sum(np.abs(np.diff(np.sign(centered))) > 0)

    # Many zero crossings indicate oscillations
    crossing_rate = zero_crossings / n

    if crossing_rate > 0.2:
        return ScientificPattern(
            type="oscillations",
            confidence=min(0.9, crossing_rate * 2),
            description="Solution exhibits oscillatory behavior",
            parameters={
                "zero_crossings": int(zero_crossings),
                "crossing_rate": float(crossing_rate),
            },
        )

    return None


def _detect_shock_1d(solution: np.ndarray) -> Optional[ScientificPattern]:
    """Detect shock-like steep gradients."""
    n = len(solution)
    if n < 10:
        return None

    gradient = np.gradient(solution)
    max_grad_idx = np.argmax(np.abs(gradient))
    max_grad = np.abs(gradient[max_grad_idx])
    mean_grad = np.mean(np.abs(gradient))

    if mean_grad < 1e-12:
        return None

    # Check if there's a localized steep gradient
    ratio = max_grad / mean_grad

    if ratio > 5:
        return ScientificPattern(
            type="shock_gradient",
            confidence=min(0.9, ratio / 10),
            location=f"index {max_grad_idx}",
            description="Solution has shock-like steep gradient",
            parameters={
                "location": int(max_grad_idx),
                "gradient_ratio": float(ratio),
                "max_gradient": float(max_grad),
            },
        )

    return None


def _detect_steady_state(result: dict) -> Optional[ScientificPattern]:
    """Detect if solution has reached steady state."""
    # Check from metadata
    converged = result.get("converged", result.get("metadata", {}).get("converged", False))
    final_residual = result.get("final_residual", result.get("metadata", {}).get("final_residual"))

    if converged and final_residual is not None and final_residual < 1e-6:
        return ScientificPattern(
            type="steady_state",
            confidence=0.95,
            description="Solution has reached steady state",
            parameters={"final_residual": float(final_residual)},
        )

    return None


def compare_patterns(
    previous_results: List[dict],
    current: dict,
) -> Dict[str, Any]:
    """
    Compare patterns between previous and current results.

    Useful for detecting pattern evolution and convergence.

    Args:
        previous_results: List of previous simulation results
        current: Current simulation result

    Returns:
        Dictionary with comparison results
    """
    comparison = {
        "new_patterns": [],
        "persistent_patterns": [],
        "disappeared_patterns": [],
        "pattern_evolution": [],
        "converging": False,
    }

    if not previous_results:
        current_patterns = detect_patterns(current)
        comparison["new_patterns"] = [p.type for p in current_patterns]
        return comparison

    # Get patterns from most recent previous result
    prev_patterns = detect_patterns(previous_results[-1])
    prev_types = {p.type for p in prev_patterns}

    current_patterns = detect_patterns(current)
    current_types = {p.type for p in current_patterns}

    # Find new, persistent, and disappeared patterns
    comparison["new_patterns"] = list(current_types - prev_types)
    comparison["persistent_patterns"] = list(current_types & prev_types)
    comparison["disappeared_patterns"] = list(prev_types - current_types)

    # Check for convergence (patterns becoming more stable)
    if "convergence" in current_types or "steady_state" in current_types:
        comparison["converging"] = True

    # Track pattern evolution across all previous results
    if len(previous_results) > 1:
        pattern_history = []
        for res in previous_results:
            patterns = detect_patterns(res)
            pattern_history.append({p.type for p in patterns})

        # Check if patterns are stabilizing
        if len(pattern_history) >= 2:
            last_change = len(pattern_history[-1].symmetric_difference(pattern_history[-2]))
            if last_change == 0:
                comparison["converging"] = True

    return comparison


def summarize_patterns(patterns: List[ScientificPattern]) -> str:
    """
    Generate human-readable summary of detected patterns.

    Args:
        patterns: List of ScientificPattern objects

    Returns:
        Summary string
    """
    if not patterns:
        return "No significant patterns detected."

    high_confidence = [p for p in patterns if p.confidence >= 0.7]
    medium_confidence = [p for p in patterns if 0.4 <= p.confidence < 0.7]

    parts = []

    if high_confidence:
        types = [p.type.replace("_", " ") for p in high_confidence]
        parts.append(f"Strong patterns: {', '.join(types)}")

    if medium_confidence:
        types = [p.type.replace("_", " ") for p in medium_confidence]
        parts.append(f"Possible patterns: {', '.join(types)}")

    return ". ".join(parts) if parts else "No significant patterns detected."
