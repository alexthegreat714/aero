"""
Pattern matching utilities for Aero Agent.

Provides pattern recognition and matching for agent workflows.
"""

import logging
import re
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional
from dataclasses import dataclass, field

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
