"""
Pipeline module for Aero Agent.

Provides the main agent processing loop and validation utilities.
"""

from aero.pipeline.aero_loop import AeroLoop, PipelineTask
from aero.pipeline.validator import Validator, ValidationResult
from aero.pipeline.patterns import Pattern, PatternMatcher

__all__ = [
    "AeroLoop",
    "PipelineTask",
    "Validator",
    "ValidationResult",
    "Pattern",
    "PatternMatcher",
]
