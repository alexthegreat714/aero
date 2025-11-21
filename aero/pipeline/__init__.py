"""
Pipeline module for Aero Agent.

Provides the main agent processing loop, scientific reasoning engine,
hypothesis generation, validation, pattern detection, and refinement.
"""

# Task processing
from aero.pipeline.aero_loop import (
    AeroLoop,
    PipelineTask,
    TaskStatus,
    ScientificReasoningLoop,
    LoopConfig,
    LoopResult,
)

# Hypothesis management
from aero.pipeline.hypothesis import (
    Hypothesis,
    generate_initial_hypotheses,
    generate_secondary_hypotheses,
    detect_domain,
    rank_hypotheses,
    filter_hypotheses,
)

# Simulation planning
from aero.pipeline.planner import (
    SimulationJobConfig,
    ExperimentConfig,
    SimulationType,
    ExperimentType,
    plan_simulations,
    plan_experiments,
    validate_simulation_config,
)

# Validation
from aero.pipeline.validator import (
    Validator,
    ValidationResult,
    SimulationValidationResult,
    validate_simulation,
    validate_experiment,
    check_physical_plausibility,
    compute_error_metrics,
)

# Pattern detection
from aero.pipeline.patterns import (
    Pattern,
    PatternMatcher,
    MatchResult,
    ScientificPattern,
    detect_patterns,
    compare_patterns,
    summarize_patterns,
)

# Hypothesis refinement
from aero.pipeline.refinement import (
    refine_hypotheses,
    merge_hypotheses,
    rank_by_validation,
    generate_refinement_summary,
)

__all__ = [
    # Task processing
    "AeroLoop",
    "PipelineTask",
    "TaskStatus",
    "ScientificReasoningLoop",
    "LoopConfig",
    "LoopResult",
    # Hypothesis
    "Hypothesis",
    "generate_initial_hypotheses",
    "generate_secondary_hypotheses",
    "detect_domain",
    "rank_hypotheses",
    "filter_hypotheses",
    # Planning
    "SimulationJobConfig",
    "ExperimentConfig",
    "SimulationType",
    "ExperimentType",
    "plan_simulations",
    "plan_experiments",
    "validate_simulation_config",
    # Validation
    "Validator",
    "ValidationResult",
    "SimulationValidationResult",
    "validate_simulation",
    "validate_experiment",
    "check_physical_plausibility",
    "compute_error_metrics",
    # Patterns
    "Pattern",
    "PatternMatcher",
    "MatchResult",
    "ScientificPattern",
    "detect_patterns",
    "compare_patterns",
    "summarize_patterns",
    # Refinement
    "refine_hypotheses",
    "merge_hypotheses",
    "rank_by_validation",
    "generate_refinement_summary",
]
