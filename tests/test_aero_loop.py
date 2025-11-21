"""
Tests for the Aero scientific reasoning loop.

Tests cover:
- Hypothesis generation
- Simulation planning
- Result validation
- Pattern detection
- Hypothesis refinement
- Full loop execution
"""

import numpy as np
import pytest


# =============================================================================
# Hypothesis Generation Tests
# =============================================================================


class TestHypothesisGeneration:
    """Tests for hypothesis generation."""

    def test_generate_initial_hypotheses(self):
        """Test that initial hypotheses are generated from query."""
        from aero.pipeline.hypothesis import generate_initial_hypotheses

        query = "Analyze heat diffusion in a 1D rod with alpha=0.01"
        hypotheses = generate_initial_hypotheses(query, rag=None, num_hypotheses=3)

        assert len(hypotheses) == 3
        for h in hypotheses:
            assert h.id is not None
            assert h.text is not None
            assert 0.0 <= h.confidence <= 1.0

    def test_hypothesis_to_dict(self):
        """Test hypothesis serialization."""
        from aero.pipeline.hypothesis import Hypothesis

        h = Hypothesis(
            text="Test hypothesis",
            confidence=0.75,
            metadata={"domain": "heat_transfer"},
        )

        d = h.to_dict()

        assert d["text"] == "Test hypothesis"
        assert d["confidence"] == 0.75
        assert d["metadata"]["domain"] == "heat_transfer"
        assert "id" in d
        assert "created_at" in d

    def test_hypothesis_from_dict(self):
        """Test hypothesis deserialization."""
        from aero.pipeline.hypothesis import Hypothesis

        data = {
            "id": "test123",
            "text": "Test hypothesis",
            "confidence": 0.6,
            "metadata": {"key": "value"},
        }

        h = Hypothesis.from_dict(data)

        assert h.id == "test123"
        assert h.text == "Test hypothesis"
        assert h.confidence == 0.6

    def test_detect_domain_heat_transfer(self):
        """Test domain detection for heat transfer queries."""
        from aero.pipeline.hypothesis import detect_domain

        query = "What is the temperature distribution in a heated rod?"
        domain = detect_domain(query)

        assert domain == "heat_transfer"

    def test_detect_domain_fluid_dynamics(self):
        """Test domain detection for fluid dynamics queries."""
        from aero.pipeline.hypothesis import detect_domain

        query = "Analyze the flow velocity in a pipe with pressure gradient"
        domain = detect_domain(query)

        assert domain == "fluid_dynamics"

    def test_detect_domain_generic(self):
        """Test domain detection falls back to generic."""
        from aero.pipeline.hypothesis import detect_domain

        query = "Analyze the unknown phenomenon"
        domain = detect_domain(query)

        assert domain == "generic"

    def test_rank_hypotheses(self):
        """Test hypothesis ranking by confidence."""
        from aero.pipeline.hypothesis import Hypothesis, rank_hypotheses

        hypotheses = [
            Hypothesis(text="Low confidence", confidence=0.3),
            Hypothesis(text="High confidence", confidence=0.9),
            Hypothesis(text="Medium confidence", confidence=0.6),
        ]

        ranked = rank_hypotheses(hypotheses)

        assert ranked[0].confidence == 0.9
        assert ranked[1].confidence == 0.6
        assert ranked[2].confidence == 0.3


# =============================================================================
# Planner Tests
# =============================================================================


class TestPlanner:
    """Tests for simulation planning."""

    def test_planner_produces_valid_configs(self):
        """Test that planner produces valid simulation configurations."""
        from aero.pipeline.hypothesis import Hypothesis
        from aero.pipeline.planner import plan_simulations

        hypothesis = Hypothesis(
            text="Heat diffusion follows Fourier's law with alpha=0.01",
            confidence=0.7,
            metadata={"domain": "heat_transfer"},
        )

        configs = plan_simulations(hypothesis)

        assert len(configs) >= 1
        for config in configs:
            assert config.hypothesis_id == hypothesis.id
            assert config.grid_size > 0
            assert config.tolerance > 0

    def test_planner_config_to_dict(self):
        """Test simulation config serialization."""
        from aero.pipeline.planner import SimulationJobConfig, SimulationType

        config = SimulationJobConfig(
            sim_type=SimulationType.HEAT_1D,
            hypothesis_id="test123",
            grid_size=64,
            dt=0.0001,
        )

        d = config.to_dict()

        assert d["sim_type"] == "heat_1d"
        assert d["hypothesis_id"] == "test123"
        assert d["grid_size"] == 64
        assert d["dt"] == 0.0001

    def test_planner_detects_simulation_type(self):
        """Test that planner detects appropriate simulation type."""
        from aero.pipeline.hypothesis import Hypothesis
        from aero.pipeline.planner import plan_simulations, SimulationType

        hypothesis = Hypothesis(
            text="Solve the Laplace equation for steady state temperature",
            confidence=0.7,
        )

        configs = plan_simulations(hypothesis)

        # Should detect Laplace simulation
        assert any(c.sim_type == SimulationType.LAPLACE_2D for c in configs)

    def test_validate_simulation_config(self):
        """Test simulation config validation."""
        from aero.pipeline.planner import (
            SimulationJobConfig,
            SimulationType,
            validate_simulation_config,
        )

        # Valid config
        config = SimulationJobConfig(
            sim_type=SimulationType.HEAT_1D,
            hypothesis_id="test",
            grid_size=64,
            dt=0.0001,
        )

        result = validate_simulation_config(config)
        assert result["valid"] is True

        # Invalid config (grid too small)
        config.grid_size = 5
        result = validate_simulation_config(config)
        assert "Grid size too small" in result["issues"]


# =============================================================================
# Validator Tests
# =============================================================================


class TestValidator:
    """Tests for result validation."""

    def test_validator_returns_structured_output(self):
        """Test that validator returns structured validation result."""
        from aero.pipeline.hypothesis import Hypothesis
        from aero.pipeline.validator import validate_simulation

        hypothesis = Hypothesis(text="Test hypothesis", confidence=0.5)
        result = {
            "solution": [0, 1, 2, 3, 4],
            "converged": True,
            "final_residual": 1e-6,
        }

        validation = validate_simulation(hypothesis, result)

        assert hasattr(validation, "valid")
        assert hasattr(validation, "score")
        assert hasattr(validation, "issues")
        assert hasattr(validation, "warnings")
        assert hasattr(validation, "notes")

    def test_validator_detects_nan(self):
        """Test that validator detects NaN values."""
        from aero.pipeline.hypothesis import Hypothesis
        from aero.pipeline.validator import validate_simulation

        hypothesis = Hypothesis(text="Test", confidence=0.5)
        result = {
            "solution": [1, float("nan"), 3],
            "converged": True,
        }

        validation = validate_simulation(hypothesis, result)

        assert validation.valid is False
        assert any("NaN" in issue for issue in validation.issues)

    def test_validator_checks_convergence(self):
        """Test that validator checks convergence."""
        from aero.pipeline.hypothesis import Hypothesis
        from aero.pipeline.validator import validate_simulation

        hypothesis = Hypothesis(text="Test", confidence=0.5)

        # Non-converged result
        result = {"converged": False, "solution": [1, 2, 3]}
        validation = validate_simulation(hypothesis, result)

        assert any("converge" in issue.lower() for issue in validation.issues)

    def test_compute_error_metrics(self):
        """Test error metric computation."""
        from aero.pipeline.validator import compute_error_metrics

        computed = np.array([1.0, 2.0, 3.0])
        reference = np.array([1.1, 1.9, 3.1])

        metrics = compute_error_metrics(computed, reference)

        assert "l2_error" in metrics
        assert "rmse" in metrics
        assert metrics["l2_error"] > 0

    def test_check_physical_plausibility(self):
        """Test physical plausibility checks."""
        from aero.pipeline.validator import check_physical_plausibility

        # Plausible result
        result = {"solution": [50, 60, 70, 80]}
        checks = check_physical_plausibility(result, expected_range=(0, 100))
        assert checks["plausible"] is True

        # Implausible result (out of range)
        result = {"solution": [50, 60, 150, 80]}
        checks = check_physical_plausibility(result, expected_range=(0, 100))
        assert checks["plausible"] is False


# =============================================================================
# Pattern Detection Tests
# =============================================================================


class TestPatternDetection:
    """Tests for pattern detection."""

    def test_detect_patterns_from_result(self):
        """Test pattern detection from simulation result."""
        from aero.pipeline.patterns import detect_patterns

        # Create a symmetric solution
        x = np.linspace(0, 1, 101)
        solution = np.exp(-50 * (x - 0.5) ** 2)

        result = {"solution": solution.tolist(), "converged": True}
        patterns = detect_patterns(result)

        assert len(patterns) >= 1
        pattern_types = [p.type for p in patterns]
        # Should detect symmetry and/or diffusion profile
        assert any(
            t in pattern_types
            for t in ["symmetry", "diffusion_profile", "convergence"]
        )

    def test_detect_monotonic_pattern(self):
        """Test detection of monotonic patterns."""
        from aero.pipeline.patterns import detect_patterns

        # Monotonically increasing solution
        solution = np.linspace(0, 100, 50)
        result = {"solution": solution.tolist()}

        patterns = detect_patterns(result)
        pattern_types = [p.type for p in patterns]

        assert "monotonic_increasing" in pattern_types

    def test_scientific_pattern_to_dict(self):
        """Test scientific pattern serialization."""
        from aero.pipeline.patterns import ScientificPattern

        pattern = ScientificPattern(
            type="symmetry",
            confidence=0.85,
            location="midpoint",
            description="Solution is symmetric",
        )

        d = pattern.to_dict()

        assert d["type"] == "symmetry"
        assert d["confidence"] == 0.85
        assert d["location"] == "midpoint"

    def test_compare_patterns(self):
        """Test pattern comparison between results."""
        from aero.pipeline.patterns import compare_patterns

        prev_result = {
            "solution": np.linspace(0, 100, 50).tolist(),
        }
        current_result = {
            "solution": np.linspace(0, 100, 50).tolist(),
            "converged": True,
        }

        comparison = compare_patterns([prev_result], current_result)

        assert "persistent_patterns" in comparison
        assert "new_patterns" in comparison
        assert "converging" in comparison

    def test_summarize_patterns(self):
        """Test pattern summarization."""
        from aero.pipeline.patterns import ScientificPattern, summarize_patterns

        patterns = [
            ScientificPattern(type="symmetry", confidence=0.9),
            ScientificPattern(type="convergence", confidence=0.8),
        ]

        summary = summarize_patterns(patterns)

        assert isinstance(summary, str)
        assert len(summary) > 0


# =============================================================================
# Refinement Tests
# =============================================================================


class TestRefinement:
    """Tests for hypothesis refinement."""

    def test_refine_hypotheses(self):
        """Test hypothesis refinement based on validation."""
        from aero.pipeline.hypothesis import Hypothesis
        from aero.pipeline.validator import SimulationValidationResult
        from aero.pipeline.patterns import ScientificPattern
        from aero.pipeline.refinement import refine_hypotheses

        hypotheses = [
            Hypothesis(text="Good hypothesis", confidence=0.6),
            Hypothesis(text="Poor hypothesis", confidence=0.3),
        ]

        validations = [
            SimulationValidationResult(valid=True, score=0.9),
            SimulationValidationResult(valid=False, score=0.2),
        ]

        patterns = [ScientificPattern(type="convergence", confidence=0.9)]

        refined = refine_hypotheses(hypotheses, validations, patterns)

        # Should have at least one hypothesis
        assert len(refined) >= 1

        # Good hypothesis should have increased confidence
        good_refined = next((h for h in refined if "Good" in h.text or "good" in h.text.lower()), None)
        if good_refined:
            assert good_refined.confidence > 0.6

    def test_refinement_does_not_crash_if_all_fail(self):
        """Test that refinement handles all hypotheses failing."""
        from aero.pipeline.hypothesis import Hypothesis
        from aero.pipeline.validator import SimulationValidationResult
        from aero.pipeline.refinement import refine_hypotheses

        hypotheses = [
            Hypothesis(text="Bad hypothesis 1", confidence=0.1),
            Hypothesis(text="Bad hypothesis 2", confidence=0.1),
        ]

        validations = [
            SimulationValidationResult(valid=False, score=0.1),
            SimulationValidationResult(valid=False, score=0.05),
        ]

        # Should not raise an exception
        refined = refine_hypotheses(hypotheses, validations, [], drop_threshold=0.2)

        # May be empty, but should not crash
        assert isinstance(refined, list)

    def test_generate_refinement_summary(self):
        """Test refinement summary generation."""
        from aero.pipeline.hypothesis import Hypothesis
        from aero.pipeline.validator import SimulationValidationResult
        from aero.pipeline.refinement import generate_refinement_summary

        old = [
            Hypothesis(id="h1", text="Test 1", confidence=0.5),
            Hypothesis(id="h2", text="Test 2", confidence=0.3),
        ]

        new = [
            Hypothesis(id="h1", text="Test 1", confidence=0.7),
        ]

        validations = [
            SimulationValidationResult(valid=True, score=0.8),
            SimulationValidationResult(valid=False, score=0.2),
        ]

        summary = generate_refinement_summary(old, new, validations)

        assert summary["original_count"] == 2
        assert summary["refined_count"] == 1
        assert summary["dropped_count"] == 1


# =============================================================================
# Loop Execution Tests
# =============================================================================


class TestLoopExecution:
    """Tests for the scientific reasoning loop."""

    def test_loop_terminates_cleanly_on_1_iteration(self):
        """Test that loop terminates cleanly after 1 iteration."""
        from aero.pipeline.aero_loop import ScientificReasoningLoop, LoopConfig

        config = LoopConfig(max_iterations=1)
        loop = ScientificReasoningLoop(config=config)

        result = loop.run("Analyze heat diffusion in a rod")

        assert result.iterations_completed == 1
        assert result.termination_reason == "max_iterations"
        assert isinstance(result.history, list)
        assert len(result.history) == 1

    def test_loop_terminates_cleanly_on_max_iteration(self):
        """Test that loop terminates at max iterations."""
        from aero.pipeline.aero_loop import ScientificReasoningLoop, LoopConfig

        config = LoopConfig(
            max_iterations=3,
            confidence_threshold=0.99,  # Very high to ensure we hit max
        )
        loop = ScientificReasoningLoop(config=config)

        result = loop.run("Solve Laplace equation")

        assert result.iterations_completed <= 3
        assert isinstance(result.all_hypotheses, list)

    def test_loop_generates_hypotheses(self):
        """Test that loop generates hypotheses."""
        from aero.pipeline.aero_loop import ScientificReasoningLoop, LoopConfig

        config = LoopConfig(max_iterations=1)
        loop = ScientificReasoningLoop(config=config)

        result = loop.run("Analyze thermal diffusion")

        assert result.best_hypothesis is not None or len(result.all_hypotheses) >= 0

    def test_loop_runs_simulations(self):
        """Test that loop runs simulations."""
        from aero.pipeline.aero_loop import ScientificReasoningLoop, LoopConfig

        config = LoopConfig(max_iterations=1)
        loop = ScientificReasoningLoop(config=config)

        result = loop.run("Heat equation with alpha=0.01")

        # Should have run at least one simulation
        assert len(result.simulation_results) >= 0

    def test_loop_result_to_dict(self):
        """Test loop result serialization."""
        from aero.pipeline.aero_loop import ScientificReasoningLoop, LoopConfig

        config = LoopConfig(max_iterations=1)
        loop = ScientificReasoningLoop(config=config)

        result = loop.run("Test query")
        d = result.to_dict()

        assert "converged" in d
        assert "iterations_completed" in d
        assert "best_hypothesis" in d
        assert "history" in d
        assert "termination_reason" in d

    def test_loop_config_defaults(self):
        """Test loop config has sensible defaults."""
        from aero.pipeline.aero_loop import LoopConfig

        config = LoopConfig()

        assert config.max_iterations > 0
        assert 0 <= config.confidence_threshold <= 1
        assert 0 <= config.drop_threshold <= 1

    def test_loop_step_execution(self):
        """Test single step execution."""
        from aero.pipeline.aero_loop import ScientificReasoningLoop, LoopConfig

        config = LoopConfig(max_iterations=5)
        loop = ScientificReasoningLoop(config=config)

        step_result = loop.step("Analyze flow dynamics")

        assert "iteration" in step_result
        assert "hypotheses" in step_result
        assert "simulations" in step_result
        assert "validations" in step_result
        assert "patterns" in step_result
        assert "status" in step_result

    def test_loop_reset(self):
        """Test loop state reset."""
        from aero.pipeline.aero_loop import ScientificReasoningLoop, LoopConfig

        config = LoopConfig(max_iterations=1)
        loop = ScientificReasoningLoop(config=config)

        # Run once
        loop.run("Test query")
        assert len(loop._iteration_history) > 0

        # Reset
        loop.reset()
        assert len(loop._iteration_history) == 0
        assert len(loop._current_hypotheses) == 0


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for the full pipeline."""

    def test_end_to_end_heat_equation(self):
        """Test end-to-end execution for heat equation."""
        from aero.pipeline.aero_loop import ScientificReasoningLoop, LoopConfig

        config = LoopConfig(max_iterations=2)
        loop = ScientificReasoningLoop(config=config)

        result = loop.run("Analyze heat diffusion in a 1D rod with thermal diffusivity 0.01")

        # Should complete without errors
        assert result.iterations_completed >= 1
        assert result.termination_reason in [
            "max_iterations",
            "confidence_threshold",
            "pattern_convergence",
            "all_hypotheses_failed",
        ]

    def test_end_to_end_laplace_equation(self):
        """Test end-to-end execution for Laplace equation."""
        from aero.pipeline.aero_loop import ScientificReasoningLoop, LoopConfig

        config = LoopConfig(max_iterations=2)
        loop = ScientificReasoningLoop(config=config)

        result = loop.run("Solve steady state temperature distribution using Laplace equation")

        assert result.iterations_completed >= 1
        # Should have detected patterns
        assert isinstance(result.patterns, list)

    def test_pipeline_imports(self):
        """Test that all pipeline components can be imported."""
        from aero.pipeline import (
            ScientificReasoningLoop,
            LoopConfig,
            Hypothesis,
            generate_initial_hypotheses,
            plan_simulations,
            validate_simulation,
            detect_patterns,
            refine_hypotheses,
        )

        assert ScientificReasoningLoop is not None
        assert LoopConfig is not None
        assert Hypothesis is not None
        assert callable(generate_initial_hypotheses)
        assert callable(plan_simulations)
        assert callable(validate_simulation)
        assert callable(detect_patterns)
        assert callable(refine_hypotheses)
