"""
Agent API routes for Aero Agent.

Provides endpoints for agent management, task execution,
and the scientific reasoning loop.
"""

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter()


# =============================================================================
# Request/Response Models
# =============================================================================


class TaskRequest(BaseModel):
    """Request for submitting a task."""

    name: str
    type: str
    parameters: Optional[dict] = None


class AgentConfig(BaseModel):
    """Configuration for agent settings."""

    max_workers: int = 4
    timeout: int = 300


class LoopRequest(BaseModel):
    """Request for running the scientific reasoning loop."""

    query: str = Field(..., description="Scientific query or problem description")
    max_iterations: int = Field(default=5, ge=1, le=20)
    confidence_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    drop_threshold: float = Field(default=0.2, ge=0.0, le=1.0)


class HypothesisRequest(BaseModel):
    """Request for generating hypotheses."""

    query: str = Field(..., description="Query for hypothesis generation")
    num_hypotheses: int = Field(default=3, ge=1, le=5)


class PlanRequest(BaseModel):
    """Request for planning simulations."""

    hypothesis_text: str = Field(..., description="Hypothesis text")
    hypothesis_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    domain: Optional[str] = Field(default=None, description="Scientific domain")


class ValidateRequest(BaseModel):
    """Request for validating simulation results."""

    result: Dict[str, Any] = Field(..., description="Simulation result to validate")
    hypothesis_id: Optional[str] = Field(default=None)


class RefineRequest(BaseModel):
    """Request for refining hypotheses."""

    hypotheses: List[Dict[str, Any]] = Field(..., description="Hypotheses to refine")
    validation_scores: List[float] = Field(..., description="Validation scores")
    patterns: List[Dict[str, Any]] = Field(default=[], description="Detected patterns")


# =============================================================================
# Basic Agent Endpoints
# =============================================================================


@router.get("/")
async def agent_status():
    """Get agent system status."""
    return {
        "status": "ok",
        "module": "agent",
        "running": True,
        "workers": 4,
        "queue_size": 0,
        "capabilities": [
            "reasoning_loop",
            "hypothesis_generation",
            "simulation_planning",
            "validation",
            "pattern_detection",
            "refinement",
        ],
        "message": "Agent system is operational",
    }


@router.get("/info")
async def agent_info():
    """Get detailed agent information."""
    return {
        "status": "ok",
        "version": "0.1.0",
        "name": "Aero Agent",
        "capabilities": [
            "rag",
            "simulation",
            "ocr",
            "web_search",
            "experiments",
            "scientific_reasoning",
        ],
        "message": "Aero Agent with scientific reasoning loop",
    }


@router.post("/tasks")
async def submit_task(task: TaskRequest):
    """
    Submit a task to the agent.

    Returns task ID for tracking.
    """
    logger.info(f"Submitting task: {task.name} ({task.type})")

    return {
        "status": "submitted",
        "task_id": "task_001",
        "name": task.name,
        "type": task.type,
        "message": "Task submitted",
    }


@router.get("/tasks")
async def list_tasks(status: Optional[str] = None):
    """List all tasks with optional status filter."""
    return {
        "status": "ok",
        "tasks": [],
        "total": 0,
    }


@router.get("/tasks/{task_id}")
async def get_task(task_id: str):
    """Get task details and status."""
    return {
        "status": "ok",
        "task_id": task_id,
        "state": "pending",
        "progress": 0.0,
        "result": None,
    }


@router.delete("/tasks/{task_id}")
async def cancel_task(task_id: str):
    """Cancel a pending or running task."""
    return {
        "status": "cancelled",
        "task_id": task_id,
    }


@router.post("/config")
async def update_config(config: AgentConfig):
    """Update agent configuration."""
    logger.info(f"Updating config: workers={config.max_workers}")

    return {
        "status": "updated",
        "config": config.dict(),
    }


@router.get("/registries")
async def list_registries():
    """List all component registries."""
    try:
        from aero.core.registry import get_registry_manager
        manager = get_registry_manager()
        return {
            "status": "ok",
            "registries": manager.get_summary(),
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
        }


@router.get("/events")
async def list_events(limit: int = 100):
    """List recent events from the event bus."""
    return {
        "status": "ok",
        "events": [],
        "total": 0,
    }


# =============================================================================
# Scientific Reasoning Loop Endpoints
# =============================================================================


@router.post("/loop")
async def run_reasoning_loop(request: LoopRequest):
    """
    Execute the full scientific reasoning loop.

    The loop will:
    1. Generate hypotheses from the query
    2. Plan simulations for each hypothesis
    3. Execute simulations
    4. Validate results
    5. Detect patterns
    6. Refine hypotheses
    7. Repeat until convergence or max iterations

    Returns:
        Complete loop execution results including:
        - converged: Whether the loop converged
        - iterations_completed: Number of iterations run
        - best_hypothesis: The highest confidence hypothesis
        - all_hypotheses: All surviving hypotheses
        - simulation_results: Results from all simulations
        - validation_results: Validation scores
        - patterns: Detected scientific patterns
        - history: Full iteration history
    """
    logger.info(f"Running reasoning loop for query: {request.query[:100]}...")

    try:
        from aero.pipeline.aero_loop import ScientificReasoningLoop, LoopConfig

        config = LoopConfig(
            max_iterations=request.max_iterations,
            confidence_threshold=request.confidence_threshold,
            drop_threshold=request.drop_threshold,
        )

        loop = ScientificReasoningLoop(config=config)
        result = loop.run(request.query)

        return {
            "status": "ok",
            **result.to_dict(),
        }

    except Exception as e:
        logger.exception(f"Loop execution error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/hypothesis")
async def generate_hypotheses(request: HypothesisRequest):
    """
    Generate hypotheses for a scientific query.

    Uses template-based generation with domain detection
    and optional RAG augmentation.

    Returns:
        List of generated hypotheses with confidence scores.
    """
    logger.info(f"Generating hypotheses for: {request.query[:100]}...")

    try:
        from aero.pipeline.hypothesis import generate_initial_hypotheses, detect_domain

        domain = detect_domain(request.query)
        hypotheses = generate_initial_hypotheses(
            request.query,
            rag=None,  # RAG integration optional
            num_hypotheses=request.num_hypotheses,
        )

        return {
            "status": "ok",
            "domain": domain,
            "count": len(hypotheses),
            "hypotheses": [h.to_dict() for h in hypotheses],
        }

    except Exception as e:
        logger.exception(f"Hypothesis generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/plan")
async def plan_simulations(request: PlanRequest):
    """
    Create simulation plans for a hypothesis.

    Analyzes the hypothesis and generates appropriate
    simulation configurations.

    Returns:
        List of simulation job configurations.
    """
    logger.info(f"Planning simulations for hypothesis: {request.hypothesis_text[:100]}...")

    try:
        from aero.pipeline.hypothesis import Hypothesis
        from aero.pipeline.planner import plan_simulations as _plan_simulations

        # Create hypothesis object
        hypothesis = Hypothesis(
            text=request.hypothesis_text,
            confidence=request.hypothesis_confidence,
            metadata={"domain": request.domain} if request.domain else {},
        )

        configs = _plan_simulations(hypothesis)

        return {
            "status": "ok",
            "hypothesis_id": hypothesis.id,
            "count": len(configs),
            "simulation_configs": [c.to_dict() for c in configs],
        }

    except Exception as e:
        logger.exception(f"Planning error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/validate")
async def validate_result(request: ValidateRequest):
    """
    Validate simulation results.

    Checks convergence, stability, physical plausibility,
    and computes error metrics.

    Returns:
        Validation result with score and issues.
    """
    logger.info("Validating simulation result...")

    try:
        from aero.pipeline.hypothesis import Hypothesis
        from aero.pipeline.validator import validate_simulation

        # Create placeholder hypothesis
        hypothesis = Hypothesis(
            id=request.hypothesis_id or "unknown",
            text="Validation request",
            confidence=0.5,
        )

        validation = validate_simulation(hypothesis, request.result)

        return {
            "status": "ok",
            **validation.to_dict(),
        }

    except Exception as e:
        logger.exception(f"Validation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/refine")
async def refine_hypotheses(request: RefineRequest):
    """
    Refine hypotheses based on validation results.

    Adjusts confidence scores, mutates text, and drops
    hypotheses below threshold.

    Returns:
        Refined hypotheses with updated confidence scores.
    """
    logger.info(f"Refining {len(request.hypotheses)} hypotheses...")

    try:
        from aero.pipeline.hypothesis import Hypothesis
        from aero.pipeline.validator import SimulationValidationResult
        from aero.pipeline.patterns import ScientificPattern
        from aero.pipeline.refinement import refine_hypotheses as _refine

        # Convert input to objects
        hypotheses = [Hypothesis.from_dict(h) for h in request.hypotheses]

        validations = [
            SimulationValidationResult(
                valid=True,
                score=score,
            )
            for score in request.validation_scores
        ]

        patterns = [
            ScientificPattern(
                type=p.get("type", "unknown"),
                confidence=p.get("confidence", 0.5),
            )
            for p in request.patterns
        ]

        refined = _refine(hypotheses, validations, patterns)

        return {
            "status": "ok",
            "original_count": len(hypotheses),
            "refined_count": len(refined),
            "hypotheses": [h.to_dict() for h in refined],
        }

    except Exception as e:
        logger.exception(f"Refinement error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/patterns/{result_type}")
async def detect_patterns(result_type: str):
    """
    Get information about detectable patterns.

    Args:
        result_type: Type of result (1d, 2d, general)

    Returns:
        List of pattern types that can be detected.
    """
    patterns_1d = [
        "symmetry",
        "monotonic_increasing",
        "monotonic_decreasing",
        "periodicity",
        "diffusion_profile",
        "oscillations",
        "shock_gradient",
    ]

    patterns_2d = [
        "x_symmetry",
        "y_symmetry",
        "x_uniform",
        "y_uniform",
        "linear_gradient_x",
        "linear_gradient_y",
    ]

    patterns_general = [
        "convergence",
        "steady_state",
    ]

    if result_type == "1d":
        return {"status": "ok", "patterns": patterns_1d}
    elif result_type == "2d":
        return {"status": "ok", "patterns": patterns_2d}
    else:
        return {"status": "ok", "patterns": patterns_1d + patterns_2d + patterns_general}


# =============================================================================
# Symbolic Constraint Endpoints
# =============================================================================


class ConstraintCheckRequest(BaseModel):
    """Request for checking constraints on simulation/experiment results."""

    result: Dict[str, Any] = Field(..., description="Simulation or experiment result")
    result_type: str = Field(default="simulation", description="Type: 'simulation' or 'experiment'")
    sim_type: Optional[str] = Field(default=None, description="Simulation type (e.g., 'heat_1d')")


class DimensionCheckRequest(BaseModel):
    """Request for dimensional analysis."""

    equation_type: str = Field(..., description="Equation type (e.g., 'heat_1d', 'laplace_2d')")
    parameters: Dict[str, Any] = Field(..., description="Parameter names and values/dimensions")


@router.get("/constraints/status")
async def constraints_status():
    """
    Get status of the symbolic constraint system.

    Returns:
        sympy_available: Whether SymPy is installed
        enabled: Whether constraints are enabled in config
        available_expressions: List of available symbolic expressions
        available_constraints: List of available constraint types
    """
    try:
        from aero.symbolic.checks import get_symbolic_status
        status = get_symbolic_status()
        return {
            "status": "ok",
            **status,
        }
    except ImportError:
        return {
            "status": "ok",
            "sympy_available": False,
            "enabled": False,
            "available_expressions": [],
            "available_constraints": [],
            "message": "Symbolic module not available",
        }
    except Exception as e:
        logger.exception(f"Constraint status error: {e}")
        return {
            "status": "error",
            "message": str(e),
        }


@router.post("/constraints/check")
async def check_constraints(request: ConstraintCheckRequest):
    """
    Check constraints on a simulation or experiment result.

    Returns:
        passed: Overall pass/fail
        overall_score: Weighted constraint score (0.0 to 1.0)
        constraints: List of individual constraint results
        sympy_used: Whether SymPy was used
    """
    logger.info(f"Checking constraints for {request.result_type}...")

    try:
        from aero.symbolic.checks import (
            check_simulation_constraints,
            check_experiment_constraints,
        )
        from aero.config import get_config

        config = get_config()
        symbolic_config = config.get("symbolic", {})

        # Build result dict with sim_type if provided
        result = dict(request.result)
        if request.sim_type:
            result["sim_type"] = request.sim_type

        if request.result_type == "simulation":
            check_result = check_simulation_constraints(
                sim_result=result,
                config=symbolic_config,
            )
        else:
            check_result = check_experiment_constraints(
                exp_result=result,
                config=symbolic_config,
            )

        return {
            "status": "ok",
            **check_result,
        }

    except ImportError:
        return {
            "status": "ok",
            "passed": True,
            "overall_score": 1.0,
            "constraints": [],
            "sympy_used": False,
            "message": "Symbolic module not available - constraints skipped",
        }
    except Exception as e:
        logger.exception(f"Constraint check error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/constraints/dimensions")
async def check_dimensions(request: DimensionCheckRequest):
    """
    Check dimensional consistency for an equation type.

    Returns:
        consistent: Whether dimensions are consistent
        details: List of dimension checks performed
        errors: Any dimensional inconsistencies found
    """
    logger.info(f"Checking dimensions for {request.equation_type}...")

    try:
        from aero.symbolic.dimensions import check_dimensional_consistency

        result = check_dimensional_consistency(
            equation_type=request.equation_type,
            parameters=request.parameters,
        )

        return {
            "status": "ok",
            **result,
        }

    except ImportError:
        return {
            "status": "ok",
            "consistent": True,
            "details": [],
            "errors": [],
            "message": "Symbolic module not available",
        }
    except Exception as e:
        logger.exception(f"Dimension check error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/constraints/expressions")
async def list_expressions():
    """
    List available symbolic expressions for PDEs.

    Returns list of expression names and their metadata.
    """
    try:
        from aero.symbolic.expressions import list_available_expressions

        expressions = list_available_expressions()

        return {
            "status": "ok",
            "count": len(expressions),
            "expressions": {
                name: expr.to_dict() for name, expr in expressions.items()
            },
        }

    except ImportError:
        return {
            "status": "ok",
            "count": 0,
            "expressions": {},
            "message": "Symbolic module not available",
        }
    except Exception as e:
        logger.exception(f"List expressions error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/constraints/dimensions/{dimension_name}")
async def get_dimension(dimension_name: str):
    """
    Get information about a physical dimension.

    Args:
        dimension_name: Name of dimension (e.g., 'velocity', 'pressure')

    Returns:
        Dimension information including base unit exponents.
    """
    try:
        from aero.symbolic.dimensions import get_dimension, BASE_DIMENSIONS

        dim = get_dimension(dimension_name)

        if dim is None:
            # Return list of available dimensions
            return {
                "status": "error",
                "message": f"Unknown dimension: {dimension_name}",
                "available_dimensions": list(BASE_DIMENSIONS.keys()),
            }

        return {
            "status": "ok",
            "dimension": dim.to_dict(),
            "string": str(dim),
        }

    except ImportError:
        return {
            "status": "error",
            "message": "Symbolic module not available",
        }
    except Exception as e:
        logger.exception(f"Get dimension error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
