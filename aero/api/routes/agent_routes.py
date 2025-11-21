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
