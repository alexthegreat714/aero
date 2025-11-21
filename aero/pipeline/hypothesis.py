"""
Hypothesis generation module for Aero Agent.

Provides hypothesis creation and management for the scientific reasoning loop.
Uses template-based generation (no LLM) for deterministic, testable output.
"""

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Hypothesis:
    """
    Represents a scientific hypothesis in the Aero reasoning loop.

    Attributes:
        id: Unique identifier
        text: Human-readable hypothesis statement
        confidence: Confidence score (0.0 to 1.0)
        metadata: Additional metadata (source, tags, etc.)
        created_at: Creation timestamp
        parent_id: ID of parent hypothesis (for refinements)
    """

    text: str
    confidence: float = 0.5
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    parent_id: Optional[str] = None

    def __post_init__(self):
        """Validate hypothesis after initialization."""
        self.confidence = max(0.0, min(1.0, self.confidence))

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "text": self.text,
            "confidence": self.confidence,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "parent_id": self.parent_id,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Hypothesis":
        """Create hypothesis from dictionary."""
        return cls(
            id=data.get("id", str(uuid.uuid4())[:8]),
            text=data["text"],
            confidence=data.get("confidence", 0.5),
            metadata=data.get("metadata", {}),
            parent_id=data.get("parent_id"),
        )


# =============================================================================
# Hypothesis Templates
# =============================================================================

# Templates for different scientific domains
HYPOTHESIS_TEMPLATES = {
    # Aerodynamics templates
    "aerodynamics": [
        "The airflow over the surface exhibits {flow_type} behavior at Reynolds number {Re}.",
        "Drag coefficient decreases when {parameter} is modified to {value}.",
        "Lift-to-drag ratio is optimized at angle of attack {aoa} degrees.",
        "Flow separation occurs at {location} due to {cause}.",
        "Turbulence intensity affects {outcome} by approximately {magnitude}.",
    ],
    # Heat transfer templates
    "heat_transfer": [
        "Heat diffusion follows {model} with thermal diffusivity alpha = {alpha}.",
        "Steady-state temperature distribution is {pattern} across the domain.",
        "Boundary temperature of {temp}K produces {effect} in the interior.",
        "Convective heat transfer coefficient is approximately {h} W/(m^2*K).",
        "Thermal equilibrium is reached after {time} seconds.",
    ],
    # Fluid dynamics templates
    "fluid_dynamics": [
        "The flow field satisfies the incompressibility condition div(u) = 0.",
        "Pressure gradient drives flow in the {direction} direction.",
        "Vorticity concentrates at {location} with magnitude {omega}.",
        "Reynolds number of {Re} indicates {flow_regime} flow.",
        "Velocity profile follows {profile_type} distribution.",
    ],
    # General PDE templates
    "pde": [
        "The solution converges to steady state with residual < {tol}.",
        "Spatial discretization of {dx} provides {order}-order accuracy.",
        "Time step {dt} satisfies the CFL stability condition.",
        "Boundary conditions of type {bc_type} are appropriate for this problem.",
        "The solution exhibits {symmetry} symmetry about {axis}.",
    ],
    # Generic scientific templates
    "generic": [
        "Parameter {param} has a significant effect on {outcome}.",
        "The relationship between {var1} and {var2} is {relationship}.",
        "Increasing {factor} by {amount} will {effect} the result.",
        "The observed pattern suggests {mechanism} is the dominant factor.",
        "Results are consistent with {theory} theory.",
    ],
}

# Keywords that map queries to domains
DOMAIN_KEYWORDS = {
    "aerodynamics": ["airflow", "drag", "lift", "wing", "airfoil", "mach", "reynolds", "flight"],
    "heat_transfer": ["heat", "temperature", "thermal", "conduction", "convection", "diffusion"],
    "fluid_dynamics": ["flow", "fluid", "pressure", "velocity", "viscosity", "navier", "stokes"],
    "pde": ["equation", "solver", "convergence", "boundary", "discretization", "numerical"],
}


def detect_domain(query: str) -> str:
    """
    Detect the scientific domain from query text.

    Args:
        query: User query or problem description

    Returns:
        Domain name (aerodynamics, heat_transfer, fluid_dynamics, pde, or generic)
    """
    query_lower = query.lower()

    scores = {}
    for domain, keywords in DOMAIN_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in query_lower)
        scores[domain] = score

    if max(scores.values()) > 0:
        return max(scores, key=scores.get)

    return "generic"


def extract_parameters(query: str) -> Dict[str, str]:
    """
    Extract numerical parameters and keywords from query.

    Args:
        query: User query text

    Returns:
        Dictionary of extracted parameters
    """
    import re

    params = {}

    # Extract Reynolds number
    re_match = re.search(r"reynolds\s*(?:number)?\s*(?:=|:)?\s*(\d+(?:\.\d+)?(?:e[+-]?\d+)?)", query, re.I)
    if re_match:
        params["Re"] = re_match.group(1)

    # Extract temperature
    temp_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:K|kelvin|degrees?)", query, re.I)
    if temp_match:
        params["temp"] = temp_match.group(1)

    # Extract alpha/diffusivity
    alpha_match = re.search(r"alpha\s*=?\s*(\d+(?:\.\d+)?(?:e[+-]?\d+)?)", query, re.I)
    if alpha_match:
        params["alpha"] = alpha_match.group(1)

    # Extract grid size
    grid_match = re.search(r"(\d+)\s*x\s*(\d+)(?:\s*grid)?", query, re.I)
    if grid_match:
        params["nx"] = grid_match.group(1)
        params["ny"] = grid_match.group(2)

    # Extract tolerance
    tol_match = re.search(r"tolerance\s*(?:=|:)?\s*(\d+(?:\.\d+)?(?:e[+-]?\d+)?)", query, re.I)
    if tol_match:
        params["tol"] = tol_match.group(1)

    return params


def generate_initial_hypotheses(
    query: str,
    rag=None,
    num_hypotheses: int = 3,
) -> List[Hypothesis]:
    """
    Generate initial hypotheses from a user query.

    Uses template-based generation with RAG context augmentation.

    Args:
        query: User query or problem description
        rag: Optional RAG store for context retrieval
        num_hypotheses: Number of hypotheses to generate (1-3)

    Returns:
        List of Hypothesis objects
    """
    logger.info(f"Generating hypotheses for query: {query[:100]}...")

    hypotheses = []
    num_hypotheses = max(1, min(3, num_hypotheses))

    # Detect domain
    domain = detect_domain(query)
    logger.debug(f"Detected domain: {domain}")

    # Extract parameters
    params = extract_parameters(query)
    logger.debug(f"Extracted parameters: {params}")

    # Get templates for domain
    templates = HYPOTHESIS_TEMPLATES.get(domain, HYPOTHESIS_TEMPLATES["generic"])

    # RAG context augmentation
    rag_context = []
    if rag is not None:
        try:
            results = rag.search(query, top_k=3)
            rag_context = [r.get("content", "")[:200] for r in results]
            logger.debug(f"Retrieved {len(rag_context)} RAG results")
        except Exception as e:
            logger.warning(f"RAG search failed: {e}")

    # Generate hypotheses
    for i in range(num_hypotheses):
        template_idx = i % len(templates)
        template = templates[template_idx]

        # Fill template with parameters or placeholders
        filled = _fill_template(template, params, domain, i)

        # Adjust confidence based on available information
        confidence = _calculate_initial_confidence(params, rag_context, i)

        # Create metadata
        metadata = {
            "source": "template",
            "domain": domain,
            "template_idx": template_idx,
            "query_params": params,
            "rag_augmented": len(rag_context) > 0,
        }

        hypothesis = Hypothesis(
            text=filled,
            confidence=confidence,
            metadata=metadata,
        )
        hypotheses.append(hypothesis)

    logger.info(f"Generated {len(hypotheses)} initial hypotheses")
    return hypotheses


def generate_secondary_hypotheses(
    previous_results: List[dict],
    rag=None,
    base_hypotheses: Optional[List[Hypothesis]] = None,
) -> List[Hypothesis]:
    """
    Generate secondary hypotheses based on previous simulation/experiment results.

    Args:
        previous_results: Results from previous iteration
        rag: Optional RAG store for context
        base_hypotheses: Previous hypotheses to refine

    Returns:
        List of new/refined Hypothesis objects
    """
    logger.info("Generating secondary hypotheses from results")

    hypotheses = []

    if not previous_results:
        logger.warning("No previous results provided")
        return hypotheses

    # Analyze results to determine next hypotheses
    for i, result in enumerate(previous_results[:3]):
        # Extract key information from result
        converged = result.get("converged", result.get("valid", False))
        error = result.get("error", result.get("final_residual", None))
        patterns = result.get("patterns", [])

        # Generate hypothesis based on result
        if converged:
            text = _generate_convergent_hypothesis(result, i)
            confidence = 0.6 + 0.1 * (i + 1) / 3
        else:
            text = _generate_divergent_hypothesis(result, i)
            confidence = 0.3 + 0.1 * (i + 1) / 3

        # Link to parent if available
        parent_id = None
        if base_hypotheses and i < len(base_hypotheses):
            parent_id = base_hypotheses[i].id

        metadata = {
            "source": "secondary",
            "based_on_result": i,
            "result_converged": converged,
            "detected_patterns": [p.get("type") if isinstance(p, dict) else str(p) for p in patterns[:3]],
        }

        hypothesis = Hypothesis(
            text=text,
            confidence=confidence,
            metadata=metadata,
            parent_id=parent_id,
        )
        hypotheses.append(hypothesis)

    return hypotheses


# =============================================================================
# Helper Functions
# =============================================================================


def _fill_template(template: str, params: Dict[str, str], domain: str, idx: int) -> str:
    """Fill a template with parameters or domain-specific defaults."""
    # Default values based on domain and index
    defaults = {
        "flow_type": ["laminar", "transitional", "turbulent"][idx % 3],
        "Re": params.get("Re", ["1000", "10000", "100000"][idx % 3]),
        "parameter": ["surface roughness", "geometry", "inlet velocity"][idx % 3],
        "value": ["optimized", "reduced", "increased"][idx % 3],
        "aoa": ["5", "10", "15"][idx % 3],
        "location": ["trailing edge", "leading edge", "mid-chord"][idx % 3],
        "cause": ["adverse pressure gradient", "geometric discontinuity", "high angle of attack"][idx % 3],
        "outcome": ["heat transfer", "pressure distribution", "separation point"][idx % 3],
        "magnitude": ["10%", "20%", "significant"][idx % 3],
        "model": ["Fourier's law", "Newton's law of cooling", "Stefan-Boltzmann"][idx % 3],
        "alpha": params.get("alpha", ["0.01", "0.001", "0.1"][idx % 3]),
        "pattern": ["linear", "exponential", "parabolic"][idx % 3],
        "temp": params.get("temp", ["300", "400", "500"][idx % 3]),
        "effect": ["uniform heating", "thermal gradients", "hot spots"][idx % 3],
        "h": ["10", "50", "100"][idx % 3],
        "time": ["10", "100", "1000"][idx % 3],
        "direction": ["x", "y", "streamwise"][idx % 3],
        "omega": ["low", "moderate", "high"][idx % 3],
        "flow_regime": ["laminar", "transitional", "turbulent"][idx % 3],
        "profile_type": ["parabolic", "logarithmic", "uniform"][idx % 3],
        "tol": params.get("tol", ["1e-6", "1e-4", "1e-8"][idx % 3]),
        "dx": ["0.01", "0.001", "0.1"][idx % 3],
        "order": ["second", "fourth", "first"][idx % 3],
        "dt": ["0.0001", "0.001", "0.01"][idx % 3],
        "bc_type": ["Dirichlet", "Neumann", "mixed"][idx % 3],
        "symmetry": ["axial", "planar", "point"][idx % 3],
        "axis": ["x=0", "y=0", "center"][idx % 3],
        "param": ["viscosity", "diffusivity", "velocity"][idx % 3],
        "var1": ["temperature", "pressure", "velocity"][idx % 3],
        "var2": ["time", "position", "Reynolds number"][idx % 3],
        "relationship": ["linear", "nonlinear", "exponential"][idx % 3],
        "factor": ["grid resolution", "time step", "boundary value"][idx % 3],
        "amount": ["2x", "50%", "10%"][idx % 3],
        "mechanism": ["diffusion", "advection", "convection"][idx % 3],
        "theory": ["classical", "modern computational", "analytical"][idx % 3],
    }

    # Fill template
    filled = template
    for key, value in defaults.items():
        filled = filled.replace("{" + key + "}", str(value))

    return filled


def _calculate_initial_confidence(
    params: Dict[str, str],
    rag_context: List[str],
    idx: int,
) -> float:
    """Calculate initial confidence based on available information."""
    base_confidence = 0.5

    # Increase confidence if we have extracted parameters
    if params:
        base_confidence += 0.1 * min(len(params), 3) / 3

    # Increase confidence if RAG context is available
    if rag_context:
        base_confidence += 0.1

    # Decrease confidence for later hypotheses (less certain)
    base_confidence -= 0.05 * idx

    return max(0.2, min(0.8, base_confidence))


def _generate_convergent_hypothesis(result: dict, idx: int) -> str:
    """Generate hypothesis for convergent result."""
    templates = [
        "The numerical solution has converged, indicating the simulation parameters are appropriate.",
        "Convergence suggests the physical model accurately captures the dominant phenomena.",
        "The steady-state solution reveals the expected physical behavior in the system.",
    ]
    return templates[idx % len(templates)]


def _generate_divergent_hypothesis(result: dict, idx: int) -> str:
    """Generate hypothesis for non-convergent result."""
    templates = [
        "The simulation did not converge, suggesting the time step or grid resolution needs adjustment.",
        "Non-convergence may indicate missing physics or inappropriate boundary conditions.",
        "The instability suggests revisiting the numerical scheme or problem formulation.",
    ]
    return templates[idx % len(templates)]


def rank_hypotheses(hypotheses: List[Hypothesis]) -> List[Hypothesis]:
    """
    Rank hypotheses by confidence score.

    Args:
        hypotheses: List of hypotheses to rank

    Returns:
        Sorted list (highest confidence first)
    """
    return sorted(hypotheses, key=lambda h: h.confidence, reverse=True)


def filter_hypotheses(
    hypotheses: List[Hypothesis],
    min_confidence: float = 0.0,
    max_results: Optional[int] = None,
) -> List[Hypothesis]:
    """
    Filter hypotheses by confidence threshold.

    Args:
        hypotheses: List of hypotheses
        min_confidence: Minimum confidence threshold
        max_results: Maximum number of results

    Returns:
        Filtered list
    """
    filtered = [h for h in hypotheses if h.confidence >= min_confidence]

    if max_results:
        filtered = filtered[:max_results]

    return filtered
