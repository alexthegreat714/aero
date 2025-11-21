"""
Planning module for Aero Agent.

Converts hypotheses into concrete simulation and experiment configurations.
Outputs structured JSON specifications for execution.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum

from aero.pipeline.hypothesis import Hypothesis

logger = logging.getLogger(__name__)


class SimulationType(str, Enum):
    """Types of simulations available."""

    HEAT_1D = "heat_1d"
    HEAT_2D = "heat_2d"
    LAPLACE_2D = "laplace_2d"
    NAVIER_STOKES = "navier_stokes"
    PINN = "pinn"


class ExperimentType(str, Enum):
    """Types of experiments available."""

    WEBCAM_CAPTURE = "webcam_capture"
    OCR_EXTRACTION = "ocr_extraction"
    FILE_ANALYSIS = "file_analysis"
    SENSOR_READ = "sensor_read"


@dataclass
class SimulationJobConfig:
    """
    Configuration for a simulation job.

    Contains all parameters needed to execute a simulation.
    """

    sim_type: SimulationType
    hypothesis_id: str
    grid_size: int = 64
    grid_size_y: Optional[int] = None  # For 2D (defaults to grid_size)
    dx: float = 0.01
    dy: Optional[float] = None  # For 2D (defaults to dx)
    dt: float = 0.0001
    max_steps: int = 1000
    max_iterations: int = 5000
    tolerance: float = 1e-6
    boundary_conditions: Dict[str, float] = field(default_factory=dict)
    initial_condition: str = "zero"  # zero, gaussian, sine, step
    solver_params: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "sim_type": self.sim_type.value if isinstance(self.sim_type, Enum) else self.sim_type,
            "hypothesis_id": self.hypothesis_id,
            "grid_size": self.grid_size,
            "grid_size_y": self.grid_size_y or self.grid_size,
            "dx": self.dx,
            "dy": self.dy or self.dx,
            "dt": self.dt,
            "max_steps": self.max_steps,
            "max_iterations": self.max_iterations,
            "tolerance": self.tolerance,
            "boundary_conditions": self.boundary_conditions,
            "initial_condition": self.initial_condition,
            "solver_params": self.solver_params,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SimulationJobConfig":
        """Create from dictionary."""
        sim_type = data.get("sim_type", "heat_1d")
        if isinstance(sim_type, str):
            try:
                sim_type = SimulationType(sim_type)
            except ValueError:
                sim_type = SimulationType.HEAT_1D

        return cls(
            sim_type=sim_type,
            hypothesis_id=data.get("hypothesis_id", ""),
            grid_size=data.get("grid_size", 64),
            grid_size_y=data.get("grid_size_y"),
            dx=data.get("dx", 0.01),
            dy=data.get("dy"),
            dt=data.get("dt", 0.0001),
            max_steps=data.get("max_steps", 1000),
            max_iterations=data.get("max_iterations", 5000),
            tolerance=data.get("tolerance", 1e-6),
            boundary_conditions=data.get("boundary_conditions", {}),
            initial_condition=data.get("initial_condition", "zero"),
            solver_params=data.get("solver_params", {}),
            metadata=data.get("metadata", {}),
        )


@dataclass
class ExperimentConfig:
    """
    Configuration for an experiment.

    Contains parameters for data collection and analysis.
    """

    exp_type: ExperimentType
    hypothesis_id: str
    source: str = ""  # File path, camera index, etc.
    parameters: Dict[str, Any] = field(default_factory=dict)
    ocr_backend: str = "auto"  # For OCR experiments
    preprocessing: bool = True
    output_format: str = "json"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "exp_type": self.exp_type.value if isinstance(self.exp_type, Enum) else self.exp_type,
            "hypothesis_id": self.hypothesis_id,
            "source": self.source,
            "parameters": self.parameters,
            "ocr_backend": self.ocr_backend,
            "preprocessing": self.preprocessing,
            "output_format": self.output_format,
            "metadata": self.metadata,
        }


# =============================================================================
# Simulation Planning
# =============================================================================


def plan_simulations(hypothesis: Hypothesis) -> List[SimulationJobConfig]:
    """
    Create simulation job configurations from a hypothesis.

    Analyzes the hypothesis text and metadata to determine appropriate
    simulation parameters.

    Args:
        hypothesis: Hypothesis to plan simulations for

    Returns:
        List of SimulationJobConfig objects
    """
    logger.info(f"Planning simulations for hypothesis: {hypothesis.id}")

    configs = []
    text_lower = hypothesis.text.lower()
    domain = hypothesis.metadata.get("domain", "generic")

    # Determine simulation type based on hypothesis content
    sim_type = _detect_simulation_type(text_lower, domain)

    # Create base configuration
    base_config = _create_base_config(hypothesis, sim_type)

    # Add domain-specific configurations
    if "heat" in text_lower or "thermal" in text_lower or "diffusion" in text_lower:
        configs.append(_create_heat_config(hypothesis, base_config))

    if "laplace" in text_lower or "steady" in text_lower or "equilibrium" in text_lower:
        configs.append(_create_laplace_config(hypothesis, base_config))

    if "flow" in text_lower or "velocity" in text_lower or "navier" in text_lower:
        configs.append(_create_ns_config(hypothesis, base_config))

    if "neural" in text_lower or "pinn" in text_lower or "learn" in text_lower:
        configs.append(_create_pinn_config(hypothesis, base_config))

    # If no specific type detected, use domain default
    if not configs:
        configs.append(_create_default_config(hypothesis, sim_type, base_config))

    # Limit to reasonable number
    configs = configs[:3]

    logger.info(f"Planned {len(configs)} simulations for hypothesis {hypothesis.id}")
    return configs


def _detect_simulation_type(text: str, domain: str) -> SimulationType:
    """Detect appropriate simulation type from text and domain."""
    if "heat" in text or "thermal" in text:
        if "2d" in text or "two" in text:
            return SimulationType.HEAT_2D
        return SimulationType.HEAT_1D

    if "laplace" in text or "steady state" in text:
        return SimulationType.LAPLACE_2D

    if "navier" in text or "stokes" in text or "flow" in text:
        return SimulationType.NAVIER_STOKES

    if "neural" in text or "pinn" in text:
        return SimulationType.PINN

    # Domain-based fallback
    domain_defaults = {
        "heat_transfer": SimulationType.HEAT_1D,
        "fluid_dynamics": SimulationType.NAVIER_STOKES,
        "aerodynamics": SimulationType.NAVIER_STOKES,
        "pde": SimulationType.LAPLACE_2D,
    }

    return domain_defaults.get(domain, SimulationType.HEAT_1D)


def _create_base_config(hypothesis: Hypothesis, sim_type: SimulationType) -> dict:
    """Create base configuration from hypothesis."""
    params = hypothesis.metadata.get("query_params", {})

    return {
        "hypothesis_id": hypothesis.id,
        "sim_type": sim_type,
        "grid_size": int(params.get("nx", 64)),
        "grid_size_y": int(params.get("ny", 64)) if "ny" in params else None,
        "tolerance": float(params.get("tol", 1e-6)),
        "metadata": {
            "hypothesis_confidence": hypothesis.confidence,
            "hypothesis_domain": hypothesis.metadata.get("domain"),
        },
    }


def _create_heat_config(hypothesis: Hypothesis, base: dict) -> SimulationJobConfig:
    """Create heat equation simulation config."""
    params = hypothesis.metadata.get("query_params", {})
    text_lower = hypothesis.text.lower()

    # Determine if 1D or 2D
    is_2d = "2d" in text_lower or "two" in text_lower

    config = SimulationJobConfig(
        sim_type=SimulationType.HEAT_2D if is_2d else SimulationType.HEAT_1D,
        hypothesis_id=base["hypothesis_id"],
        grid_size=base["grid_size"],
        grid_size_y=base.get("grid_size_y"),
        dx=0.01,
        dt=0.0001,
        max_steps=1000,
        tolerance=base["tolerance"],
        boundary_conditions={
            "left": 0.0,
            "right": 0.0,
            "top": 0.0,
            "bottom": 0.0,
        },
        initial_condition="gaussian",
        solver_params={
            "alpha": float(params.get("alpha", 0.01)),
        },
        metadata=base["metadata"],
    )

    # Adjust boundary conditions from hypothesis
    if "hot" in text_lower:
        config.boundary_conditions["left"] = float(params.get("temp", 100))

    return config


def _create_laplace_config(hypothesis: Hypothesis, base: dict) -> SimulationJobConfig:
    """Create Laplace equation simulation config."""
    params = hypothesis.metadata.get("query_params", {})
    text_lower = hypothesis.text.lower()

    config = SimulationJobConfig(
        sim_type=SimulationType.LAPLACE_2D,
        hypothesis_id=base["hypothesis_id"],
        grid_size=base["grid_size"],
        grid_size_y=base.get("grid_size_y"),
        dx=0.1,
        max_iterations=5000,
        tolerance=base["tolerance"],
        boundary_conditions={
            "left": 0.0,
            "right": 100.0,
            "top": 50.0,
            "bottom": 50.0,
        },
        initial_condition="zero",
        solver_params={
            "omega": 1.0,  # Gauss-Seidel
        },
        metadata=base["metadata"],
    )

    # Extract temperatures from hypothesis
    if "temperature" in text_lower:
        temp = float(params.get("temp", 100))
        config.boundary_conditions["right"] = temp

    return config


def _create_ns_config(hypothesis: Hypothesis, base: dict) -> SimulationJobConfig:
    """Create Navier-Stokes simulation config."""
    params = hypothesis.metadata.get("query_params", {})
    text_lower = hypothesis.text.lower()

    config = SimulationJobConfig(
        sim_type=SimulationType.NAVIER_STOKES,
        hypothesis_id=base["hypothesis_id"],
        grid_size=base["grid_size"],
        grid_size_y=base.get("grid_size_y"),
        dx=0.1,
        dt=0.001,
        max_steps=500,
        tolerance=base["tolerance"],
        boundary_conditions={
            "inlet_velocity": 1.0,
            "outlet_pressure": 0.0,
        },
        initial_condition="zero",
        solver_params={
            "Re": float(params.get("Re", 100)),
            "viscosity": 0.01,
            "density": 1.0,
        },
        metadata=base["metadata"],
    )

    return config


def _create_pinn_config(hypothesis: Hypothesis, base: dict) -> SimulationJobConfig:
    """Create PINN training config."""
    params = hypothesis.metadata.get("query_params", {})

    config = SimulationJobConfig(
        sim_type=SimulationType.PINN,
        hypothesis_id=base["hypothesis_id"],
        grid_size=base["grid_size"],
        max_iterations=5000,  # epochs
        tolerance=1e-4,  # loss threshold
        solver_params={
            "layers": [2, 64, 64, 64, 1],
            "learning_rate": 0.001,
            "activation": "tanh",
            "lambda_physics": 1.0,
            "lambda_bc": 10.0,
        },
        metadata=base["metadata"],
    )

    return config


def _create_default_config(
    hypothesis: Hypothesis,
    sim_type: SimulationType,
    base: dict,
) -> SimulationJobConfig:
    """Create default simulation config when no specific type matched."""
    return SimulationJobConfig(
        sim_type=sim_type,
        hypothesis_id=base["hypothesis_id"],
        grid_size=base["grid_size"],
        grid_size_y=base.get("grid_size_y"),
        dx=0.01,
        dt=0.0001,
        max_steps=1000,
        max_iterations=5000,
        tolerance=base["tolerance"],
        boundary_conditions={
            "left": 0.0,
            "right": 100.0,
            "top": 50.0,
            "bottom": 50.0,
        },
        initial_condition="zero",
        metadata=base["metadata"],
    )


# =============================================================================
# Experiment Planning
# =============================================================================


def plan_experiments(hypothesis: Hypothesis) -> List[ExperimentConfig]:
    """
    Create experiment configurations from a hypothesis.

    Determines what data collection or analysis is needed.

    Args:
        hypothesis: Hypothesis to plan experiments for

    Returns:
        List of ExperimentConfig objects
    """
    logger.info(f"Planning experiments for hypothesis: {hypothesis.id}")

    configs = []
    text_lower = hypothesis.text.lower()

    # Check for OCR-related experiments
    if "paper" in text_lower or "document" in text_lower or "extract" in text_lower:
        configs.append(_create_ocr_experiment(hypothesis))

    # Check for webcam experiments
    if "camera" in text_lower or "visual" in text_lower or "image" in text_lower:
        configs.append(_create_webcam_experiment(hypothesis))

    # Check for file analysis
    if "data" in text_lower or "measurement" in text_lower or "file" in text_lower:
        configs.append(_create_file_experiment(hypothesis))

    # Check for sensor experiments
    if "sensor" in text_lower or "reading" in text_lower or "hardware" in text_lower:
        configs.append(_create_sensor_experiment(hypothesis))

    # Limit experiments
    configs = configs[:2]

    logger.info(f"Planned {len(configs)} experiments for hypothesis {hypothesis.id}")
    return configs


def _create_ocr_experiment(hypothesis: Hypothesis) -> ExperimentConfig:
    """Create OCR extraction experiment config."""
    return ExperimentConfig(
        exp_type=ExperimentType.OCR_EXTRACTION,
        hypothesis_id=hypothesis.id,
        source="",  # To be filled by user or discovered
        parameters={
            "extract_equations": True,
            "extract_tables": True,
            "extract_figures": True,
        },
        ocr_backend="auto",
        preprocessing=True,
        metadata={
            "purpose": "Extract data from scientific documents",
        },
    )


def _create_webcam_experiment(hypothesis: Hypothesis) -> ExperimentConfig:
    """Create webcam capture experiment config."""
    return ExperimentConfig(
        exp_type=ExperimentType.WEBCAM_CAPTURE,
        hypothesis_id=hypothesis.id,
        source="0",  # Default camera index
        parameters={
            "capture_mode": "single",  # or "continuous"
            "resolution": [640, 480],
            "format": "png",
        },
        metadata={
            "purpose": "Visual data collection",
        },
    )


def _create_file_experiment(hypothesis: Hypothesis) -> ExperimentConfig:
    """Create file analysis experiment config."""
    return ExperimentConfig(
        exp_type=ExperimentType.FILE_ANALYSIS,
        hypothesis_id=hypothesis.id,
        source="",  # To be filled
        parameters={
            "file_types": [".csv", ".json", ".txt"],
            "parse_numbers": True,
            "extract_columns": [],
        },
        output_format="json",
        metadata={
            "purpose": "Analyze measurement data files",
        },
    )


def _create_sensor_experiment(hypothesis: Hypothesis) -> ExperimentConfig:
    """Create sensor reading experiment config."""
    return ExperimentConfig(
        exp_type=ExperimentType.SENSOR_READ,
        hypothesis_id=hypothesis.id,
        source="",  # Sensor identifier
        parameters={
            "poll_interval": 0.1,
            "duration": 10.0,
            "average": True,
        },
        metadata={
            "purpose": "Collect sensor measurements",
        },
    )


# =============================================================================
# Utility Functions
# =============================================================================


def validate_simulation_config(config: SimulationJobConfig) -> Dict[str, Any]:
    """
    Validate a simulation configuration.

    Args:
        config: Configuration to validate

    Returns:
        Dictionary with validation results
    """
    issues = []
    warnings = []

    # Check grid size
    if config.grid_size < 10:
        issues.append("Grid size too small (< 10)")
    elif config.grid_size > 1000:
        warnings.append("Large grid size may be slow")

    # Check time step (for transient simulations)
    if config.sim_type in [SimulationType.HEAT_1D, SimulationType.HEAT_2D]:
        alpha = config.solver_params.get("alpha", 0.01)
        r = alpha * config.dt / (config.dx ** 2)
        if r > 0.5:
            issues.append(f"Unstable time step: r = {r:.4f} > 0.5")

    # Check tolerance
    if config.tolerance > 0.01:
        warnings.append("Tolerance may be too loose for accurate results")
    elif config.tolerance < 1e-12:
        warnings.append("Very tight tolerance may require many iterations")

    # Check iterations
    if config.max_iterations < 100:
        warnings.append("Low iteration limit may prevent convergence")

    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "warnings": warnings,
    }


def estimate_runtime(config: SimulationJobConfig) -> float:
    """
    Estimate simulation runtime in seconds.

    Args:
        config: Simulation configuration

    Returns:
        Estimated runtime in seconds
    """
    # Very rough estimates
    grid_factor = config.grid_size * (config.grid_size_y or config.grid_size)

    if config.sim_type == SimulationType.HEAT_1D:
        return config.max_steps * config.grid_size * 1e-6

    elif config.sim_type == SimulationType.HEAT_2D:
        return config.max_steps * grid_factor * 1e-6

    elif config.sim_type == SimulationType.LAPLACE_2D:
        return config.max_iterations * grid_factor * 1e-6

    elif config.sim_type == SimulationType.NAVIER_STOKES:
        return config.max_steps * grid_factor * 5e-6

    elif config.sim_type == SimulationType.PINN:
        return config.max_iterations * 0.01  # ~10ms per epoch

    return 1.0  # Default estimate
