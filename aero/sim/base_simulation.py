"""
Base simulation class for Aero Agent.

Provides the foundation for all simulation implementations.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import time

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SimulationResult:
    """Container for simulation results."""

    success: bool
    data: dict = field(default_factory=dict)
    iterations: int = 0
    residual: float = 0.0
    elapsed_time: float = 0.0
    metadata: dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        """Convert result to dictionary."""
        return {
            "success": self.success,
            "iterations": self.iterations,
            "residual": self.residual,
            "elapsed_time": self.elapsed_time,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class SimulationConfig:
    """Configuration for simulation runs."""

    max_iterations: int = 10000
    convergence_threshold: float = 1e-6
    device: str = "auto"  # auto, cpu, cuda
    verbose: bool = False
    save_intermediate: bool = False
    output_dir: Optional[str] = None


class BaseSimulation(ABC):
    """
    Base class for all Aero simulations.

    Provides:
    - Configuration management
    - Device selection (CPU/GPU)
    - Convergence tracking
    - Result handling

    Subclasses must implement:
    - setup(): Initialize simulation
    - step(): Execute one simulation step
    - run(): Execute the full simulation

    Example:
        class MySimulation(BaseSimulation):
            def setup(self):
                # Initialize mesh, boundary conditions, etc.
                pass

            def step(self):
                # Execute one iteration
                return residual

            def run(self):
                self.setup()
                for i in range(self.config.max_iterations):
                    residual = self.step()
                    if residual < self.config.convergence_threshold:
                        break
                return SimulationResult(success=True)
    """

    def __init__(self, config: Optional[SimulationConfig] = None):
        """
        Initialize the simulation.

        Args:
            config: Simulation configuration
        """
        self.config = config or SimulationConfig()
        self._logger = logging.getLogger(f"aero.sim.{self.name}")
        self._device = self._select_device()
        self._iteration = 0
        self._residual_history: list[float] = []

    @property
    @abstractmethod
    def name(self) -> str:
        """Simulation name."""
        raise NotImplementedError

    @property
    def device(self) -> str:
        """Current compute device."""
        return self._device

    def _select_device(self) -> str:
        """Select compute device based on configuration and availability."""
        if self.config.device == "auto":
            try:
                import torch
                if torch.cuda.is_available():
                    device = "cuda"
                    self._logger.info(f"Auto-selected device: {device}")
                    return device
            except ImportError:
                pass
            return "cpu"
        return self.config.device

    # -------------------------------------------------------------------------
    # Abstract Methods
    # -------------------------------------------------------------------------

    @abstractmethod
    def setup(self) -> None:
        """
        Initialize the simulation.

        Called before the simulation loop begins. Subclasses should
        set up mesh, boundary conditions, initial conditions, etc.
        """
        raise NotImplementedError

    @abstractmethod
    def step(self) -> float:
        """
        Execute one simulation step.

        Returns:
            Residual or error metric for convergence checking
        """
        raise NotImplementedError

    @abstractmethod
    def run(self) -> SimulationResult:
        """
        Execute the simulation.

        Returns:
            SimulationResult with outcome and data
        """
        raise NotImplementedError

    # -------------------------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------------------------

    def is_converged(self, residual: float) -> bool:
        """Check if simulation has converged."""
        return residual < self.config.convergence_threshold

    def log_progress(self, iteration: int, residual: float) -> None:
        """Log simulation progress."""
        if self.config.verbose:
            self._logger.info(f"Iteration {iteration}: residual = {residual:.2e}")

    def run_loop(self) -> SimulationResult:
        """
        Run the standard simulation loop.

        Convenience method that handles setup, iteration, and convergence.
        """
        start_time = time.time()

        try:
            self.setup()
        except Exception as e:
            self._logger.error(f"Setup failed: {e}")
            return SimulationResult(success=False, metadata={"error": str(e)})

        converged = False
        final_residual = float("inf")

        for i in range(self.config.max_iterations):
            self._iteration = i

            try:
                residual = self.step()
            except Exception as e:
                self._logger.error(f"Step {i} failed: {e}")
                return SimulationResult(
                    success=False,
                    iterations=i,
                    metadata={"error": str(e)},
                )

            final_residual = residual
            self._residual_history.append(residual)
            self.log_progress(i, residual)

            if self.is_converged(residual):
                converged = True
                self._logger.info(f"Converged at iteration {i}")
                break

        elapsed = time.time() - start_time

        return SimulationResult(
            success=converged,
            iterations=self._iteration + 1,
            residual=final_residual,
            elapsed_time=elapsed,
            metadata={
                "converged": converged,
                "device": self._device,
            },
        )

    def get_residual_history(self) -> list[float]:
        """Get the residual history."""
        return self._residual_history.copy()

    def reset(self) -> None:
        """Reset the simulation state."""
        self._iteration = 0
        self._residual_history.clear()
