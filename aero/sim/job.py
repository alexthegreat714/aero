"""
Simulation job management for Aero Agent.

Provides structured job execution with status tracking and result capture.
"""

import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from aero.sim.base_simulation import BaseSimulation
    from aero.sim.results import SimulationResult

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    """Simulation job status."""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    ERROR = "error"
    CANCELLED = "cancelled"


@dataclass
class SimulationJob:
    """
    Represents a simulation job in the queue.

    Tracks:
    - Job identity and type
    - Simulation instance
    - Execution status
    - Timing information
    - Result capture

    Example:
        from aero.sim.numerics.fd_solver import HeatEquation

        sim = HeatEquation(nx=100, ny=100)
        job = SimulationJob(sim_type="heat_1d", sim=sim)

        result = job.run()
        print(f"Job {job.id} completed: {job.status}")
    """

    sim_type: str
    sim: "BaseSimulation"
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    status: JobStatus = JobStatus.QUEUED
    created: datetime = field(default_factory=datetime.now)
    started: Optional[datetime] = None
    completed: Optional[datetime] = None
    result: Optional["SimulationResult"] = None
    error: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    def run(self) -> "SimulationResult":
        """
        Execute the simulation job.

        Updates status throughout execution and captures results or errors.

        Returns:
            SimulationResult from the simulation
        """
        from aero.sim.results import SimulationResult

        logger.info(f"Starting job {self.id} ({self.sim_type})")

        self.status = JobStatus.RUNNING
        self.started = datetime.now()

        try:
            # Run the simulation
            if hasattr(self.sim, 'execute'):
                # New-style simulation with execute()
                self.result = self.sim.execute()
            elif hasattr(self.sim, 'run'):
                # Old-style simulation with run()
                result = self.sim.run()
                # Convert SimulationResult if needed
                if hasattr(result, 'data'):
                    self.result = SimulationResult(
                        fields=result.data,
                        metadata={
                            "success": result.success,
                            "iterations": result.iterations,
                            "residual": result.residual,
                            "elapsed_time": result.elapsed_time,
                            **result.metadata,
                        }
                    )
                else:
                    self.result = result
            else:
                raise AttributeError("Simulation has no run() or execute() method")

            self.status = JobStatus.COMPLETED
            logger.info(f"Job {self.id} completed successfully")

        except Exception as e:
            self.status = JobStatus.ERROR
            self.error = str(e)
            logger.error(f"Job {self.id} failed: {e}", exc_info=True)

            # Create error result
            self.result = SimulationResult(
                fields={},
                metadata={
                    "status": "error",
                    "error": str(e),
                    "job_id": self.id,
                }
            )

        finally:
            self.completed = datetime.now()

        return self.result

    @property
    def runtime(self) -> Optional[float]:
        """Get job runtime in seconds."""
        if self.started and self.completed:
            return (self.completed - self.started).total_seconds()
        elif self.started:
            return (datetime.now() - self.started).total_seconds()
        return None

    @property
    def is_finished(self) -> bool:
        """Check if job has finished (success or error)."""
        return self.status in (JobStatus.COMPLETED, JobStatus.ERROR, JobStatus.CANCELLED)

    @property
    def is_success(self) -> bool:
        """Check if job completed successfully."""
        return self.status == JobStatus.COMPLETED

    def cancel(self) -> None:
        """Cancel the job if queued."""
        if self.status == JobStatus.QUEUED:
            self.status = JobStatus.CANCELLED
            self.completed = datetime.now()
            logger.info(f"Job {self.id} cancelled")

    def to_dict(self) -> dict:
        """Convert job to dictionary representation."""
        return {
            "id": self.id,
            "sim_type": self.sim_type,
            "status": self.status.value,
            "created": self.created.isoformat(),
            "started": self.started.isoformat() if self.started else None,
            "completed": self.completed.isoformat() if self.completed else None,
            "runtime": self.runtime,
            "error": self.error,
            "metadata": self.metadata,
        }

    def __repr__(self) -> str:
        return f"SimulationJob(id={self.id}, type={self.sim_type}, status={self.status.value})"


def create_job(
    sim_type: str,
    sim: "BaseSimulation",
    metadata: Optional[dict] = None,
) -> SimulationJob:
    """
    Create a new simulation job.

    Args:
        sim_type: Type identifier for the simulation
        sim: Simulation instance to execute
        metadata: Optional metadata dictionary

    Returns:
        New SimulationJob instance
    """
    return SimulationJob(
        sim_type=sim_type,
        sim=sim,
        metadata=metadata or {},
    )


def run_job(job: SimulationJob) -> "SimulationResult":
    """
    Execute a simulation job and return the result.

    Args:
        job: SimulationJob to execute

    Returns:
        SimulationResult from the job
    """
    return job.run()
