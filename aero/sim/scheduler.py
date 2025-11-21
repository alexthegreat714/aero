"""
Simulation job scheduler for Aero Agent.

Provides FIFO queue management for simulation jobs.
Single-threaded execution - no concurrency.
"""

import logging
from collections import deque
from datetime import datetime
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from aero.sim.job import SimulationJob
    from aero.sim.results import SimulationResult

logger = logging.getLogger(__name__)


class SimulationScheduler:
    """
    Simple FIFO job scheduler for simulations.

    Provides:
    - Job queue management
    - Sequential execution
    - Job tracking and history
    - Result retrieval

    Example:
        scheduler = SimulationScheduler()

        # Add jobs
        scheduler.add_job(job1)
        scheduler.add_job(job2)

        # Run all jobs
        scheduler.run_all()

        # Get results
        for job_id, result in scheduler.get_results().items():
            print(f"{job_id}: {result.is_success}")
    """

    def __init__(self):
        """Initialize the scheduler."""
        self._queue: deque["SimulationJob"] = deque()
        self._jobs: Dict[str, "SimulationJob"] = {}
        self._history: List[str] = []
        self._running: Optional[str] = None

        logger.info("SimulationScheduler initialized")

    def add_job(self, job: "SimulationJob") -> str:
        """
        Add a job to the queue.

        Args:
            job: SimulationJob to queue

        Returns:
            Job ID
        """
        self._queue.append(job)
        self._jobs[job.id] = job

        logger.info(f"Added job {job.id} ({job.sim_type}) to queue")
        return job.id

    def run_next(self) -> Optional["SimulationResult"]:
        """
        Run the next job in the queue.

        Returns:
            SimulationResult or None if queue is empty
        """
        if not self._queue:
            logger.debug("Queue is empty")
            return None

        job = self._queue.popleft()
        self._running = job.id

        logger.info(f"Running job {job.id}")

        try:
            result = job.run()
            self._history.append(job.id)
            return result
        finally:
            self._running = None

    def run_all(self) -> Dict[str, "SimulationResult"]:
        """
        Run all jobs in the queue.

        Returns:
            Dictionary mapping job IDs to results
        """
        results = {}

        logger.info(f"Running {len(self._queue)} jobs")

        while self._queue:
            job = self._queue[0]  # Peek
            result = self.run_next()
            if result is not None:
                results[job.id] = result

        logger.info(f"Completed {len(results)} jobs")
        return results

    def get_job(self, job_id: str) -> Optional["SimulationJob"]:
        """
        Get a job by ID.

        Args:
            job_id: Job identifier

        Returns:
            SimulationJob or None if not found
        """
        return self._jobs.get(job_id)

    def get_result(self, job_id: str) -> Optional["SimulationResult"]:
        """
        Get the result for a job.

        Args:
            job_id: Job identifier

        Returns:
            SimulationResult or None if job not found or not completed
        """
        job = self._jobs.get(job_id)
        if job and job.result:
            return job.result
        return None

    def get_results(self) -> Dict[str, "SimulationResult"]:
        """
        Get all completed job results.

        Returns:
            Dictionary mapping job IDs to results
        """
        return {
            job_id: job.result
            for job_id, job in self._jobs.items()
            if job.result is not None
        }

    def list_jobs(self, status: Optional[str] = None) -> List[dict]:
        """
        List all jobs.

        Args:
            status: Optional status filter

        Returns:
            List of job dictionaries
        """
        jobs = []
        for job in self._jobs.values():
            if status is None or job.status.value == status:
                jobs.append(job.to_dict())
        return jobs

    def list_queued(self) -> List[str]:
        """Get list of queued job IDs."""
        return [job.id for job in self._queue]

    def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a queued job.

        Args:
            job_id: Job identifier

        Returns:
            True if cancelled, False if not found or not queued
        """
        job = self._jobs.get(job_id)
        if job and job.status.value == "queued":
            job.cancel()
            # Remove from queue
            self._queue = deque(j for j in self._queue if j.id != job_id)
            return True
        return False

    def clear_queue(self) -> int:
        """
        Clear all queued jobs.

        Returns:
            Number of jobs cleared
        """
        count = len(self._queue)

        for job in self._queue:
            job.cancel()

        self._queue.clear()
        logger.info(f"Cleared {count} jobs from queue")
        return count

    def clear_history(self) -> None:
        """Clear completed job history (keeps jobs in _jobs dict)."""
        self._history.clear()

    @property
    def queue_size(self) -> int:
        """Get number of jobs in queue."""
        return len(self._queue)

    @property
    def total_jobs(self) -> int:
        """Get total number of jobs (queued + completed)."""
        return len(self._jobs)

    @property
    def running_job(self) -> Optional[str]:
        """Get ID of currently running job."""
        return self._running

    def get_stats(self) -> dict:
        """
        Get scheduler statistics.

        Returns:
            Dictionary with queue stats
        """
        completed = sum(1 for j in self._jobs.values() if j.status.value == "completed")
        errors = sum(1 for j in self._jobs.values() if j.status.value == "error")

        return {
            "queued": self.queue_size,
            "running": 1 if self._running else 0,
            "completed": completed,
            "errors": errors,
            "total": self.total_jobs,
        }


# Global scheduler instance
_global_scheduler: Optional[SimulationScheduler] = None


def get_scheduler() -> SimulationScheduler:
    """
    Get the global scheduler instance.

    Returns:
        SimulationScheduler instance
    """
    global _global_scheduler
    if _global_scheduler is None:
        _global_scheduler = SimulationScheduler()
    return _global_scheduler


def reset_scheduler() -> None:
    """Reset the global scheduler."""
    global _global_scheduler
    _global_scheduler = None
