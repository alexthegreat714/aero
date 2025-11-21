"""
Main processing loop for Aero Agent.

Orchestrates the agent's workflow and task execution.
"""

import logging
import time
import threading
from typing import Any, Callable, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from queue import Queue, Empty

from aero.config.loader import get_config
from aero.core.events import EventBus, Event

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Status of a pipeline task."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class PipelineTask:
    """Represents a task in the pipeline."""

    id: str
    name: str
    handler: Callable[..., Any]
    args: tuple = field(default_factory=tuple)
    kwargs: dict = field(default_factory=dict)
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "status": self.status.value,
            "error": self.error,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


class AeroLoop:
    """
    Main processing loop for the Aero Agent.

    Provides:
    - Task queue management
    - Concurrent task execution
    - Event-driven workflow
    - Error handling and retry

    Example:
        loop = AeroLoop()

        def process_data(data):
            return data.upper()

        loop.submit("task1", "uppercase", process_data, args=("hello",))
        loop.start()
        # ... later
        loop.stop()
    """

    def __init__(
        self,
        max_workers: int = 4,
        event_bus: Optional[EventBus] = None,
    ):
        """
        Initialize the Aero Loop.

        Args:
            max_workers: Maximum concurrent workers
            event_bus: Event bus for notifications
        """
        self.max_workers = max_workers
        self.event_bus = event_bus or EventBus()

        self._task_queue: Queue[PipelineTask] = Queue()
        self._active_tasks: dict[str, PipelineTask] = {}
        self._completed_tasks: dict[str, PipelineTask] = {}

        self._workers: list[threading.Thread] = []
        self._running = False
        self._lock = threading.RLock()

        self._config = get_config()

        logger.info(f"AeroLoop initialized with {max_workers} workers")

    @property
    def is_running(self) -> bool:
        """Check if the loop is running."""
        return self._running

    def submit(
        self,
        task_id: str,
        name: str,
        handler: Callable[..., Any],
        args: tuple = (),
        kwargs: dict = None,
    ) -> PipelineTask:
        """
        Submit a task to the pipeline.

        Args:
            task_id: Unique task identifier
            name: Human-readable task name
            handler: Function to execute
            args: Positional arguments
            kwargs: Keyword arguments

        Returns:
            The created PipelineTask
        """
        task = PipelineTask(
            id=task_id,
            name=name,
            handler=handler,
            args=args,
            kwargs=kwargs or {},
        )

        with self._lock:
            self._active_tasks[task_id] = task

        self._task_queue.put(task)
        self.event_bus.emit_simple("task_submitted", "aero_loop", {"task_id": task_id})

        logger.debug(f"Task submitted: {task_id} ({name})")
        return task

    def start(self) -> None:
        """Start the processing loop."""
        if self._running:
            logger.warning("AeroLoop already running")
            return

        self._running = True

        # Start worker threads
        for i in range(self.max_workers):
            worker = threading.Thread(
                target=self._worker_loop,
                name=f"aero-worker-{i}",
                daemon=True,
            )
            worker.start()
            self._workers.append(worker)

        self.event_bus.emit_simple("loop_started", "aero_loop")
        logger.info(f"AeroLoop started with {self.max_workers} workers")

    def stop(self, wait: bool = True, timeout: float = 30.0) -> None:
        """
        Stop the processing loop.

        Args:
            wait: Wait for workers to finish
            timeout: Maximum time to wait
        """
        self._running = False

        if wait:
            for worker in self._workers:
                worker.join(timeout=timeout / len(self._workers))

        self._workers.clear()
        self.event_bus.emit_simple("loop_stopped", "aero_loop")
        logger.info("AeroLoop stopped")

    def _worker_loop(self) -> None:
        """Worker thread main loop."""
        while self._running:
            try:
                task = self._task_queue.get(timeout=1.0)
                self._execute_task(task)
            except Empty:
                continue
            except Exception as e:
                logger.exception(f"Worker error: {e}")

    def _execute_task(self, task: PipelineTask) -> None:
        """Execute a single task."""
        task.status = TaskStatus.RUNNING
        task.started_at = datetime.now()

        self.event_bus.emit_simple(
            "task_started",
            "aero_loop",
            {"task_id": task.id, "name": task.name},
        )

        try:
            result = task.handler(*task.args, **task.kwargs)
            task.result = result
            task.status = TaskStatus.COMPLETED

            self.event_bus.emit_simple(
                "task_completed",
                "aero_loop",
                {"task_id": task.id, "name": task.name},
            )

            logger.debug(f"Task completed: {task.id}")

        except Exception as e:
            task.error = str(e)
            task.status = TaskStatus.FAILED

            self.event_bus.emit_simple(
                "task_failed",
                "aero_loop",
                {"task_id": task.id, "error": str(e)},
            )

            logger.error(f"Task failed: {task.id} - {e}")

        finally:
            task.completed_at = datetime.now()

            with self._lock:
                if task.id in self._active_tasks:
                    del self._active_tasks[task.id]
                self._completed_tasks[task.id] = task

    def get_task(self, task_id: str) -> Optional[PipelineTask]:
        """Get a task by ID."""
        with self._lock:
            return (
                self._active_tasks.get(task_id) or
                self._completed_tasks.get(task_id)
            )

    def get_status(self) -> dict:
        """Get loop status."""
        with self._lock:
            return {
                "running": self._running,
                "workers": len(self._workers),
                "queued": self._task_queue.qsize(),
                "active": len(self._active_tasks),
                "completed": len(self._completed_tasks),
            }

    def wait_for_task(self, task_id: str, timeout: float = None) -> Optional[PipelineTask]:
        """
        Wait for a task to complete.

        Args:
            task_id: Task ID to wait for
            timeout: Maximum time to wait

        Returns:
            Completed task or None if timeout
        """
        start = time.time()

        while True:
            task = self.get_task(task_id)

            if task and task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED):
                return task

            if timeout and (time.time() - start) > timeout:
                return None

            time.sleep(0.1)
