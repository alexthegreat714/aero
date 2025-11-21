"""
Base agent class for Aero Agent framework.

Provides the foundation for all agent implementations with
configuration, logging, and event dispatch capabilities.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional
from dataclasses import dataclass, field

from aero.config.loader import get_config, AeroConfig
from aero.core.events import EventBus, Event


@dataclass
class AgentState:
    """Represents the current state of an agent."""

    status: str = "idle"  # idle, running, paused, stopped, error
    current_task: Optional[str] = None
    last_error: Optional[str] = None
    metrics: dict = field(default_factory=dict)


class BaseAgent(ABC):
    """
    Base class for all Aero agents.

    Provides:
    - Configuration management
    - Logging infrastructure
    - Event dispatch hooks
    - State management
    - Lifecycle methods

    Subclasses must implement the `run()` method.

    Example:
        class MyAgent(BaseAgent):
            def run(self):
                self.log_info("Starting my agent")
                # Agent logic here
                self.dispatch_event("task_completed", {"result": "success"})
    """

    def __init__(
        self,
        name: str,
        config: Optional[AeroConfig] = None,
        event_bus: Optional[EventBus] = None,
    ):
        """
        Initialize the base agent.

        Args:
            name: Unique name for this agent instance
            config: Configuration object (uses global config if not provided)
            event_bus: Event bus for event dispatch (creates new if not provided)
        """
        self.name = name
        self.config = config or get_config()
        self.event_bus = event_bus or EventBus()
        self.state = AgentState()

        # Setup logging
        self._logger = logging.getLogger(f"aero.agent.{name}")
        self._setup_logging()

        # Event hooks
        self._pre_run_hooks: list[Callable] = []
        self._post_run_hooks: list[Callable] = []

        self.log_info(f"Agent '{name}' initialized")

    def _setup_logging(self) -> None:
        """Configure agent-specific logging."""
        log_level = self.config.get("app.log_level", "INFO")
        self._logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # -------------------------------------------------------------------------
    # Logging Helpers
    # -------------------------------------------------------------------------

    def log_debug(self, message: str, **kwargs: Any) -> None:
        """Log a debug message."""
        self._logger.debug(message, **kwargs)

    def log_info(self, message: str, **kwargs: Any) -> None:
        """Log an info message."""
        self._logger.info(message, **kwargs)

    def log_warning(self, message: str, **kwargs: Any) -> None:
        """Log a warning message."""
        self._logger.warning(message, **kwargs)

    def log_error(self, message: str, **kwargs: Any) -> None:
        """Log an error message."""
        self._logger.error(message, **kwargs)

    def log_exception(self, message: str, **kwargs: Any) -> None:
        """Log an exception with traceback."""
        self._logger.exception(message, **kwargs)

    # -------------------------------------------------------------------------
    # Event Dispatch
    # -------------------------------------------------------------------------

    def dispatch_event(self, event_type: str, data: Optional[dict] = None) -> None:
        """
        Dispatch an event to the event bus.

        Args:
            event_type: Type of event (e.g., "task_started", "task_completed")
            data: Optional event data
        """
        event = Event(
            type=event_type,
            source=self.name,
            data=data or {},
        )
        self.event_bus.emit(event)
        self.log_debug(f"Dispatched event: {event_type}")

    def subscribe(self, event_type: str, handler: Callable[[Event], None]) -> None:
        """
        Subscribe to events of a specific type.

        Args:
            event_type: Type of event to listen for
            handler: Callback function to handle the event
        """
        self.event_bus.subscribe(event_type, handler)

    # -------------------------------------------------------------------------
    # Lifecycle Hooks
    # -------------------------------------------------------------------------

    def add_pre_run_hook(self, hook: Callable[["BaseAgent"], None]) -> None:
        """Add a hook to run before the main run() method."""
        self._pre_run_hooks.append(hook)

    def add_post_run_hook(self, hook: Callable[["BaseAgent", Any], None]) -> None:
        """Add a hook to run after the main run() method."""
        self._post_run_hooks.append(hook)

    def _execute_pre_run_hooks(self) -> None:
        """Execute all pre-run hooks."""
        for hook in self._pre_run_hooks:
            try:
                hook(self)
            except Exception as e:
                self.log_error(f"Pre-run hook failed: {e}")

    def _execute_post_run_hooks(self, result: Any) -> None:
        """Execute all post-run hooks."""
        for hook in self._post_run_hooks:
            try:
                hook(self, result)
            except Exception as e:
                self.log_error(f"Post-run hook failed: {e}")

    # -------------------------------------------------------------------------
    # State Management
    # -------------------------------------------------------------------------

    def set_status(self, status: str) -> None:
        """Update the agent's status."""
        old_status = self.state.status
        self.state.status = status
        self.dispatch_event("status_changed", {"old": old_status, "new": status})

    def set_current_task(self, task: Optional[str]) -> None:
        """Update the current task being executed."""
        self.state.current_task = task
        if task:
            self.dispatch_event("task_started", {"task": task})

    def update_metric(self, key: str, value: Any) -> None:
        """Update a metric value."""
        self.state.metrics[key] = value

    # -------------------------------------------------------------------------
    # Main Execution
    # -------------------------------------------------------------------------

    @abstractmethod
    def run(self) -> Any:
        """
        Execute the agent's main logic.

        Subclasses must implement this method.

        Returns:
            Result of the agent's execution
        """
        raise NotImplementedError("Subclasses must implement run()")

    def execute(self) -> Any:
        """
        Execute the agent with lifecycle management.

        This method:
        1. Sets status to 'running'
        2. Executes pre-run hooks
        3. Calls the run() method
        4. Executes post-run hooks
        5. Handles any errors

        Returns:
            Result of the run() method
        """
        self.set_status("running")
        self.dispatch_event("agent_started")
        result = None

        try:
            self._execute_pre_run_hooks()
            result = self.run()
            self._execute_post_run_hooks(result)
            self.set_status("idle")
            self.dispatch_event("agent_completed", {"result": result})
        except Exception as e:
            self.state.last_error = str(e)
            self.set_status("error")
            self.log_exception(f"Agent execution failed: {e}")
            self.dispatch_event("agent_error", {"error": str(e)})
            raise

        return result

    def stop(self) -> None:
        """Request the agent to stop."""
        self.set_status("stopped")
        self.dispatch_event("agent_stopped")
        self.log_info("Agent stop requested")

    def pause(self) -> None:
        """Request the agent to pause."""
        self.set_status("paused")
        self.dispatch_event("agent_paused")
        self.log_info("Agent pause requested")

    def resume(self) -> None:
        """Request the agent to resume."""
        if self.state.status == "paused":
            self.set_status("running")
            self.dispatch_event("agent_resumed")
            self.log_info("Agent resumed")

    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------

    def get_status(self) -> dict:
        """Get the current agent status as a dictionary."""
        return {
            "name": self.name,
            "status": self.state.status,
            "current_task": self.state.current_task,
            "last_error": self.state.last_error,
            "metrics": self.state.metrics,
        }

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(name='{self.name}', status='{self.state.status}')>"
