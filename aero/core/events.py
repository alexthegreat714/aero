"""
Event system for Aero Agent framework.

Provides a simple pub/sub event bus for inter-component communication.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Optional
from collections import defaultdict
import threading

logger = logging.getLogger(__name__)


@dataclass
class Event:
    """
    Represents an event in the system.

    Attributes:
        type: Event type identifier (e.g., "task_completed")
        source: Source of the event (e.g., agent name)
        data: Event payload
        timestamp: When the event was created
    """

    type: str
    source: str
    data: dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        """Convert event to dictionary."""
        return {
            "type": self.type,
            "source": self.source,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
        }


class EventBus:
    """
    Simple event bus for pub/sub communication.

    Thread-safe implementation supporting:
    - Event subscription by type
    - Wildcard subscriptions
    - Event history (optional)
    - Async handlers (future)

    Example:
        bus = EventBus()

        def on_task_completed(event):
            print(f"Task completed: {event.data}")

        bus.subscribe("task_completed", on_task_completed)
        bus.emit(Event(type="task_completed", source="agent1", data={"result": "ok"}))
    """

    def __init__(self, keep_history: bool = False, max_history: int = 1000):
        """
        Initialize the event bus.

        Args:
            keep_history: Whether to keep event history
            max_history: Maximum number of events to keep in history
        """
        self._subscribers: dict[str, list[Callable[[Event], None]]] = defaultdict(list)
        self._wildcard_subscribers: list[Callable[[Event], None]] = []
        self._lock = threading.RLock()
        self._keep_history = keep_history
        self._max_history = max_history
        self._history: list[Event] = []

    def subscribe(
        self,
        event_type: str,
        handler: Callable[[Event], None]
    ) -> Callable[[], None]:
        """
        Subscribe to events of a specific type.

        Args:
            event_type: Event type to subscribe to ("*" for all events)
            handler: Callback function to handle events

        Returns:
            Unsubscribe function
        """
        with self._lock:
            if event_type == "*":
                self._wildcard_subscribers.append(handler)
            else:
                self._subscribers[event_type].append(handler)

        def unsubscribe():
            self.unsubscribe(event_type, handler)

        return unsubscribe

    def unsubscribe(
        self,
        event_type: str,
        handler: Callable[[Event], None]
    ) -> None:
        """
        Unsubscribe a handler from an event type.

        Args:
            event_type: Event type to unsubscribe from
            handler: Handler to remove
        """
        with self._lock:
            if event_type == "*":
                if handler in self._wildcard_subscribers:
                    self._wildcard_subscribers.remove(handler)
            else:
                if handler in self._subscribers[event_type]:
                    self._subscribers[event_type].remove(handler)

    def emit(self, event: Event) -> None:
        """
        Emit an event to all subscribers.

        Args:
            event: Event to emit
        """
        with self._lock:
            handlers = list(self._subscribers.get(event.type, []))
            wildcard_handlers = list(self._wildcard_subscribers)

            if self._keep_history:
                self._history.append(event)
                if len(self._history) > self._max_history:
                    self._history = self._history[-self._max_history:]

        # Call handlers outside of lock
        for handler in handlers + wildcard_handlers:
            try:
                handler(event)
            except Exception as e:
                logger.error(f"Event handler error for '{event.type}': {e}")

    def emit_simple(
        self,
        event_type: str,
        source: str,
        data: Optional[dict] = None
    ) -> None:
        """
        Emit an event using simple parameters.

        Args:
            event_type: Type of event
            source: Event source
            data: Event data
        """
        event = Event(type=event_type, source=source, data=data or {})
        self.emit(event)

    def get_history(
        self,
        event_type: Optional[str] = None,
        limit: Optional[int] = None
    ) -> list[Event]:
        """
        Get event history, optionally filtered by type.

        Args:
            event_type: Optional type to filter by
            limit: Maximum number of events to return

        Returns:
            List of events
        """
        with self._lock:
            history = list(self._history)

        if event_type:
            history = [e for e in history if e.type == event_type]

        if limit:
            history = history[-limit:]

        return history

    def clear_history(self) -> None:
        """Clear the event history."""
        with self._lock:
            self._history.clear()

    def clear_subscribers(self) -> None:
        """Remove all subscribers."""
        with self._lock:
            self._subscribers.clear()
            self._wildcard_subscribers.clear()

    @property
    def subscriber_count(self) -> int:
        """Get total number of subscribers."""
        with self._lock:
            count = sum(len(handlers) for handlers in self._subscribers.values())
            count += len(self._wildcard_subscribers)
            return count


# Global event bus instance
_global_bus: Optional[EventBus] = None


def get_event_bus() -> EventBus:
    """Get the global event bus instance."""
    global _global_bus
    if _global_bus is None:
        _global_bus = EventBus(keep_history=True)
    return _global_bus


def emit_event(event_type: str, source: str, data: Optional[dict] = None) -> None:
    """Convenience function to emit an event on the global bus."""
    get_event_bus().emit_simple(event_type, source, data)


def subscribe_event(
    event_type: str,
    handler: Callable[[Event], None]
) -> Callable[[], None]:
    """Convenience function to subscribe to events on the global bus."""
    return get_event_bus().subscribe(event_type, handler)
