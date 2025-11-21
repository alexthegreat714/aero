"""
Local message bus for agent communication.

Provides in-process message routing between agents.
This is NOT a network layer - external agents communicate via HTTP,
which is then mapped to AgentMessage objects and routed through this bus.
"""

import logging
import time
from datetime import datetime
from typing import Callable, Dict, List, Optional

from aero.agents.messages import (
    AgentMessage,
    AgentMessageLogEntry,
    MESSAGE_KIND_RESPONSE,
)

logger = logging.getLogger(__name__)


class AgentBus:
    """
    Simple in-memory bus for agent messages.

    This is an abstraction for:
    - Routing AgentMessage objects to registered handlers
    - Logging all messages
    - Dispatching to handlers synchronously

    Example:
        bus = AgentBus()

        def my_handler(msg: AgentMessage) -> dict:
            return {"status": "ok", "data": {"received": msg.action}}

        bus.register_handler("MyAgent", my_handler)

        response = bus.send(AgentMessage.new(
            sender="Caller",
            recipient="MyAgent",
            kind="request",
            action="test",
            payload={},
        ))
    """

    def __init__(self, enabled: bool = True, log_limit: int = 1000):
        """
        Initialize the agent bus.

        Args:
            enabled: Whether the bus is enabled for message routing
            log_limit: Maximum number of messages to keep in log
        """
        self.enabled = enabled
        self.log_limit = log_limit
        self._handlers: Dict[str, Callable[[AgentMessage], dict]] = {}
        self._log: List[AgentMessageLogEntry] = []

        logger.info(f"AgentBus initialized (enabled={enabled}, log_limit={log_limit})")

    def register_handler(
        self,
        agent_name: str,
        handler: Callable[[AgentMessage], dict],
    ) -> None:
        """
        Register a message handler for an agent.

        Args:
            agent_name: Name of the agent
            handler: Function that takes AgentMessage and returns dict
        """
        self._handlers[agent_name] = handler
        logger.debug(f"Registered handler for agent: {agent_name}")

    def unregister_handler(self, agent_name: str) -> bool:
        """
        Unregister a handler for an agent.

        Args:
            agent_name: Name of the agent

        Returns:
            True if handler was removed, False if not found
        """
        if agent_name in self._handlers:
            del self._handlers[agent_name]
            logger.debug(f"Unregistered handler for agent: {agent_name}")
            return True
        return False

    def has_handler(self, agent_name: str) -> bool:
        """Check if a handler is registered for an agent."""
        return agent_name in self._handlers

    def list_handlers(self) -> List[str]:
        """List all registered handler agent names."""
        return list(self._handlers.keys())

    def send(self, msg: AgentMessage) -> dict:
        """
        Synchronously deliver message to recipient handler if registered.

        Logs all messages and returns handler's result as dict.

        Args:
            msg: AgentMessage to send

        Returns:
            Handler's result as dict, or error dict if handler not found
        """
        start_time = time.time()
        timestamp = datetime.utcnow().isoformat() + "Z"

        # Create log entry
        log_entry = AgentMessageLogEntry(
            message=msg,
            timestamp=timestamp,
            processed=False,
        )

        # Check if bus is enabled
        if not self.enabled:
            log_entry.result_status = "skipped"
            log_entry.error_message = "Bus messaging is disabled"
            self._add_log_entry(log_entry)

            logger.debug(f"Bus disabled, skipping message: {msg.id}")
            return {
                "status": "skipped",
                "message": "Agent bus messaging is disabled",
                "message_id": msg.id,
            }

        # Check for handler
        handler = self._handlers.get(msg.recipient)
        if handler is None:
            log_entry.result_status = "error"
            log_entry.error_message = f"No handler registered for: {msg.recipient}"
            self._add_log_entry(log_entry)

            logger.warning(f"No handler for recipient: {msg.recipient}")
            return {
                "status": "error",
                "message": f"No handler registered for agent: {msg.recipient}",
                "message_id": msg.id,
            }

        # Execute handler
        try:
            result = handler(msg)
            elapsed_ms = (time.time() - start_time) * 1000

            log_entry.processed = True
            log_entry.result_status = result.get("status", "ok")
            log_entry.processing_time_ms = elapsed_ms
            self._add_log_entry(log_entry)

            logger.debug(
                f"Message {msg.id} processed: {msg.sender} -> {msg.recipient} "
                f"({msg.action}) in {elapsed_ms:.2f}ms"
            )

            # Add message metadata to result
            result["message_id"] = msg.id
            result["correlation_id"] = msg.correlation_id
            return result

        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000

            log_entry.processed = True
            log_entry.result_status = "error"
            log_entry.error_message = str(e)
            log_entry.processing_time_ms = elapsed_ms
            self._add_log_entry(log_entry)

            logger.exception(f"Error processing message {msg.id}: {e}")
            return {
                "status": "error",
                "message": str(e),
                "message_id": msg.id,
            }

    def send_and_create_response(
        self,
        msg: AgentMessage,
    ) -> tuple[dict, Optional[AgentMessage]]:
        """
        Send a message and create a response AgentMessage.

        Args:
            msg: AgentMessage to send

        Returns:
            Tuple of (result dict, response AgentMessage or None)
        """
        result = self.send(msg)

        if result.get("status") == "ok":
            response_msg = AgentMessage.response_to(
                original=msg,
                sender=msg.recipient,
                payload=result,
            )
            return result, response_msg

        return result, None

    def _add_log_entry(self, entry: AgentMessageLogEntry) -> None:
        """Add a log entry, maintaining log limit."""
        self._log.append(entry)

        # Trim log if over limit
        if len(self._log) > self.log_limit:
            self._log = self._log[-self.log_limit:]

    def get_log(self, limit: int = 100) -> List[AgentMessageLogEntry]:
        """
        Get recent messages from the log.

        Args:
            limit: Maximum number of entries to return

        Returns:
            List of AgentMessageLogEntry (most recent last)
        """
        return self._log[-limit:]

    def get_log_dicts(self, limit: int = 100) -> List[dict]:
        """
        Get recent messages from the log as dictionaries.

        Args:
            limit: Maximum number of entries to return

        Returns:
            List of log entry dicts
        """
        return [entry.to_dict() for entry in self.get_log(limit)]

    def clear_log(self) -> int:
        """
        Clear the message log.

        Returns:
            Number of entries cleared
        """
        count = len(self._log)
        self._log.clear()
        return count

    def get_stats(self) -> dict:
        """Get bus statistics."""
        total_messages = len(self._log)
        processed = sum(1 for e in self._log if e.processed)
        errors = sum(1 for e in self._log if e.result_status == "error")

        return {
            "enabled": self.enabled,
            "handlers_registered": len(self._handlers),
            "handler_names": list(self._handlers.keys()),
            "log_size": total_messages,
            "log_limit": self.log_limit,
            "messages_processed": processed,
            "messages_errors": errors,
        }


# Module-level default bus instance
_default_bus: Optional[AgentBus] = None


def get_default_bus() -> AgentBus:
    """Get the default AgentBus instance."""
    global _default_bus
    if _default_bus is None:
        _default_bus = AgentBus()
    return _default_bus


def set_default_bus(bus: AgentBus) -> None:
    """Set the default AgentBus instance."""
    global _default_bus
    _default_bus = bus
