"""
Base agent API for inter-agent communication.

Defines the abstract base class that all agents must implement
to interface with the message bus.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from aero.agents.messages import AgentMessage

logger = logging.getLogger(__name__)


class BaseAgentAPI(ABC):
    """
    Abstract base class for agent APIs.

    All agents that want to interface with the AgentBus must
    inherit from this class and implement handle_message().

    Example:
        class MyAgent(BaseAgentAPI):
            def __init__(self):
                super().__init__(name="MyAgent")

            def handle_message(self, msg: AgentMessage) -> dict:
                if msg.action == "greet":
                    return {"status": "ok", "data": {"greeting": "Hello!"}}
                return {"status": "error", "message": "Unknown action"}

            def get_capabilities(self) -> List[str]:
                return ["greet", "farewell"]
    """

    def __init__(self, name: str, version: str = "1.0.0"):
        """
        Initialize the agent.

        Args:
            name: Unique name for this agent
            version: Agent version string
        """
        self.name = name
        self.version = version
        self._metadata: Dict[str, Any] = {}

        logger.debug(f"BaseAgentAPI initialized: {name} v{version}")

    @abstractmethod
    def handle_message(self, msg: AgentMessage) -> dict:
        """
        Process an incoming message and return a JSON-serializable dict.

        This method must be implemented by all agent subclasses.

        Args:
            msg: The incoming AgentMessage

        Returns:
            Dictionary with at least:
                - status: "ok" or "error"
                - data: result payload (optional)
                - message: human-readable message (optional)
        """
        pass

    def get_capabilities(self) -> List[str]:
        """
        Return list of action names this agent supports.

        Override this method to provide a list of supported actions.

        Returns:
            List of action name strings
        """
        return []

    def describe(self) -> dict:
        """
        Return a capability description for this agent.

        Returns:
            Dictionary with agent description including:
                - name: Agent name
                - version: Agent version
                - capabilities: List of supported actions
                - metadata: Additional agent metadata
        """
        return {
            "name": self.name,
            "version": self.version,
            "capabilities": self.get_capabilities(),
            "metadata": self._metadata,
        }

    def set_metadata(self, key: str, value: Any) -> None:
        """Set a metadata value."""
        self._metadata[key] = value

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get a metadata value."""
        return self._metadata.get(key, default)

    def _make_ok_response(
        self,
        data: Optional[Dict[str, Any]] = None,
        message: Optional[str] = None,
    ) -> dict:
        """
        Helper to create a successful response.

        Args:
            data: Response payload
            message: Optional human-readable message

        Returns:
            Response dictionary
        """
        response = {"status": "ok"}
        if data is not None:
            response["data"] = data
        if message is not None:
            response["message"] = message
        return response

    def _make_error_response(
        self,
        message: str,
        error_code: Optional[str] = None,
        data: Optional[Dict[str, Any]] = None,
    ) -> dict:
        """
        Helper to create an error response.

        Args:
            message: Error message
            error_code: Optional error code
            data: Optional additional error data

        Returns:
            Error response dictionary
        """
        response = {"status": "error", "message": message}
        if error_code is not None:
            response["error_code"] = error_code
        if data is not None:
            response["data"] = data
        return response

    def _unknown_action_response(self, action: str) -> dict:
        """
        Helper to create response for unknown action.

        Args:
            action: The unknown action name

        Returns:
            Error response dictionary
        """
        return self._make_error_response(
            message=f"Unknown action: {action}",
            error_code="UNKNOWN_ACTION",
            data={"available_actions": self.get_capabilities()},
        )


class EchoAgent(BaseAgentAPI):
    """
    Simple echo agent for testing purposes.

    Echoes back the payload it receives.
    """

    def __init__(self):
        super().__init__(name="Echo", version="1.0.0")

    def handle_message(self, msg: AgentMessage) -> dict:
        """Echo back the message payload."""
        return self._make_ok_response(
            data={
                "echoed_action": msg.action,
                "echoed_payload": msg.payload,
                "sender": msg.sender,
            },
            message="Message echoed successfully",
        )

    def get_capabilities(self) -> List[str]:
        return ["echo", "any"]
