"""
Agent message types for inter-agent communication.

Defines the canonical message format used for communication between
Aero and other agents (Sky, Aegis, Apollo, Congress, etc.).
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)


@dataclass
class AgentMessage:
    """
    Standard message format for inter-agent communication.

    Attributes:
        id: Unique message identifier (UUID)
        sender: Name of the sending agent
        recipient: Name of the receiving agent
        kind: Message type ("request", "response", "event")
        action: Action to perform or event name
        payload: Message data/parameters
        created: ISO timestamp of message creation
        correlation_id: Optional ID to link request/response chains
        metadata: Optional additional metadata
    """

    id: str
    sender: str
    recipient: str
    kind: str  # "request", "response", "event"
    action: str  # e.g., "run_simulation", "rag_query", "health_check"
    payload: Dict[str, Any]
    created: str
    correlation_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def new(
        sender: str,
        recipient: str,
        kind: str,
        action: str,
        payload: Dict[str, Any],
        correlation_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "AgentMessage":
        """
        Create a new AgentMessage with auto-generated ID and timestamp.

        Args:
            sender: Name of the sending agent
            recipient: Name of the receiving agent
            kind: Message type ("request", "response", "event")
            action: Action to perform or event name
            payload: Message data/parameters
            correlation_id: Optional ID to link request/response chains
            metadata: Optional additional metadata

        Returns:
            New AgentMessage instance
        """
        return AgentMessage(
            id=str(uuid4()),
            sender=sender,
            recipient=recipient,
            kind=kind,
            action=action,
            payload=payload,
            created=datetime.utcnow().isoformat() + "Z",
            correlation_id=correlation_id,
            metadata=metadata or {},
        )

    @staticmethod
    def response_to(
        original: "AgentMessage",
        sender: str,
        payload: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "AgentMessage":
        """
        Create a response message to an original request.

        Args:
            original: The original request message
            sender: Name of the responding agent
            payload: Response data
            metadata: Optional additional metadata

        Returns:
            New AgentMessage as a response
        """
        return AgentMessage.new(
            sender=sender,
            recipient=original.sender,
            kind="response",
            action=original.action,
            payload=payload,
            correlation_id=original.id,
            metadata=metadata,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary."""
        return {
            "id": self.id,
            "sender": self.sender,
            "recipient": self.recipient,
            "kind": self.kind,
            "action": self.action,
            "payload": self.payload,
            "created": self.created,
            "correlation_id": self.correlation_id,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentMessage":
        """Create message from dictionary."""
        return cls(
            id=data.get("id", str(uuid4())),
            sender=data.get("sender", "unknown"),
            recipient=data.get("recipient", "unknown"),
            kind=data.get("kind", "request"),
            action=data.get("action", "unknown"),
            payload=data.get("payload", {}),
            created=data.get("created", datetime.utcnow().isoformat() + "Z"),
            correlation_id=data.get("correlation_id"),
            metadata=data.get("metadata", {}),
        )

    def is_request(self) -> bool:
        """Check if this is a request message."""
        return self.kind == "request"

    def is_response(self) -> bool:
        """Check if this is a response message."""
        return self.kind == "response"

    def is_event(self) -> bool:
        """Check if this is an event message."""
        return self.kind == "event"


@dataclass
class AgentMessageLogEntry:
    """
    Log entry for agent messages.

    Tracks message details along with processing information.
    """

    message: AgentMessage
    timestamp: str
    processed: bool = False
    result_status: Optional[str] = None  # "ok", "error", None
    error_message: Optional[str] = None
    processing_time_ms: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert log entry to dictionary."""
        return {
            "message": self.message.to_dict(),
            "timestamp": self.timestamp,
            "processed": self.processed,
            "result_status": self.result_status,
            "error_message": self.error_message,
            "processing_time_ms": self.processing_time_ms,
        }


# Message kind constants
MESSAGE_KIND_REQUEST = "request"
MESSAGE_KIND_RESPONSE = "response"
MESSAGE_KIND_EVENT = "event"

# Common action constants
ACTION_HEALTH_CHECK = "health_check"
ACTION_HYPOTHESIS_LOOP = "hypothesis_loop"
ACTION_RUN_SIMULATION = "run_simulation"
ACTION_RAG_SEARCH = "rag_search"
ACTION_LIST_SURROGATES = "list_surrogates"
ACTION_CHECK_CONSTRAINTS = "check_constraints"
ACTION_LOOP_COMPLETED = "loop_completed"
ACTION_LOOP_STARTED = "loop_started"
