"""
Multi-Agent Integration Layer for Aero.

This module provides inter-agent communication capabilities:
- Message definitions for agent communication
- Local message bus for routing messages
- Base agent API for implementing agents
- Aero agent wrapper for exposing Aero capabilities
- Agent registry for managing registered agents

Example:
    from aero.agents import (
        AgentMessage,
        AgentBus,
        AeroAgent,
        AgentRegistry,
    )

    # Create bus and registry
    bus = AgentBus()
    registry = AgentRegistry()

    # Create and register Aero agent
    aero = AeroAgent(loop=my_loop, rag_store=my_rag)
    registry.register(aero)
    bus.register_handler("Aero", aero.handle_message)

    # Send a message
    msg = AgentMessage.new(
        sender="Sky",
        recipient="Aero",
        kind="request",
        action="health_check",
        payload={},
    )
    response = bus.send(msg)
"""

# Message types
from aero.agents.messages import (
    AgentMessage,
    AgentMessageLogEntry,
    MESSAGE_KIND_REQUEST,
    MESSAGE_KIND_RESPONSE,
    MESSAGE_KIND_EVENT,
    ACTION_HEALTH_CHECK,
    ACTION_HYPOTHESIS_LOOP,
    ACTION_RUN_SIMULATION,
    ACTION_RAG_SEARCH,
    ACTION_LIST_SURROGATES,
    ACTION_CHECK_CONSTRAINTS,
    ACTION_LOOP_COMPLETED,
    ACTION_LOOP_STARTED,
)

# Message bus
from aero.agents.bus import (
    AgentBus,
    get_default_bus,
    set_default_bus,
)

# Base agent API
from aero.agents.base_agent_api import (
    BaseAgentAPI,
    EchoAgent,
)

# Aero agent implementation
from aero.agents.aero_agent import AeroAgent

# Agent registry
from aero.agents.registry import (
    AgentRegistry,
    get_default_registry,
    set_default_registry,
)

__all__ = [
    # Messages
    "AgentMessage",
    "AgentMessageLogEntry",
    "MESSAGE_KIND_REQUEST",
    "MESSAGE_KIND_RESPONSE",
    "MESSAGE_KIND_EVENT",
    "ACTION_HEALTH_CHECK",
    "ACTION_HYPOTHESIS_LOOP",
    "ACTION_RUN_SIMULATION",
    "ACTION_RAG_SEARCH",
    "ACTION_LIST_SURROGATES",
    "ACTION_CHECK_CONSTRAINTS",
    "ACTION_LOOP_COMPLETED",
    "ACTION_LOOP_STARTED",
    # Bus
    "AgentBus",
    "get_default_bus",
    "set_default_bus",
    # Base API
    "BaseAgentAPI",
    "EchoAgent",
    # Aero Agent
    "AeroAgent",
    # Registry
    "AgentRegistry",
    "get_default_registry",
    "set_default_registry",
]
