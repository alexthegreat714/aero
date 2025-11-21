"""
Tests for the aero.agents module.

Tests agent messages, message bus, agent registry, and AeroAgent.
"""

import pytest
from typing import Dict, Any
from datetime import datetime
from uuid import UUID


# =============================================================================
# AgentMessage Tests
# =============================================================================


class TestAgentMessage:
    """Tests for AgentMessage dataclass."""

    def test_message_new_creates_valid_id(self):
        """Test that new() creates a valid UUID."""
        from aero.agents.messages import AgentMessage

        msg = AgentMessage.new(
            sender="TestSender",
            recipient="TestRecipient",
            kind="request",
            action="test_action",
            payload={"key": "value"},
        )

        # Should be a valid UUID
        UUID(msg.id)  # Will raise if invalid
        assert len(msg.id) == 36  # Standard UUID string length

    def test_message_new_creates_timestamp(self):
        """Test that new() creates a proper ISO timestamp."""
        from aero.agents.messages import AgentMessage

        msg = AgentMessage.new(
            sender="TestSender",
            recipient="TestRecipient",
            kind="request",
            action="test_action",
            payload={},
        )

        # Should end with Z (UTC)
        assert msg.created.endswith("Z")

        # Should be parseable as ISO format
        timestamp = msg.created.rstrip("Z")
        datetime.fromisoformat(timestamp)

    def test_message_new_sets_fields(self):
        """Test that new() sets all fields correctly."""
        from aero.agents.messages import AgentMessage

        msg = AgentMessage.new(
            sender="Sky",
            recipient="Aero",
            kind="request",
            action="health_check",
            payload={"foo": "bar"},
            correlation_id="corr-123",
            metadata={"test": True},
        )

        assert msg.sender == "Sky"
        assert msg.recipient == "Aero"
        assert msg.kind == "request"
        assert msg.action == "health_check"
        assert msg.payload == {"foo": "bar"}
        assert msg.correlation_id == "corr-123"
        assert msg.metadata == {"test": True}

    def test_message_response_to(self):
        """Test creating a response to a request."""
        from aero.agents.messages import AgentMessage

        original = AgentMessage.new(
            sender="Sky",
            recipient="Aero",
            kind="request",
            action="health_check",
            payload={},
        )

        response = AgentMessage.response_to(
            original=original,
            sender="Aero",
            payload={"status": "ok"},
        )

        assert response.sender == "Aero"
        assert response.recipient == "Sky"  # Original sender
        assert response.kind == "response"
        assert response.action == "health_check"
        assert response.correlation_id == original.id
        assert response.payload == {"status": "ok"}

    def test_message_to_dict(self):
        """Test message serialization to dict."""
        from aero.agents.messages import AgentMessage

        msg = AgentMessage.new(
            sender="Sky",
            recipient="Aero",
            kind="request",
            action="test",
            payload={"x": 1},
        )

        d = msg.to_dict()

        assert "id" in d
        assert "sender" in d
        assert "recipient" in d
        assert "kind" in d
        assert "action" in d
        assert "payload" in d
        assert "created" in d
        assert "correlation_id" in d
        assert "metadata" in d

    def test_message_from_dict(self):
        """Test message deserialization from dict."""
        from aero.agents.messages import AgentMessage

        data = {
            "id": "test-id-123",
            "sender": "Sky",
            "recipient": "Aero",
            "kind": "request",
            "action": "test",
            "payload": {"x": 1},
            "created": "2024-01-01T00:00:00Z",
            "correlation_id": "corr-456",
            "metadata": {"key": "val"},
        }

        msg = AgentMessage.from_dict(data)

        assert msg.id == "test-id-123"
        assert msg.sender == "Sky"
        assert msg.recipient == "Aero"
        assert msg.action == "test"
        assert msg.correlation_id == "corr-456"

    def test_message_kind_helpers(self):
        """Test is_request(), is_response(), is_event() helpers."""
        from aero.agents.messages import AgentMessage

        request = AgentMessage.new(
            sender="A", recipient="B", kind="request", action="x", payload={}
        )
        response = AgentMessage.new(
            sender="A", recipient="B", kind="response", action="x", payload={}
        )
        event = AgentMessage.new(
            sender="A", recipient="B", kind="event", action="x", payload={}
        )

        assert request.is_request()
        assert not request.is_response()
        assert not request.is_event()

        assert response.is_response()
        assert not response.is_request()

        assert event.is_event()
        assert not event.is_request()


# =============================================================================
# AgentBus Tests
# =============================================================================


class TestAgentBus:
    """Tests for AgentBus message routing."""

    def test_bus_creation(self):
        """Test basic bus creation."""
        from aero.agents.bus import AgentBus

        bus = AgentBus()
        assert bus.enabled is True
        assert bus.log_limit == 1000
        assert len(bus.list_handlers()) == 0

    def test_bus_disabled(self):
        """Test bus when disabled."""
        from aero.agents.bus import AgentBus
        from aero.agents.messages import AgentMessage

        bus = AgentBus(enabled=False)

        msg = AgentMessage.new(
            sender="Sky", recipient="Aero", kind="request", action="test", payload={}
        )

        result = bus.send(msg)
        assert result["status"] == "skipped"
        assert "disabled" in result["message"].lower()

    def test_register_handler(self):
        """Test registering a handler."""
        from aero.agents.bus import AgentBus

        bus = AgentBus()

        def dummy_handler(msg):
            return {"status": "ok"}

        bus.register_handler("TestAgent", dummy_handler)

        assert bus.has_handler("TestAgent")
        assert "TestAgent" in bus.list_handlers()

    def test_unregister_handler(self):
        """Test unregistering a handler."""
        from aero.agents.bus import AgentBus

        bus = AgentBus()

        def dummy_handler(msg):
            return {"status": "ok"}

        bus.register_handler("TestAgent", dummy_handler)
        assert bus.has_handler("TestAgent")

        bus.unregister_handler("TestAgent")
        assert not bus.has_handler("TestAgent")

    def test_send_to_registered_handler(self):
        """Test sending message to registered handler."""
        from aero.agents.bus import AgentBus
        from aero.agents.messages import AgentMessage

        bus = AgentBus()

        received_messages = []

        def handler(msg):
            received_messages.append(msg)
            return {"status": "ok", "data": {"echoed": msg.action}}

        bus.register_handler("Echo", handler)

        msg = AgentMessage.new(
            sender="Caller",
            recipient="Echo",
            kind="request",
            action="test_action",
            payload={"key": "value"},
        )

        result = bus.send(msg)

        assert result["status"] == "ok"
        assert len(received_messages) == 1
        assert received_messages[0].action == "test_action"

    def test_send_to_unknown_recipient(self):
        """Test sending to unregistered recipient."""
        from aero.agents.bus import AgentBus
        from aero.agents.messages import AgentMessage

        bus = AgentBus()

        msg = AgentMessage.new(
            sender="Caller",
            recipient="Unknown",
            kind="request",
            action="test",
            payload={},
        )

        result = bus.send(msg)
        assert result["status"] == "error"
        assert "no handler" in result["message"].lower()

    def test_message_log(self):
        """Test that messages are logged."""
        from aero.agents.bus import AgentBus
        from aero.agents.messages import AgentMessage

        bus = AgentBus()

        def handler(msg):
            return {"status": "ok"}

        bus.register_handler("TestAgent", handler)

        # Send a few messages
        for i in range(5):
            msg = AgentMessage.new(
                sender="Caller",
                recipient="TestAgent",
                kind="request",
                action=f"action_{i}",
                payload={},
            )
            bus.send(msg)

        log = bus.get_log(limit=10)
        assert len(log) == 5

        # Check that entries have expected structure
        for entry in log:
            assert hasattr(entry, "message")
            assert hasattr(entry, "timestamp")
            assert hasattr(entry, "processed")

    def test_message_log_limit(self):
        """Test that log respects limit."""
        from aero.agents.bus import AgentBus
        from aero.agents.messages import AgentMessage

        bus = AgentBus(log_limit=3)

        def handler(msg):
            return {"status": "ok"}

        bus.register_handler("TestAgent", handler)

        # Send more messages than limit
        for i in range(10):
            msg = AgentMessage.new(
                sender="Caller",
                recipient="TestAgent",
                kind="request",
                action=f"action_{i}",
                payload={},
            )
            bus.send(msg)

        # Should only keep last 3
        log = bus.get_log(limit=100)
        assert len(log) <= 3

    def test_bus_stats(self):
        """Test getting bus statistics."""
        from aero.agents.bus import AgentBus
        from aero.agents.messages import AgentMessage

        bus = AgentBus()

        def handler(msg):
            return {"status": "ok"}

        bus.register_handler("TestAgent", handler)

        msg = AgentMessage.new(
            sender="Caller",
            recipient="TestAgent",
            kind="request",
            action="test",
            payload={},
        )
        bus.send(msg)

        stats = bus.get_stats()

        assert stats["enabled"] is True
        assert stats["handlers_registered"] == 1
        assert "TestAgent" in stats["handler_names"]
        assert stats["log_size"] == 1
        assert stats["messages_processed"] == 1


# =============================================================================
# AgentRegistry Tests
# =============================================================================


class TestAgentRegistry:
    """Tests for AgentRegistry."""

    def test_registry_creation(self):
        """Test basic registry creation."""
        from aero.agents.registry import AgentRegistry

        registry = AgentRegistry()
        assert registry.count() == 0
        assert len(registry) == 0

    def test_register_agent(self):
        """Test registering an agent."""
        from aero.agents.registry import AgentRegistry
        from aero.agents.base_agent_api import EchoAgent

        registry = AgentRegistry()
        agent = EchoAgent()

        registry.register(agent)

        assert registry.count() == 1
        assert registry.has("Echo")
        assert "Echo" in registry

    def test_unregister_agent(self):
        """Test unregistering an agent."""
        from aero.agents.registry import AgentRegistry
        from aero.agents.base_agent_api import EchoAgent

        registry = AgentRegistry()
        agent = EchoAgent()

        registry.register(agent)
        assert registry.has("Echo")

        registry.unregister("Echo")
        assert not registry.has("Echo")

    def test_get_agent(self):
        """Test getting an agent by name."""
        from aero.agents.registry import AgentRegistry
        from aero.agents.base_agent_api import EchoAgent

        registry = AgentRegistry()
        agent = EchoAgent()

        registry.register(agent)

        retrieved = registry.get("Echo")
        assert retrieved is agent

    def test_get_nonexistent_agent(self):
        """Test getting a non-existent agent returns None."""
        from aero.agents.registry import AgentRegistry

        registry = AgentRegistry()

        result = registry.get("NonExistent")
        assert result is None

    def test_list_agents(self):
        """Test listing all agents."""
        from aero.agents.registry import AgentRegistry
        from aero.agents.base_agent_api import EchoAgent

        registry = AgentRegistry()
        agent = EchoAgent()

        registry.register(agent)

        agents = registry.list_agents()
        assert len(agents) == 1
        assert agents[0]["name"] == "Echo"
        assert "capabilities" in agents[0]

    def test_find_by_capability(self):
        """Test finding agents by capability."""
        from aero.agents.registry import AgentRegistry
        from aero.agents.base_agent_api import EchoAgent

        registry = AgentRegistry()
        agent = EchoAgent()

        registry.register(agent)

        # EchoAgent has "echo" and "any" capabilities
        matching = registry.find_by_capability("echo")
        assert len(matching) == 1
        assert matching[0].name == "Echo"

        no_match = registry.find_by_capability("nonexistent")
        assert len(no_match) == 0


# =============================================================================
# BaseAgentAPI Tests
# =============================================================================


class TestBaseAgentAPI:
    """Tests for BaseAgentAPI."""

    def test_echo_agent_handle_message(self):
        """Test EchoAgent message handling."""
        from aero.agents.base_agent_api import EchoAgent
        from aero.agents.messages import AgentMessage

        agent = EchoAgent()

        msg = AgentMessage.new(
            sender="Tester",
            recipient="Echo",
            kind="request",
            action="test",
            payload={"foo": "bar"},
        )

        result = agent.handle_message(msg)

        assert result["status"] == "ok"
        assert result["data"]["echoed_action"] == "test"
        assert result["data"]["echoed_payload"] == {"foo": "bar"}
        assert result["data"]["sender"] == "Tester"

    def test_agent_describe(self):
        """Test agent description."""
        from aero.agents.base_agent_api import EchoAgent

        agent = EchoAgent()

        desc = agent.describe()

        assert desc["name"] == "Echo"
        assert desc["version"] == "1.0.0"
        assert "capabilities" in desc
        assert "echo" in desc["capabilities"]

    def test_agent_metadata(self):
        """Test setting and getting metadata."""
        from aero.agents.base_agent_api import EchoAgent

        agent = EchoAgent()

        agent.set_metadata("test_key", "test_value")
        assert agent.get_metadata("test_key") == "test_value"
        assert agent.get_metadata("nonexistent", "default") == "default"


# =============================================================================
# AeroAgent Tests
# =============================================================================


class TestAeroAgent:
    """Tests for AeroAgent implementation."""

    def test_aero_agent_creation(self):
        """Test creating AeroAgent."""
        from aero.agents.aero_agent import AeroAgent

        agent = AeroAgent()

        assert agent.name == "Aero"
        assert "health_check" in agent.get_capabilities()
        assert "hypothesis_loop" in agent.get_capabilities()

    def test_aero_agent_health_check(self):
        """Test AeroAgent health check action."""
        from aero.agents.aero_agent import AeroAgent
        from aero.agents.messages import AgentMessage

        agent = AeroAgent()

        msg = AgentMessage.new(
            sender="Sky",
            recipient="Aero",
            kind="request",
            action="health_check",
            payload={},
        )

        result = agent.handle_message(msg)

        assert result["status"] == "ok"
        assert "data" in result
        assert result["data"]["agent"] == "Aero"
        assert result["data"]["status"] == "healthy"
        assert "components" in result["data"]

    def test_aero_agent_describe(self):
        """Test AeroAgent describe action."""
        from aero.agents.aero_agent import AeroAgent
        from aero.agents.messages import AgentMessage

        agent = AeroAgent()

        msg = AgentMessage.new(
            sender="Sky",
            recipient="Aero",
            kind="request",
            action="describe",
            payload={},
        )

        result = agent.handle_message(msg)

        assert result["status"] == "ok"
        assert result["data"]["name"] == "Aero"

    def test_aero_agent_unknown_action(self):
        """Test AeroAgent with unknown action."""
        from aero.agents.aero_agent import AeroAgent
        from aero.agents.messages import AgentMessage

        agent = AeroAgent()

        msg = AgentMessage.new(
            sender="Sky",
            recipient="Aero",
            kind="request",
            action="unknown_action_xyz",
            payload={},
        )

        result = agent.handle_message(msg)

        assert result["status"] == "error"
        assert "unknown" in result["message"].lower()

    def test_aero_agent_hypothesis_loop_without_loop(self):
        """Test hypothesis_loop action without loop initialized."""
        from aero.agents.aero_agent import AeroAgent
        from aero.agents.messages import AgentMessage

        agent = AeroAgent()  # No loop provided

        msg = AgentMessage.new(
            sender="Sky",
            recipient="Aero",
            kind="request",
            action="hypothesis_loop",
            payload={"query": "Test query"},
        )

        result = agent.handle_message(msg)

        # Should return error since loop is not initialized
        assert result["status"] == "error"
        assert "not initialized" in result["message"].lower()

    def test_aero_agent_missing_query(self):
        """Test hypothesis_loop without query."""
        from aero.agents.aero_agent import AeroAgent
        from aero.agents.messages import AgentMessage

        agent = AeroAgent()

        msg = AgentMessage.new(
            sender="Sky",
            recipient="Aero",
            kind="request",
            action="hypothesis_loop",
            payload={},  # No query
        )

        result = agent.handle_message(msg)

        assert result["status"] == "error"
        assert "query" in result["message"].lower()


# =============================================================================
# Integration Tests
# =============================================================================


class TestAgentIntegration:
    """Integration tests for the agent system."""

    def test_full_message_flow(self):
        """Test complete message flow through bus and registry."""
        from aero.agents.bus import AgentBus
        from aero.agents.registry import AgentRegistry
        from aero.agents.aero_agent import AeroAgent
        from aero.agents.messages import AgentMessage

        # Setup
        bus = AgentBus()
        registry = AgentRegistry()
        agent = AeroAgent()

        registry.register(agent)
        bus.register_handler("Aero", agent.handle_message)

        # Send message
        msg = AgentMessage.new(
            sender="Sky",
            recipient="Aero",
            kind="request",
            action="health_check",
            payload={},
        )

        result = bus.send(msg)

        # Verify
        assert result["status"] == "ok"
        assert result["message_id"] == msg.id

        # Check log
        log = bus.get_log()
        assert len(log) == 1
        assert log[0].processed is True

    def test_module_exports(self):
        """Test that all expected items are exported from __init__."""
        import aero.agents as agents

        # Messages
        assert hasattr(agents, "AgentMessage")
        assert hasattr(agents, "AgentMessageLogEntry")
        assert hasattr(agents, "MESSAGE_KIND_REQUEST")
        assert hasattr(agents, "MESSAGE_KIND_RESPONSE")
        assert hasattr(agents, "MESSAGE_KIND_EVENT")

        # Bus
        assert hasattr(agents, "AgentBus")
        assert hasattr(agents, "get_default_bus")
        assert hasattr(agents, "set_default_bus")

        # Base API
        assert hasattr(agents, "BaseAgentAPI")
        assert hasattr(agents, "EchoAgent")

        # Aero Agent
        assert hasattr(agents, "AeroAgent")

        # Registry
        assert hasattr(agents, "AgentRegistry")
        assert hasattr(agents, "get_default_registry")
        assert hasattr(agents, "set_default_registry")

    def test_default_instances(self):
        """Test default bus and registry instances."""
        from aero.agents.bus import get_default_bus, set_default_bus, AgentBus
        from aero.agents.registry import get_default_registry, set_default_registry, AgentRegistry

        # Set up fresh instances
        bus = AgentBus()
        registry = AgentRegistry()

        set_default_bus(bus)
        set_default_registry(registry)

        assert get_default_bus() is bus
        assert get_default_registry() is registry
