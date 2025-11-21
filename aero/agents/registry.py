"""
Agent registry for managing registered agents.

Provides a centralized registry for tracking agents
and their capabilities.
"""

import logging
from typing import Dict, List, Optional

from aero.agents.base_agent_api import BaseAgentAPI

logger = logging.getLogger(__name__)


class AgentRegistry:
    """
    Registry for managing agents in the system.

    Keeps track of all registered agents and provides
    methods to query their capabilities.

    Example:
        registry = AgentRegistry()

        # Register an agent
        registry.register(my_agent)

        # Get agent by name
        agent = registry.get("MyAgent")

        # List all agents
        for desc in registry.list_agents():
            print(desc["name"], desc["capabilities"])
    """

    def __init__(self):
        """Initialize the agent registry."""
        self._agents: Dict[str, BaseAgentAPI] = {}
        logger.info("AgentRegistry initialized")

    def register(self, agent: BaseAgentAPI) -> None:
        """
        Register an agent.

        Args:
            agent: Agent instance to register
        """
        if agent.name in self._agents:
            logger.warning(f"Overwriting existing agent: {agent.name}")

        self._agents[agent.name] = agent
        logger.info(f"Registered agent: {agent.name}")

    def unregister(self, name: str) -> bool:
        """
        Unregister an agent.

        Args:
            name: Name of the agent to unregister

        Returns:
            True if agent was removed, False if not found
        """
        if name in self._agents:
            del self._agents[name]
            logger.info(f"Unregistered agent: {name}")
            return True

        logger.warning(f"Agent not found for unregister: {name}")
        return False

    def get(self, name: str) -> Optional[BaseAgentAPI]:
        """
        Get an agent by name.

        Args:
            name: Agent name

        Returns:
            Agent instance or None if not found
        """
        return self._agents.get(name)

    def has(self, name: str) -> bool:
        """Check if an agent is registered."""
        return name in self._agents

    def list_agents(self) -> List[dict]:
        """
        List all registered agents with their descriptions.

        Returns:
            List of agent description dictionaries
        """
        return [agent.describe() for agent in self._agents.values()]

    def list_agent_names(self) -> List[str]:
        """
        List all registered agent names.

        Returns:
            List of agent names
        """
        return list(self._agents.keys())

    def get_all(self) -> Dict[str, BaseAgentAPI]:
        """
        Get all registered agents.

        Returns:
            Dictionary mapping names to agent instances
        """
        return dict(self._agents)

    def count(self) -> int:
        """Get the number of registered agents."""
        return len(self._agents)

    def __len__(self) -> int:
        """Get the number of registered agents."""
        return len(self._agents)

    def __contains__(self, name: str) -> bool:
        """Check if an agent is registered."""
        return name in self._agents

    def find_by_capability(self, capability: str) -> List[BaseAgentAPI]:
        """
        Find agents that support a specific capability.

        Args:
            capability: Capability/action name to search for

        Returns:
            List of agents that support the capability
        """
        matching = []
        for agent in self._agents.values():
            if capability in agent.get_capabilities():
                matching.append(agent)
        return matching

    def get_summary(self) -> dict:
        """
        Get a summary of the registry.

        Returns:
            Summary dictionary with agent count and names
        """
        return {
            "agent_count": len(self._agents),
            "agent_names": list(self._agents.keys()),
            "agents": self.list_agents(),
        }


# Module-level default registry instance
_default_registry: Optional[AgentRegistry] = None


def get_default_registry() -> AgentRegistry:
    """Get the default AgentRegistry instance."""
    global _default_registry
    if _default_registry is None:
        _default_registry = AgentRegistry()
    return _default_registry


def set_default_registry(registry: AgentRegistry) -> None:
    """Set the default AgentRegistry instance."""
    global _default_registry
    _default_registry = registry
