"""
Core infrastructure for Aero Agent.

Provides base classes, event system, and registry management.
"""

from aero.core.base_agent import BaseAgent
from aero.core.base_module import BaseModule
from aero.core.events import EventBus, Event
from aero.core.registry import Registry

__all__ = [
    "BaseAgent",
    "BaseModule",
    "EventBus",
    "Event",
    "Registry",
]
