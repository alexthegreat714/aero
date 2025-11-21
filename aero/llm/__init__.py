"""
LLM integration module for Aero.

Provides Ollama client and reasoning orchestration for Gemma 3 and DeepCoder.
"""

from aero.llm.ollama_client import (
    OllamaClient,
    ChatMessage,
    LLMResponse,
    get_ollama_client,
    set_ollama_client,
)

from aero.llm.reasoning import (
    ReasoningOrchestrator,
    ReasoningResult,
    ConversationTurn,
    get_orchestrator,
    set_orchestrator,
    AERO_SYSTEM_PROMPT,
)

__all__ = [
    # Client
    "OllamaClient",
    "ChatMessage",
    "LLMResponse",
    "get_ollama_client",
    "set_ollama_client",
    # Reasoning
    "ReasoningOrchestrator",
    "ReasoningResult",
    "ConversationTurn",
    "get_orchestrator",
    "set_orchestrator",
    "AERO_SYSTEM_PROMPT",
]
