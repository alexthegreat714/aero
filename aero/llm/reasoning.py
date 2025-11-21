"""
Reasoning orchestrator for Aero.

Manages conversation flow between Gemma 3 (main chat) and DeepCoder (deep reasoning).
Uses LLM to decide when deep reasoning is needed (not hardcoded triggers).
"""

import logging
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Generator, List, Optional, Tuple

from aero.llm.ollama_client import OllamaClient, ChatMessage, LLMResponse, get_ollama_client

logger = logging.getLogger(__name__)


# System prompts
AERO_SYSTEM_PROMPT = """You are Aero, an advanced aerodynamics research agent. You help users with:
- Scientific hypothesis generation and testing
- Aerodynamics simulations and analysis
- Document retrieval and research
- Data analysis and visualization

You have access to simulation engines, RAG document stores, and scientific reasoning capabilities.
Be helpful, precise, and scientific in your responses. When you need to perform deep analysis
or complex reasoning, indicate this clearly."""

REASONING_CHECK_PROMPT = """Analyze if this user message requires deep analytical reasoning.

Deep reasoning is needed when:
- Complex mathematical or scientific analysis is required
- Multi-step problem solving is needed
- Code generation or algorithm design is requested
- The question requires careful logical deduction
- Simulation parameters need to be determined
- Scientific hypothesis formulation is needed

Respond with JSON only:
{"needs_reasoning": true/false, "reason": "brief explanation", "complexity": 1-5}

User message: {message}"""

DEEPCODER_SYSTEM_PROMPT = """You are DeepCoder, an analytical reasoning engine. Your role is to:
- Break down complex problems into steps
- Perform deep logical analysis
- Generate algorithms and code when needed
- Analyze scientific data and formulate hypotheses
- Show your reasoning process clearly

Structure your response with clear sections:
## Analysis
## Reasoning Steps
## Conclusion/Solution

Be thorough but concise."""

SUMMARIZE_PROMPT = """Based on this deep analysis, provide a clear, user-friendly summary.
Keep the key insights but make it accessible. Don't repeat everything - distill the essence.

Deep Analysis:
{reasoning}

Provide a helpful summary for the user:"""


@dataclass
class ReasoningResult:
    """Result of a reasoning step."""
    content: str
    model: str
    reasoning_used: bool = False
    deep_reasoning: Optional[str] = None
    complexity: int = 1
    reasoning_reason: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "model": self.model,
            "reasoning_used": self.reasoning_used,
            "deep_reasoning": self.deep_reasoning,
            "complexity": self.complexity,
            "reasoning_reason": self.reasoning_reason,
            "timestamp": self.timestamp,
        }


@dataclass
class ConversationTurn:
    """A single conversation turn."""
    user_message: str
    assistant_response: str
    reasoning_result: Optional[ReasoningResult] = None
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    references: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_message": self.user_message,
            "assistant_response": self.assistant_response,
            "reasoning_result": self.reasoning_result.to_dict() if self.reasoning_result else None,
            "timestamp": self.timestamp,
            "references": self.references,
        }


class ReasoningOrchestrator:
    """
    Orchestrates conversation between user, Gemma 3, and DeepCoder.

    Flow:
    1. User sends message
    2. Check if deep reasoning is needed (using LLM, not triggers)
    3. If needed, invoke DeepCoder for analysis
    4. Gemma 3 responds, optionally summarizing DeepCoder's reasoning
    """

    def __init__(
        self,
        client: Optional[OllamaClient] = None,
        main_model: str = "gemma3",
        reasoning_model: str = "deepcoder",
        system_prompt: str = AERO_SYSTEM_PROMPT,
        reasoning_threshold: int = 2,  # Complexity threshold for deep reasoning
    ):
        """
        Initialize orchestrator.

        Args:
            client: Ollama client (uses default if None)
            main_model: Model for main conversation (Gemma 3)
            reasoning_model: Model for deep reasoning (DeepCoder)
            system_prompt: System prompt for main model
            reasoning_threshold: Minimum complexity to trigger deep reasoning
        """
        self.client = client or get_ollama_client()
        self.main_model = main_model
        self.reasoning_model = reasoning_model
        self.system_prompt = system_prompt
        self.reasoning_threshold = reasoning_threshold

        self.conversation_history: List[ChatMessage] = []
        self.turns: List[ConversationTurn] = []

    def _check_needs_reasoning(self, message: str) -> Tuple[bool, int, str]:
        """
        Use LLM to determine if deep reasoning is needed.

        Returns:
            Tuple of (needs_reasoning, complexity, reason)
        """
        try:
            prompt = REASONING_CHECK_PROMPT.format(message=message)
            response = self.client.generate(
                model=self.main_model,
                prompt=prompt,
                options={"temperature": 0.1},  # Low temp for consistent判断
            )

            # Parse JSON response
            content = response.content.strip()
            # Try to extract JSON from response
            if "{" in content:
                json_str = content[content.find("{"):content.rfind("}") + 1]
                result = json.loads(json_str)
                needs = result.get("needs_reasoning", False)
                complexity = result.get("complexity", 1)
                reason = result.get("reason", "")
                return needs and complexity >= self.reasoning_threshold, complexity, reason

        except Exception as e:
            logger.warning(f"Error checking reasoning need: {e}")

        return False, 1, ""

    def _invoke_deep_reasoning(self, message: str, context: str = "") -> str:
        """
        Invoke DeepCoder for deep analysis.

        Args:
            message: User message
            context: Additional context

        Returns:
            Deep reasoning output
        """
        prompt = f"""User Query: {message}

{f'Context: {context}' if context else ''}

Provide thorough analytical reasoning:"""

        response = self.client.generate(
            model=self.reasoning_model,
            prompt=prompt,
            system=DEEPCODER_SYSTEM_PROMPT,
            options={"temperature": 0.3},
        )

        return response.content

    def _summarize_reasoning(self, reasoning: str, original_message: str) -> str:
        """
        Have Gemma summarize DeepCoder's reasoning.

        Args:
            reasoning: Deep reasoning output
            original_message: Original user message

        Returns:
            User-friendly summary
        """
        messages = self.conversation_history.copy()
        messages.append(ChatMessage(
            role="user",
            content=SUMMARIZE_PROMPT.format(reasoning=reasoning)
        ))

        response = self.client.chat(
            model=self.main_model,
            messages=messages,
            system=self.system_prompt,
            options={"temperature": 0.7},
        )

        return response.content

    def chat(
        self,
        message: str,
        include_rag: bool = True,
        rag_query: Optional[str] = None,
    ) -> ReasoningResult:
        """
        Process a chat message with optional deep reasoning.

        Args:
            message: User message
            include_rag: Whether to include RAG references
            rag_query: Custom RAG query (uses message if None)

        Returns:
            ReasoningResult with response and reasoning info
        """
        references = []

        # Get RAG references if enabled
        if include_rag:
            try:
                from aero.rag.store import get_default_store
                store = get_default_store()
                results = store.search(rag_query or message, top_k=3)
                references = [
                    {"content": doc.content[:300], "score": float(score), "metadata": doc.metadata}
                    for doc, score in results
                ]
            except Exception as e:
                logger.debug(f"RAG lookup failed: {e}")

        # Check if deep reasoning is needed
        needs_reasoning, complexity, reason = self._check_needs_reasoning(message)
        deep_reasoning = None

        if needs_reasoning:
            logger.info(f"Invoking deep reasoning (complexity={complexity}, reason={reason})")

            # Build context from RAG
            context = ""
            if references:
                context = "Relevant documents:\n" + "\n".join(
                    r["content"] for r in references[:2]
                )

            # Get deep reasoning
            deep_reasoning = self._invoke_deep_reasoning(message, context)

            # Summarize for user
            response_content = self._summarize_reasoning(deep_reasoning, message)
        else:
            # Direct response from Gemma
            self.conversation_history.append(ChatMessage(role="user", content=message))

            response = self.client.chat(
                model=self.main_model,
                messages=self.conversation_history,
                system=self.system_prompt,
                options={"temperature": 0.7},
            )

            response_content = response.content

        # Update history
        self.conversation_history.append(ChatMessage(role="user", content=message))
        self.conversation_history.append(ChatMessage(role="assistant", content=response_content))

        result = ReasoningResult(
            content=response_content,
            model=self.main_model,
            reasoning_used=needs_reasoning,
            deep_reasoning=deep_reasoning,
            complexity=complexity,
            reasoning_reason=reason if needs_reasoning else None,
        )

        # Store turn
        turn = ConversationTurn(
            user_message=message,
            assistant_response=response_content,
            reasoning_result=result,
            references=references,
        )
        self.turns.append(turn)

        return result

    def chat_stream(
        self,
        message: str,
        include_rag: bool = True,
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Stream chat response with reasoning updates.

        Yields dicts with type and content:
        - {"type": "reasoning_check", "needs_reasoning": bool, "complexity": int}
        - {"type": "deep_reasoning", "content": str}  # If reasoning needed
        - {"type": "response", "content": str, "done": bool}
        - {"type": "references", "data": [...]}
        """
        references = []

        # Get RAG references
        if include_rag:
            try:
                from aero.rag.store import get_default_store
                store = get_default_store()
                results = store.search(message, top_k=3)
                references = [
                    {"content": doc.content[:300], "score": float(score), "metadata": doc.metadata}
                    for doc, score in results
                ]
                yield {"type": "references", "data": references}
            except Exception as e:
                logger.debug(f"RAG lookup failed: {e}")

        # Check reasoning need
        needs_reasoning, complexity, reason = self._check_needs_reasoning(message)
        yield {
            "type": "reasoning_check",
            "needs_reasoning": needs_reasoning,
            "complexity": complexity,
            "reason": reason,
        }

        if needs_reasoning:
            # Stream deep reasoning
            context = "\n".join(r["content"] for r in references[:2]) if references else ""
            prompt = f"User Query: {message}\n{f'Context: {context}' if context else ''}"

            yield {"type": "deep_reasoning_start", "content": ""}
            full_reasoning = ""

            for chunk in self.client.generate(
                model=self.reasoning_model,
                prompt=prompt,
                system=DEEPCODER_SYSTEM_PROMPT,
                stream=True,
                options={"temperature": 0.3},
            ):
                full_reasoning += chunk
                yield {"type": "deep_reasoning", "content": chunk, "done": False}

            yield {"type": "deep_reasoning", "content": "", "done": True}

            # Now summarize (streaming)
            messages = self.conversation_history + [
                ChatMessage(role="user", content=SUMMARIZE_PROMPT.format(reasoning=full_reasoning))
            ]

            full_response = ""
            for chunk in self.client.chat(
                model=self.main_model,
                messages=messages,
                system=self.system_prompt,
                stream=True,
                options={"temperature": 0.7},
            ):
                full_response += chunk
                yield {"type": "response", "content": chunk, "done": False}

            yield {"type": "response", "content": "", "done": True}

            # Update history
            self.conversation_history.append(ChatMessage(role="user", content=message))
            self.conversation_history.append(ChatMessage(role="assistant", content=full_response))

        else:
            # Direct streaming response
            self.conversation_history.append(ChatMessage(role="user", content=message))

            full_response = ""
            for chunk in self.client.chat(
                model=self.main_model,
                messages=self.conversation_history,
                system=self.system_prompt,
                stream=True,
                options={"temperature": 0.7},
            ):
                full_response += chunk
                yield {"type": "response", "content": chunk, "done": False}

            yield {"type": "response", "content": "", "done": True}

            self.conversation_history.append(ChatMessage(role="assistant", content=full_response))

    def get_history(self) -> List[Dict[str, Any]]:
        """Get conversation history."""
        return [turn.to_dict() for turn in self.turns]

    def clear_history(self) -> None:
        """Clear conversation history."""
        self.conversation_history.clear()
        self.turns.clear()


# Default orchestrator
_default_orchestrator: Optional[ReasoningOrchestrator] = None


def get_orchestrator() -> ReasoningOrchestrator:
    """Get default orchestrator."""
    global _default_orchestrator
    if _default_orchestrator is None:
        _default_orchestrator = ReasoningOrchestrator()
    return _default_orchestrator


def set_orchestrator(orch: ReasoningOrchestrator) -> None:
    """Set default orchestrator."""
    global _default_orchestrator
    _default_orchestrator = orch
