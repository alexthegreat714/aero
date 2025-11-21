"""
Ollama client for LLM integration.

Connects to local Ollama instance for Gemma 3 and DeepCoder models.
"""

import json
import logging
import requests
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ChatMessage:
    """A single chat message."""
    role: str  # "system", "user", "assistant"
    content: str

    def to_dict(self) -> Dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass
class LLMResponse:
    """Response from LLM."""
    content: str
    model: str
    done: bool = True
    total_duration: Optional[int] = None
    eval_count: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "model": self.model,
            "done": self.done,
            "total_duration": self.total_duration,
            "eval_count": self.eval_count,
        }


class OllamaClient:
    """
    Client for Ollama API.

    Supports both streaming and non-streaming chat completions.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        timeout: float = 120.0,
    ):
        """
        Initialize Ollama client.

        Args:
            base_url: Ollama server URL
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _post(self, endpoint: str, data: Dict[str, Any]) -> requests.Response:
        """Make POST request to Ollama."""
        url = f"{self.base_url}{endpoint}"
        return requests.post(url, json=data, timeout=self.timeout)

    def _post_stream(self, endpoint: str, data: Dict[str, Any]) -> Generator[Dict[str, Any], None, None]:
        """Make streaming POST request to Ollama."""
        url = f"{self.base_url}{endpoint}"
        with requests.post(url, json=data, stream=True, timeout=self.timeout) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    yield json.loads(line)

    def chat(
        self,
        model: str,
        messages: List[ChatMessage],
        system: Optional[str] = None,
        stream: bool = False,
        options: Optional[Dict[str, Any]] = None,
    ) -> LLMResponse | Generator[str, None, None]:
        """
        Send chat completion request.

        Args:
            model: Model name (e.g., "gemma3", "deepcoder")
            messages: List of chat messages
            system: Optional system prompt
            stream: Whether to stream response
            options: Model options (temperature, etc.)

        Returns:
            LLMResponse or generator of content chunks if streaming
        """
        msg_dicts = [m.to_dict() for m in messages]

        # Add system message if provided
        if system:
            msg_dicts.insert(0, {"role": "system", "content": system})

        data = {
            "model": model,
            "messages": msg_dicts,
            "stream": stream,
        }

        if options:
            data["options"] = options

        if stream:
            return self._stream_chat(data)
        else:
            response = self._post("/api/chat", data)
            response.raise_for_status()
            result = response.json()

            return LLMResponse(
                content=result.get("message", {}).get("content", ""),
                model=model,
                done=result.get("done", True),
                total_duration=result.get("total_duration"),
                eval_count=result.get("eval_count"),
            )

    def _stream_chat(self, data: Dict[str, Any]) -> Generator[str, None, None]:
        """Stream chat response."""
        for chunk in self._post_stream("/api/chat", data):
            content = chunk.get("message", {}).get("content", "")
            if content:
                yield content

    def generate(
        self,
        model: str,
        prompt: str,
        system: Optional[str] = None,
        stream: bool = False,
        options: Optional[Dict[str, Any]] = None,
    ) -> LLMResponse | Generator[str, None, None]:
        """
        Send generation request (non-chat).

        Args:
            model: Model name
            prompt: Input prompt
            system: Optional system prompt
            stream: Whether to stream
            options: Model options

        Returns:
            LLMResponse or generator if streaming
        """
        data = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
        }

        if system:
            data["system"] = system
        if options:
            data["options"] = options

        if stream:
            return self._stream_generate(data)
        else:
            response = self._post("/api/generate", data)
            response.raise_for_status()
            result = response.json()

            return LLMResponse(
                content=result.get("response", ""),
                model=model,
                done=result.get("done", True),
                total_duration=result.get("total_duration"),
                eval_count=result.get("eval_count"),
            )

    def _stream_generate(self, data: Dict[str, Any]) -> Generator[str, None, None]:
        """Stream generation response."""
        for chunk in self._post_stream("/api/generate", data):
            content = chunk.get("response", "")
            if content:
                yield content

    def list_models(self) -> List[Dict[str, Any]]:
        """List available models."""
        response = requests.get(f"{self.base_url}/api/tags", timeout=self.timeout)
        response.raise_for_status()
        return response.json().get("models", [])

    def is_available(self) -> bool:
        """Check if Ollama is available."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5.0)
            return response.status_code == 200
        except Exception:
            return False

    def has_model(self, model: str) -> bool:
        """Check if a specific model is available."""
        models = self.list_models()
        model_names = [m.get("name", "").split(":")[0] for m in models]
        return model in model_names or any(model in name for name in model_names)


# Default client instance
_default_client: Optional[OllamaClient] = None


def get_ollama_client() -> OllamaClient:
    """Get default Ollama client."""
    global _default_client
    if _default_client is None:
        _default_client = OllamaClient()
    return _default_client


def set_ollama_client(client: OllamaClient) -> None:
    """Set default Ollama client."""
    global _default_client
    _default_client = client
