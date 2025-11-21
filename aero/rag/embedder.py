"""
Embedding generation for Aero Agent RAG system.

Provides text embedding using various backends:
- Local llama.cpp via Ollama API
- SentenceTransformers (local)
- Fallback hash-based embeddings

All backends work offline without cloud API dependencies.
"""

import logging
import hashlib
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import requests

logger = logging.getLogger(__name__)


class BaseEmbedder(ABC):
    """Base class for text embedding models."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the embedder."""
        raise NotImplementedError

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Dimension of the embedding vectors."""
        raise NotImplementedError

    @property
    def is_available(self) -> bool:
        """Check if the embedder is available."""
        return True

    @abstractmethod
    def embed(self, text: str) -> np.ndarray:
        """Generate embedding for text."""
        raise NotImplementedError

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Generate embeddings for multiple texts."""
        return [self.embed(text) for text in texts]


class OllamaEmbedder(BaseEmbedder):
    """Embedding using Ollama API (llama.cpp backend)."""

    def __init__(
        self,
        model: str = "nomic-embed-text",
        host: str = "http://localhost:11434",
        timeout: int = 30,
    ):
        self._model = model
        self._host = host.rstrip("/")
        self._timeout = timeout
        self._dimension_value: Optional[int] = None
        self._available = self._check_availability()

        if self._available:
            logger.info(f"Ollama embedder initialized: {model}")

    def _check_availability(self) -> bool:
        try:
            response = requests.get(f"{self._host}/api/tags", timeout=5)
            if response.status_code != 200:
                return False
            data = response.json()
            models = [m.get("name", "").split(":")[0] for m in data.get("models", [])]
            return self._model in models or any(self._model in m for m in models)
        except Exception:
            return False

    @property
    def name(self) -> str:
        return f"ollama_{self._model}"

    @property
    def dimension(self) -> int:
        if self._dimension_value is None:
            try:
                test_emb = self._embed_single("test")
                self._dimension_value = len(test_emb)
            except Exception:
                self._dimension_value = 768
        return self._dimension_value

    @property
    def is_available(self) -> bool:
        return self._available

    def _embed_single(self, text: str) -> np.ndarray:
        response = requests.post(
            f"{self._host}/api/embeddings",
            json={"model": self._model, "prompt": text},
            timeout=self._timeout,
        )
        if response.status_code != 200:
            raise RuntimeError(f"Ollama API error: {response.status_code}")
        return np.array(response.json().get("embedding", []), dtype=np.float32)

    def embed(self, text: str) -> np.ndarray:
        if not self._available:
            raise RuntimeError(f"Ollama not available (model: {self._model})")
        return self._embed_single(text)


class SentenceTransformerEmbedder(BaseEmbedder):
    """SentenceTransformers-based embedder with GPU support."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self._model_name = model_name
        self._model = None
        self._dimension_value: Optional[int] = None
        self._available = False

        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(model_name)
            self._dimension_value = self._model.get_sentence_embedding_dimension()
            self._available = True
            logger.info(f"SentenceTransformer loaded: {model_name} (dim={self._dimension_value})")
        except ImportError:
            logger.warning("sentence-transformers not installed")
        except Exception as e:
            logger.warning(f"Failed to load SentenceTransformer: {e}")

    @property
    def name(self) -> str:
        return f"sentence_transformer_{self._model_name}"

    @property
    def dimension(self) -> int:
        return self._dimension_value or 384

    @property
    def is_available(self) -> bool:
        return self._available

    def embed(self, text: str) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("SentenceTransformer not loaded")
        return self._model.encode(text, convert_to_numpy=True).astype(np.float32)

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        if self._model is None:
            raise RuntimeError("SentenceTransformer not loaded")
        embeddings = self._model.encode(texts, convert_to_numpy=True)
        return [emb.astype(np.float32) for emb in embeddings]


class HashEmbedder(BaseEmbedder):
    """Fallback hash-based embedder (always available)."""

    def __init__(self, dimension: int = 384):
        self._dimension = dimension
        self._cache: dict[str, np.ndarray] = {}
        logger.info(f"Hash embedder initialized (dim={dimension})")

    @property
    def name(self) -> str:
        return "hash_embedder"

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, text: str) -> np.ndarray:
        cache_key = text[:100]
        if cache_key in self._cache:
            return self._cache[cache_key].copy()

        text_norm = text.lower().strip()
        embeddings = []
        for i in range(self._dimension // 32 + 1):
            hash_bytes = hashlib.sha256(f"{text_norm}_{i}".encode()).digest()
            chunk = np.frombuffer(hash_bytes, dtype=np.uint8).astype(np.float32)
            embeddings.append(chunk)

        embedding = np.concatenate(embeddings)[:self._dimension] - 128.0
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        self._cache[cache_key] = embedding
        return embedding.copy()

    def similarity(self, text1: str, text2: str) -> float:
        return float(np.dot(self.embed(text1), self.embed(text2)))

    def clear_cache(self) -> None:
        self._cache.clear()


# Backward compatibility
Embedder = HashEmbedder


class AutoEmbedder(BaseEmbedder):
    """Automatic embedder selecting best available backend."""

    def __init__(
        self,
        prefer_ollama: bool = True,
        ollama_model: str = "nomic-embed-text",
        st_model: str = "all-MiniLM-L6-v2",
    ):
        self._backend: Optional[BaseEmbedder] = None
        self._backend_name = "none"

        if prefer_ollama:
            ollama = OllamaEmbedder(model=ollama_model)
            if ollama.is_available:
                self._backend = ollama
                self._backend_name = "ollama"
                logger.info(f"AutoEmbedder using Ollama ({ollama_model})")
                return

        st = SentenceTransformerEmbedder(model_name=st_model)
        if st.is_available:
            self._backend = st
            self._backend_name = "sentence_transformers"
            logger.info(f"AutoEmbedder using SentenceTransformers ({st_model})")
            return

        if not prefer_ollama:
            ollama = OllamaEmbedder(model=ollama_model)
            if ollama.is_available:
                self._backend = ollama
                self._backend_name = "ollama"
                return

        self._backend = HashEmbedder()
        self._backend_name = "hash_fallback"
        logger.warning("AutoEmbedder using hash fallback")

    @property
    def name(self) -> str:
        return f"auto_{self._backend_name}"

    @property
    def dimension(self) -> int:
        return self._backend.dimension if self._backend else 384

    @property
    def is_available(self) -> bool:
        return self._backend is not None

    @property
    def backend_name(self) -> str:
        return self._backend_name

    def embed(self, text: str) -> np.ndarray:
        if self._backend is None:
            raise RuntimeError("No embedding backend available")
        return self._backend.embed(text)

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        if self._backend is None:
            raise RuntimeError("No embedding backend available")
        return self._backend.embed_batch(texts)


_default_embedder: Optional[BaseEmbedder] = None


def get_embedder(backend: Optional[str] = None, **kwargs) -> BaseEmbedder:
    """Get an embedder instance."""
    if backend == "ollama":
        return OllamaEmbedder(**kwargs)
    elif backend == "sentence_transformers":
        return SentenceTransformerEmbedder(**kwargs)
    elif backend == "hash":
        return HashEmbedder(**kwargs)
    return AutoEmbedder(**kwargs)


def get_default_embedder() -> BaseEmbedder:
    """Get cached default embedder."""
    global _default_embedder
    if _default_embedder is None:
        _default_embedder = AutoEmbedder()
    return _default_embedder


def embed_text(text: str) -> np.ndarray:
    """Embed text using default embedder."""
    return get_default_embedder().embed(text)


def embed_batch(texts: list[str]) -> list[np.ndarray]:
    """Embed multiple texts using default embedder."""
    return get_default_embedder().embed_batch(texts)


def detect_embedding_backends() -> dict:
    """Detect available embedding backends."""
    result = {"ollama": False, "sentence_transformers": False, "hash": True, "details": {}}

    try:
        ollama = OllamaEmbedder()
        result["ollama"] = ollama.is_available
        result["details"]["ollama"] = {"available": ollama.is_available, "model": "nomic-embed-text"}
    except Exception as e:
        result["details"]["ollama"] = {"available": False, "error": str(e)}

    try:
        st = SentenceTransformerEmbedder()
        result["sentence_transformers"] = st.is_available
        result["details"]["sentence_transformers"] = {
            "available": st.is_available,
            "model": "all-MiniLM-L6-v2",
            "dimension": st.dimension if st.is_available else None,
        }
    except Exception as e:
        result["details"]["sentence_transformers"] = {"available": False, "error": str(e)}

    return result
