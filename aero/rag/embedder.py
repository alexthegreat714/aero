"""
Embedding generation for Aero Agent RAG system.

Provides text embedding using various backends.
"""

import logging
import hashlib
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class BaseEmbedder(ABC):
    """
    Base class for text embedding models.

    Subclasses must implement the embed() method.
    """

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

    @abstractmethod
    def embed(self, text: str) -> np.ndarray:
        """
        Generate embedding for text.

        Args:
            text: Input text

        Returns:
            Embedding vector as numpy array
        """
        raise NotImplementedError

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of input texts

        Returns:
            List of embedding vectors
        """
        return [self.embed(text) for text in texts]


class Embedder(BaseEmbedder):
    """
    Default local embedder using simple hash-based embeddings.

    This is a placeholder implementation. For production use, consider:
    - sentence-transformers (local)
    - OpenAI embeddings (API)
    - Cohere embeddings (API)

    Example:
        embedder = Embedder()
        vector = embedder.embed("Hello world")
    """

    def __init__(self, dimension: int = 384):
        """
        Initialize the embedder.

        Args:
            dimension: Embedding dimension
        """
        self._dimension = dimension
        self._cache: dict[str, np.ndarray] = {}
        logger.info(f"Initialized local embedder with dimension {dimension}")

    @property
    def name(self) -> str:
        return "local_hash_embedder"

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, text: str) -> np.ndarray:
        """
        Generate a hash-based embedding for text.

        This creates deterministic embeddings based on text content.
        Useful for testing but not semantically meaningful.

        Args:
            text: Input text

        Returns:
            Normalized embedding vector
        """
        # Check cache
        cache_key = text[:100]  # Use first 100 chars as key
        if cache_key in self._cache:
            return self._cache[cache_key].copy()

        # Normalize text
        text = text.lower().strip()

        # Create multiple hashes for more variance
        embeddings = []
        for i in range(self._dimension // 32 + 1):
            hash_input = f"{text}_{i}"
            hash_bytes = hashlib.sha256(hash_input.encode()).digest()
            chunk = np.frombuffer(hash_bytes, dtype=np.uint8).astype(np.float32)
            embeddings.append(chunk)

        # Concatenate and truncate
        embedding = np.concatenate(embeddings)[:self._dimension]

        # Center around 0
        embedding = embedding - 128.0

        # Normalize to unit length
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        # Cache result
        self._cache[cache_key] = embedding

        return embedding.copy()

    def similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two texts.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Cosine similarity score
        """
        emb1 = self.embed(text1)
        emb2 = self.embed(text2)
        return float(np.dot(emb1, emb2))

    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self._cache.clear()


class SentenceTransformerEmbedder(BaseEmbedder):
    """
    Sentence Transformer based embedder.

    Requires sentence-transformers package to be installed.
    This is a stub that will raise ImportError if the package is not available.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the sentence transformer embedder.

        Args:
            model_name: Name of the sentence-transformers model
        """
        self._model_name = model_name
        self._model = None
        self._dimension_value: Optional[int] = None

        # Try to load the model
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(model_name)
            self._dimension_value = self._model.get_sentence_embedding_dimension()
            logger.info(f"Loaded sentence transformer model: {model_name}")
        except ImportError:
            logger.warning(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )

    @property
    def name(self) -> str:
        return f"sentence_transformer_{self._model_name}"

    @property
    def dimension(self) -> int:
        return self._dimension_value or 384

    def embed(self, text: str) -> np.ndarray:
        """
        Generate embedding using sentence transformer.

        Args:
            text: Input text

        Returns:
            Embedding vector

        Raises:
            RuntimeError: If model not loaded
        """
        if self._model is None:
            raise RuntimeError(
                "Sentence transformer model not loaded. "
                "Install sentence-transformers package."
            )

        embedding = self._model.encode(text, convert_to_numpy=True)
        return embedding

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """
        Generate embeddings for multiple texts efficiently.

        Args:
            texts: List of input texts

        Returns:
            List of embedding vectors
        """
        if self._model is None:
            raise RuntimeError("Sentence transformer model not loaded.")

        embeddings = self._model.encode(texts, convert_to_numpy=True)
        return list(embeddings)


def get_default_embedder() -> Embedder:
    """Get the default embedder instance."""
    return Embedder()
