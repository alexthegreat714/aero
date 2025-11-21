"""
RAG document store for Aero Agent.

Provides vector storage and retrieval for documents.
"""

import json
import logging
import hashlib
from pathlib import Path
from typing import Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """Represents a document in the RAG store."""

    id: str
    content: str
    metadata: dict = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        """Convert document to dictionary (without embedding)."""
        return {
            "id": self.id,
            "content": self.content,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class SearchResult:
    """Represents a search result."""

    document: Document
    score: float
    rank: int


class RAGStore:
    """
    Simple in-memory RAG document store.

    Provides:
    - Document storage with embeddings
    - Similarity search
    - Persistence to disk

    This is a placeholder implementation. In production, consider using
    vector databases like ChromaDB, Pinecone, or FAISS.

    Example:
        store = RAGStore()
        store.add_document("doc1", "Hello world", {"source": "test"})
        results = store.search("hello", top_k=5)
    """

    def __init__(
        self,
        store_path: Optional[str] = None,
        embedding_dim: int = 384,
    ):
        """
        Initialize the RAG store.

        Args:
            store_path: Path to persist the store (optional)
            embedding_dim: Dimension of embeddings
        """
        self.store_path = Path(store_path) if store_path else None
        self.embedding_dim = embedding_dim
        self._documents: dict[str, Document] = {}
        self._embeddings: dict[str, np.ndarray] = {}

        # Load existing store if path provided
        if self.store_path and self.store_path.exists():
            self._load()

        logger.info(f"RAGStore initialized with {len(self._documents)} documents")

    def add_document(
        self,
        content: str,
        metadata: Optional[dict] = None,
        doc_id: Optional[str] = None,
        embedding: Optional[np.ndarray] = None,
    ) -> str:
        """
        Add a document to the store.

        Args:
            content: Document text content
            metadata: Optional metadata dictionary
            doc_id: Optional document ID (auto-generated if not provided)
            embedding: Optional pre-computed embedding

        Returns:
            Document ID
        """
        # Generate ID if not provided
        if doc_id is None:
            doc_id = self._generate_id(content)

        # Create document
        doc = Document(
            id=doc_id,
            content=content,
            metadata=metadata or {},
            embedding=embedding,
        )

        # Generate placeholder embedding if not provided
        if embedding is None:
            doc.embedding = self._generate_placeholder_embedding(content)
        else:
            doc.embedding = embedding

        # Store document
        self._documents[doc_id] = doc
        if doc.embedding is not None:
            self._embeddings[doc_id] = doc.embedding

        logger.debug(f"Added document: {doc_id}")

        # Auto-save if store path is set
        if self.store_path:
            self._save()

        return doc_id

    def get_document(self, doc_id: str) -> Optional[Document]:
        """
        Get a document by ID.

        Args:
            doc_id: Document ID

        Returns:
            Document or None if not found
        """
        return self._documents.get(doc_id)

    def delete_document(self, doc_id: str) -> bool:
        """
        Delete a document from the store.

        Args:
            doc_id: Document ID

        Returns:
            True if deleted, False if not found
        """
        if doc_id in self._documents:
            del self._documents[doc_id]
            if doc_id in self._embeddings:
                del self._embeddings[doc_id]

            if self.store_path:
                self._save()

            logger.debug(f"Deleted document: {doc_id}")
            return True

        return False

    def search(
        self,
        query: str,
        top_k: int = 5,
        threshold: float = 0.0,
        query_embedding: Optional[np.ndarray] = None,
    ) -> list[SearchResult]:
        """
        Search for similar documents.

        Args:
            query: Search query text
            top_k: Number of results to return
            threshold: Minimum similarity score
            query_embedding: Optional pre-computed query embedding

        Returns:
            List of SearchResult objects
        """
        if not self._documents:
            return []

        # Generate query embedding
        if query_embedding is None:
            query_embedding = self._generate_placeholder_embedding(query)

        # Calculate similarities
        scores: list[tuple[str, float]] = []

        for doc_id, doc_embedding in self._embeddings.items():
            similarity = self._cosine_similarity(query_embedding, doc_embedding)
            if similarity >= threshold:
                scores.append((doc_id, similarity))

        # Sort by similarity (descending)
        scores.sort(key=lambda x: x[1], reverse=True)

        # Build results
        results = []
        for rank, (doc_id, score) in enumerate(scores[:top_k], start=1):
            doc = self._documents[doc_id]
            results.append(SearchResult(document=doc, score=score, rank=rank))

        return results

    def list_docs(self) -> list[dict]:
        """
        List all documents in the store.

        Returns:
            List of document dictionaries (without embeddings)
        """
        return [doc.to_dict() for doc in self._documents.values()]

    def count(self) -> int:
        """Get the number of documents in the store."""
        return len(self._documents)

    def clear(self) -> None:
        """Clear all documents from the store."""
        self._documents.clear()
        self._embeddings.clear()

        if self.store_path:
            self._save()

        logger.info("RAGStore cleared")

    # -------------------------------------------------------------------------
    # Private Methods
    # -------------------------------------------------------------------------

    def _generate_id(self, content: str) -> str:
        """Generate a unique ID for content."""
        hash_obj = hashlib.sha256(content.encode())
        return hash_obj.hexdigest()[:16]

    def _generate_placeholder_embedding(self, text: str) -> np.ndarray:
        """
        Generate a placeholder embedding for text.

        This is a simple hash-based embedding for demonstration.
        In production, use proper embedding models.
        """
        # Simple deterministic embedding based on text hash
        hash_bytes = hashlib.sha256(text.encode()).digest()

        # Convert to float array
        embedding = np.frombuffer(hash_bytes, dtype=np.uint8).astype(np.float32)

        # Pad or truncate to embedding_dim
        if len(embedding) < self.embedding_dim:
            embedding = np.pad(embedding, (0, self.embedding_dim - len(embedding)))
        else:
            embedding = embedding[:self.embedding_dim]

        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return float(dot_product / (norm_a * norm_b))

    def _save(self) -> None:
        """Save the store to disk."""
        if not self.store_path:
            return

        self.store_path.parent.mkdir(parents=True, exist_ok=True)

        # Save documents (without embeddings in JSON)
        docs_data = {
            doc_id: doc.to_dict()
            for doc_id, doc in self._documents.items()
        }

        docs_file = self.store_path / "documents.json"
        with open(docs_file, "w", encoding="utf-8") as f:
            json.dump(docs_data, f, indent=2)

        # Save embeddings as numpy array
        if self._embeddings:
            embeddings_file = self.store_path / "embeddings.npz"
            np.savez_compressed(embeddings_file, **self._embeddings)

        logger.debug(f"Saved store to {self.store_path}")

    def _load(self) -> None:
        """Load the store from disk."""
        if not self.store_path or not self.store_path.exists():
            return

        # Load documents
        docs_file = self.store_path / "documents.json"
        if docs_file.exists():
            with open(docs_file, "r", encoding="utf-8") as f:
                docs_data = json.load(f)

            for doc_id, doc_dict in docs_data.items():
                doc = Document(
                    id=doc_dict["id"],
                    content=doc_dict["content"],
                    metadata=doc_dict.get("metadata", {}),
                    created_at=datetime.fromisoformat(doc_dict["created_at"]),
                )
                self._documents[doc_id] = doc

        # Load embeddings
        embeddings_file = self.store_path / "embeddings.npz"
        if embeddings_file.exists():
            data = np.load(embeddings_file)
            for doc_id in data.files:
                self._embeddings[doc_id] = data[doc_id]
                if doc_id in self._documents:
                    self._documents[doc_id].embedding = data[doc_id]

        logger.debug(f"Loaded store from {self.store_path}")
