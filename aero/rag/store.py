"""
RAG document store for Aero Agent.

Provides vector storage and retrieval with multiple backends:
- DuckDB (preferred, with VSS extension for vector search)
- SQLite with numpy-based vector search
- In-memory fallback

All backends work offline without cloud dependencies.
"""

import json
import logging
import hashlib
import sqlite3
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional, Union
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
    source: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None

    def to_dict(self) -> dict:
        """Convert document to dictionary (without embedding)."""
        return {
            "id": self.id,
            "content": self.content,
            "metadata": self.metadata,
            "source": self.source,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


@dataclass
class Chunk:
    """Represents a document chunk with embedding."""

    id: str
    document_id: str
    content: str
    chunk_index: int
    embedding: Optional[np.ndarray] = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert chunk to dictionary (without embedding)."""
        return {
            "id": self.id,
            "document_id": self.document_id,
            "content": self.content,
            "chunk_index": self.chunk_index,
            "metadata": self.metadata,
        }


@dataclass
class SearchResult:
    """Represents a search result."""

    chunk: Chunk
    document: Optional[Document]
    score: float
    rank: int


class BaseVectorStore(ABC):
    """Base class for vector stores."""

    @abstractmethod
    def add_document(
        self,
        content: str,
        metadata: Optional[dict] = None,
        doc_id: Optional[str] = None,
        source: Optional[str] = None,
        chunks: Optional[list[dict]] = None,
    ) -> str:
        """Add a document to the store."""
        raise NotImplementedError

    @abstractmethod
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        threshold: float = 0.0,
        filters: Optional[dict] = None,
    ) -> list[SearchResult]:
        """Search for similar documents."""
        raise NotImplementedError

    @abstractmethod
    def get_document(self, doc_id: str) -> Optional[Document]:
        """Get a document by ID."""
        raise NotImplementedError

    @abstractmethod
    def delete_document(self, doc_id: str) -> bool:
        """Delete a document."""
        raise NotImplementedError

    @abstractmethod
    def count(self) -> int:
        """Get document count."""
        raise NotImplementedError


class DuckDBVectorStore(BaseVectorStore):
    """DuckDB-based vector store with VSS extension support."""

    def __init__(
        self,
        db_path: Optional[str] = None,
        embedding_dim: int = 384,
    ):
        self._db_path = db_path or ":memory:"
        self._embedding_dim = embedding_dim
        self._conn = None
        self._vss_available = False

        self._connect()
        self._init_schema()

    def _connect(self) -> None:
        """Connect to DuckDB."""
        try:
            import duckdb
            self._conn = duckdb.connect(self._db_path)

            # Try to load VSS extension
            try:
                self._conn.execute("INSTALL vss; LOAD vss;")
                self._vss_available = True
                logger.info("DuckDB VSS extension loaded")
            except Exception:
                logger.info("DuckDB VSS extension not available, using numpy fallback")

        except ImportError:
            raise RuntimeError("DuckDB not installed")

    def _init_schema(self) -> None:
        """Initialize database schema."""
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id VARCHAR PRIMARY KEY,
                content TEXT NOT NULL,
                source VARCHAR,
                metadata JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP
            )
        """)

        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id VARCHAR PRIMARY KEY,
                document_id VARCHAR NOT NULL,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                embedding BLOB,
                metadata JSON,
                FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
            )
        """)

        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id)
        """)

    def add_document(
        self,
        content: str,
        metadata: Optional[dict] = None,
        doc_id: Optional[str] = None,
        source: Optional[str] = None,
        chunks: Optional[list[dict]] = None,
    ) -> str:
        """Add a document with optional chunks."""
        if doc_id is None:
            doc_id = self._generate_id(content)

        metadata_json = json.dumps(metadata or {})

        # Insert or update document
        self._conn.execute("""
            INSERT OR REPLACE INTO documents (id, content, source, metadata, created_at, updated_at)
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
        """, [doc_id, content, source, metadata_json])

        # Add chunks if provided
        if chunks:
            for i, chunk_data in enumerate(chunks):
                chunk_id = f"{doc_id}_chunk_{i}"
                chunk_content = chunk_data.get("content", "")
                chunk_embedding = chunk_data.get("embedding")
                chunk_metadata = json.dumps(chunk_data.get("metadata", {}))

                embedding_blob = None
                if chunk_embedding is not None:
                    embedding_blob = np.array(chunk_embedding, dtype=np.float32).tobytes()

                self._conn.execute("""
                    INSERT OR REPLACE INTO chunks (id, document_id, chunk_index, content, embedding, metadata)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, [chunk_id, doc_id, i, chunk_content, embedding_blob, chunk_metadata])

        logger.debug(f"Added document {doc_id} with {len(chunks or [])} chunks")
        return doc_id

    def add_chunk(
        self,
        document_id: str,
        content: str,
        chunk_index: int,
        embedding: Optional[np.ndarray] = None,
        metadata: Optional[dict] = None,
    ) -> str:
        """Add a single chunk to an existing document."""
        chunk_id = f"{document_id}_chunk_{chunk_index}"
        metadata_json = json.dumps(metadata or {})

        embedding_blob = None
        if embedding is not None:
            embedding_blob = np.array(embedding, dtype=np.float32).tobytes()

        self._conn.execute("""
            INSERT OR REPLACE INTO chunks (id, document_id, chunk_index, content, embedding, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
        """, [chunk_id, document_id, chunk_index, content, embedding_blob, metadata_json])

        return chunk_id

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        threshold: float = 0.0,
        filters: Optional[dict] = None,
    ) -> list[SearchResult]:
        """Search for similar chunks using cosine similarity."""
        query_embedding = np.array(query_embedding, dtype=np.float32)

        # Get all chunks with embeddings
        result = self._conn.execute("""
            SELECT c.id, c.document_id, c.chunk_index, c.content, c.embedding, c.metadata,
                   d.content as doc_content, d.source, d.metadata as doc_metadata
            FROM chunks c
            JOIN documents d ON c.document_id = d.id
            WHERE c.embedding IS NOT NULL
        """).fetchall()

        # Calculate similarities
        scored_results = []
        for row in result:
            chunk_id, doc_id, chunk_idx, chunk_content, emb_blob, chunk_meta, doc_content, source, doc_meta = row

            if emb_blob is None:
                continue

            chunk_embedding = np.frombuffer(emb_blob, dtype=np.float32)
            similarity = self._cosine_similarity(query_embedding, chunk_embedding)

            if similarity >= threshold:
                # Apply filters if provided
                if filters:
                    chunk_metadata = json.loads(chunk_meta) if chunk_meta else {}
                    doc_metadata = json.loads(doc_meta) if doc_meta else {}

                    if not self._matches_filters({**chunk_metadata, **doc_metadata}, filters):
                        continue

                chunk = Chunk(
                    id=chunk_id,
                    document_id=doc_id,
                    content=chunk_content,
                    chunk_index=chunk_idx,
                    metadata=json.loads(chunk_meta) if chunk_meta else {},
                )

                document = Document(
                    id=doc_id,
                    content=doc_content,
                    source=source,
                    metadata=json.loads(doc_meta) if doc_meta else {},
                )

                scored_results.append((chunk, document, similarity))

        # Sort by similarity
        scored_results.sort(key=lambda x: x[2], reverse=True)

        # Build results
        results = []
        for rank, (chunk, document, score) in enumerate(scored_results[:top_k], start=1):
            results.append(SearchResult(
                chunk=chunk,
                document=document,
                score=score,
                rank=rank,
            ))

        return results

    def get_document(self, doc_id: str) -> Optional[Document]:
        """Get a document by ID."""
        result = self._conn.execute("""
            SELECT id, content, source, metadata, created_at, updated_at
            FROM documents WHERE id = ?
        """, [doc_id]).fetchone()

        if result is None:
            return None

        return Document(
            id=result[0],
            content=result[1],
            source=result[2],
            metadata=json.loads(result[3]) if result[3] else {},
            created_at=result[4],
            updated_at=result[5],
        )

    def get_chunks(self, doc_id: str) -> list[Chunk]:
        """Get all chunks for a document."""
        result = self._conn.execute("""
            SELECT id, document_id, chunk_index, content, embedding, metadata
            FROM chunks WHERE document_id = ?
            ORDER BY chunk_index
        """, [doc_id]).fetchall()

        chunks = []
        for row in result:
            embedding = None
            if row[4]:
                embedding = np.frombuffer(row[4], dtype=np.float32)

            chunks.append(Chunk(
                id=row[0],
                document_id=row[1],
                chunk_index=row[2],
                content=row[3],
                embedding=embedding,
                metadata=json.loads(row[5]) if row[5] else {},
            ))

        return chunks

    def delete_document(self, doc_id: str) -> bool:
        """Delete a document and its chunks."""
        # Delete chunks first
        self._conn.execute("DELETE FROM chunks WHERE document_id = ?", [doc_id])

        # Delete document
        result = self._conn.execute("DELETE FROM documents WHERE id = ?", [doc_id])

        return result.rowcount > 0

    def count(self) -> int:
        """Get document count."""
        result = self._conn.execute("SELECT COUNT(*) FROM documents").fetchone()
        return result[0] if result else 0

    def chunk_count(self) -> int:
        """Get chunk count."""
        result = self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()
        return result[0] if result else 0

    def list_documents(self, limit: int = 100, offset: int = 0) -> list[Document]:
        """List documents with pagination."""
        result = self._conn.execute("""
            SELECT id, content, source, metadata, created_at, updated_at
            FROM documents
            ORDER BY created_at DESC
            LIMIT ? OFFSET ?
        """, [limit, offset]).fetchall()

        return [
            Document(
                id=row[0],
                content=row[1][:500] + "..." if len(row[1]) > 500 else row[1],
                source=row[2],
                metadata=json.loads(row[3]) if row[3] else {},
                created_at=row[4],
                updated_at=row[5],
            )
            for row in result
        ]

    def clear(self) -> None:
        """Clear all data."""
        self._conn.execute("DELETE FROM chunks")
        self._conn.execute("DELETE FROM documents")
        logger.info("Vector store cleared")

    def close(self) -> None:
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    def _generate_id(self, content: str) -> str:
        """Generate unique ID from content."""
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity."""
        dot = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot / (norm_a * norm_b))

    def _matches_filters(self, metadata: dict, filters: dict) -> bool:
        """Check if metadata matches filters."""
        for key, value in filters.items():
            if key not in metadata:
                return False
            if isinstance(value, list):
                if metadata[key] not in value:
                    return False
            elif metadata[key] != value:
                return False
        return True


class SQLiteVectorStore(BaseVectorStore):
    """SQLite-based vector store with numpy similarity search."""

    def __init__(
        self,
        db_path: Optional[str] = None,
        embedding_dim: int = 384,
    ):
        self._db_path = db_path or ":memory:"
        self._embedding_dim = embedding_dim
        self._conn = None

        self._connect()
        self._init_schema()

    def _connect(self) -> None:
        """Connect to SQLite."""
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row

    def _init_schema(self) -> None:
        """Initialize database schema."""
        cursor = self._conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                source TEXT,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                embedding BLOB,
                metadata TEXT,
                FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(document_id)
        """)

        self._conn.commit()

    def add_document(
        self,
        content: str,
        metadata: Optional[dict] = None,
        doc_id: Optional[str] = None,
        source: Optional[str] = None,
        chunks: Optional[list[dict]] = None,
    ) -> str:
        """Add a document with optional chunks."""
        if doc_id is None:
            doc_id = hashlib.sha256(content.encode()).hexdigest()[:16]

        cursor = self._conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO documents (id, content, source, metadata, updated_at)
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
        """, [doc_id, content, source, json.dumps(metadata or {})])

        if chunks:
            for i, chunk_data in enumerate(chunks):
                chunk_id = f"{doc_id}_chunk_{i}"
                embedding = chunk_data.get("embedding")
                embedding_blob = np.array(embedding, dtype=np.float32).tobytes() if embedding is not None else None

                cursor.execute("""
                    INSERT OR REPLACE INTO chunks (id, document_id, chunk_index, content, embedding, metadata)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, [
                    chunk_id,
                    doc_id,
                    i,
                    chunk_data.get("content", ""),
                    embedding_blob,
                    json.dumps(chunk_data.get("metadata", {})),
                ])

        self._conn.commit()
        return doc_id

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        threshold: float = 0.0,
        filters: Optional[dict] = None,
    ) -> list[SearchResult]:
        """Search for similar chunks."""
        query_embedding = np.array(query_embedding, dtype=np.float32)

        cursor = self._conn.cursor()
        cursor.execute("""
            SELECT c.id, c.document_id, c.chunk_index, c.content, c.embedding, c.metadata,
                   d.content, d.source, d.metadata
            FROM chunks c
            JOIN documents d ON c.document_id = d.id
            WHERE c.embedding IS NOT NULL
        """)

        scored = []
        for row in cursor.fetchall():
            if row[4] is None:
                continue

            emb = np.frombuffer(row[4], dtype=np.float32)
            score = float(np.dot(query_embedding, emb) / (np.linalg.norm(query_embedding) * np.linalg.norm(emb) + 1e-8))

            if score >= threshold:
                chunk = Chunk(
                    id=row[0],
                    document_id=row[1],
                    chunk_index=row[2],
                    content=row[3],
                    metadata=json.loads(row[5]) if row[5] else {},
                )
                doc = Document(
                    id=row[1],
                    content=row[6],
                    source=row[7],
                    metadata=json.loads(row[8]) if row[8] else {},
                )
                scored.append((chunk, doc, score))

        scored.sort(key=lambda x: x[2], reverse=True)

        return [
            SearchResult(chunk=c, document=d, score=s, rank=i+1)
            for i, (c, d, s) in enumerate(scored[:top_k])
        ]

    def get_document(self, doc_id: str) -> Optional[Document]:
        """Get document by ID."""
        cursor = self._conn.cursor()
        cursor.execute("SELECT * FROM documents WHERE id = ?", [doc_id])
        row = cursor.fetchone()

        if row is None:
            return None

        return Document(
            id=row["id"],
            content=row["content"],
            source=row["source"],
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
        )

    def delete_document(self, doc_id: str) -> bool:
        """Delete document and chunks."""
        cursor = self._conn.cursor()
        cursor.execute("DELETE FROM chunks WHERE document_id = ?", [doc_id])
        cursor.execute("DELETE FROM documents WHERE id = ?", [doc_id])
        self._conn.commit()
        return cursor.rowcount > 0

    def count(self) -> int:
        """Get document count."""
        cursor = self._conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM documents")
        return cursor.fetchone()[0]

    def close(self) -> None:
        """Close connection."""
        if self._conn:
            self._conn.close()


class InMemoryVectorStore(BaseVectorStore):
    """Simple in-memory vector store."""

    def __init__(self, embedding_dim: int = 384):
        self._embedding_dim = embedding_dim
        self._documents: dict[str, Document] = {}
        self._chunks: dict[str, Chunk] = {}

    def add_document(
        self,
        content: str,
        metadata: Optional[dict] = None,
        doc_id: Optional[str] = None,
        source: Optional[str] = None,
        chunks: Optional[list[dict]] = None,
    ) -> str:
        if doc_id is None:
            doc_id = hashlib.sha256(content.encode()).hexdigest()[:16]

        self._documents[doc_id] = Document(
            id=doc_id,
            content=content,
            metadata=metadata or {},
            source=source,
        )

        if chunks:
            for i, chunk_data in enumerate(chunks):
                chunk_id = f"{doc_id}_chunk_{i}"
                embedding = chunk_data.get("embedding")
                self._chunks[chunk_id] = Chunk(
                    id=chunk_id,
                    document_id=doc_id,
                    content=chunk_data.get("content", ""),
                    chunk_index=i,
                    embedding=np.array(embedding, dtype=np.float32) if embedding else None,
                    metadata=chunk_data.get("metadata", {}),
                )

        return doc_id

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        threshold: float = 0.0,
        filters: Optional[dict] = None,
    ) -> list[SearchResult]:
        query_embedding = np.array(query_embedding, dtype=np.float32)
        scored = []

        for chunk in self._chunks.values():
            if chunk.embedding is None:
                continue

            score = float(np.dot(query_embedding, chunk.embedding) /
                         (np.linalg.norm(query_embedding) * np.linalg.norm(chunk.embedding) + 1e-8))

            if score >= threshold:
                doc = self._documents.get(chunk.document_id)
                scored.append((chunk, doc, score))

        scored.sort(key=lambda x: x[2], reverse=True)

        return [
            SearchResult(chunk=c, document=d, score=s, rank=i+1)
            for i, (c, d, s) in enumerate(scored[:top_k])
        ]

    def get_document(self, doc_id: str) -> Optional[Document]:
        return self._documents.get(doc_id)

    def delete_document(self, doc_id: str) -> bool:
        if doc_id not in self._documents:
            return False

        del self._documents[doc_id]

        # Delete associated chunks
        to_delete = [cid for cid, c in self._chunks.items() if c.document_id == doc_id]
        for cid in to_delete:
            del self._chunks[cid]

        return True

    def count(self) -> int:
        return len(self._documents)


# Factory function
def create_vector_store(
    db_path: Optional[str] = None,
    prefer_duckdb: bool = True,
    embedding_dim: int = 384,
) -> BaseVectorStore:
    """
    Create a vector store with automatic backend selection.

    Args:
        db_path: Database file path (None for in-memory)
        prefer_duckdb: Try DuckDB first (default True)
        embedding_dim: Embedding dimension

    Returns:
        Vector store instance
    """
    if prefer_duckdb:
        try:
            store = DuckDBVectorStore(db_path, embedding_dim)
            logger.info(f"Using DuckDB vector store: {db_path or ':memory:'}")
            return store
        except Exception as e:
            logger.warning(f"DuckDB not available: {e}")

    try:
        store = SQLiteVectorStore(db_path, embedding_dim)
        logger.info(f"Using SQLite vector store: {db_path or ':memory:'}")
        return store
    except Exception as e:
        logger.warning(f"SQLite failed: {e}")

    logger.info("Using in-memory vector store")
    return InMemoryVectorStore(embedding_dim)


# Backward compatibility alias
RAGStore = InMemoryVectorStore


# Global default store
_default_store: Optional[BaseVectorStore] = None


def get_default_store() -> BaseVectorStore:
    """Get or create the default vector store."""
    global _default_store
    if _default_store is None:
        _default_store = create_vector_store()
    return _default_store


def set_default_store(store: BaseVectorStore) -> None:
    """Set the default vector store."""
    global _default_store
    _default_store = store
