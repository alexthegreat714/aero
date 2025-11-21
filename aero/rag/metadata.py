"""
Metadata storage for Aero Agent RAG system.

Provides DuckDB/SQLite-based storage for documents, chunks, and metadata.
Falls back to SQLite if DuckDB is not available.
"""

import json
import logging
import sqlite3
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Try to import DuckDB
try:
    import duckdb
    DUCKDB_AVAILABLE = True
except ImportError:
    DUCKDB_AVAILABLE = False
    logger.info("DuckDB not available, will use SQLite fallback")


class BaseMetadataStore(ABC):
    """Base class for metadata storage backends."""

    @abstractmethod
    def initialize(self) -> None:
        """Initialize the database schema."""
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        """Close the database connection."""
        raise NotImplementedError

    # Document operations
    @abstractmethod
    def add_document(
        self,
        doc_id: str,
        path: str,
        title: str,
        tags: list[str],
        metadata: dict,
    ) -> None:
        """Add a document to the store."""
        raise NotImplementedError

    @abstractmethod
    def get_document(self, doc_id: str) -> Optional[dict]:
        """Get a document by ID."""
        raise NotImplementedError

    @abstractmethod
    def update_document(self, doc_id: str, **kwargs) -> bool:
        """Update a document."""
        raise NotImplementedError

    @abstractmethod
    def delete_document(self, doc_id: str) -> bool:
        """Delete a document and its chunks."""
        raise NotImplementedError

    @abstractmethod
    def list_documents(self) -> list[dict]:
        """List all documents."""
        raise NotImplementedError

    # Chunk operations
    @abstractmethod
    def add_chunk(
        self,
        chunk_id: str,
        document_id: str,
        chunk_index: int,
        text: str,
        embedding: np.ndarray,
        metadata: dict,
    ) -> None:
        """Add a chunk to the store."""
        raise NotImplementedError

    @abstractmethod
    def get_chunks_for_document(self, document_id: str) -> list[dict]:
        """Get all chunks for a document."""
        raise NotImplementedError

    @abstractmethod
    def search_chunks(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        threshold: float = 0.0,
    ) -> list[dict]:
        """Search for similar chunks."""
        raise NotImplementedError

    # Metadata operations
    @abstractmethod
    def get_metadata(self, key: str) -> Optional[str]:
        """Get a metadata value."""
        raise NotImplementedError

    @abstractmethod
    def set_metadata(self, key: str, value: str) -> None:
        """Set a metadata value."""
        raise NotImplementedError


class SQLiteMetadataStore(BaseMetadataStore):
    """SQLite-based metadata storage."""

    def __init__(self, db_path: str):
        """
        Initialize SQLite store.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: Optional[sqlite3.Connection] = None
        self._connect()
        self.initialize()

    def _connect(self) -> None:
        """Connect to the database."""
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row

    def initialize(self) -> None:
        """Initialize the database schema."""
        cursor = self._conn.cursor()

        # Documents table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                path TEXT,
                title TEXT,
                tags TEXT,
                metadata TEXT,
                created TEXT,
                updated TEXT
            )
        """)

        # Chunks table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                document_id TEXT,
                chunk_index INTEGER,
                text TEXT,
                embedding BLOB,
                metadata TEXT,
                FOREIGN KEY (document_id) REFERENCES documents(id)
            )
        """)

        # Metadata table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """)

        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(document_id)")

        self._conn.commit()
        logger.info(f"SQLite database initialized at {self.db_path}")

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    def add_document(
        self,
        doc_id: str,
        path: str,
        title: str,
        tags: list[str],
        metadata: dict,
    ) -> None:
        """Add a document to the store."""
        cursor = self._conn.cursor()
        now = datetime.now().isoformat()

        cursor.execute(
            """
            INSERT OR REPLACE INTO documents (id, path, title, tags, metadata, created, updated)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (doc_id, path, title, json.dumps(tags), json.dumps(metadata), now, now),
        )
        self._conn.commit()

    def get_document(self, doc_id: str) -> Optional[dict]:
        """Get a document by ID."""
        cursor = self._conn.cursor()
        cursor.execute("SELECT * FROM documents WHERE id = ?", (doc_id,))
        row = cursor.fetchone()

        if row:
            return {
                "id": row["id"],
                "path": row["path"],
                "title": row["title"],
                "tags": json.loads(row["tags"]),
                "metadata": json.loads(row["metadata"]),
                "created": row["created"],
                "updated": row["updated"],
            }
        return None

    def update_document(self, doc_id: str, **kwargs) -> bool:
        """Update a document."""
        doc = self.get_document(doc_id)
        if not doc:
            return False

        # Build update query
        updates = []
        values = []

        for key, value in kwargs.items():
            if key in ("tags", "metadata"):
                value = json.dumps(value)
            updates.append(f"{key} = ?")
            values.append(value)

        updates.append("updated = ?")
        values.append(datetime.now().isoformat())
        values.append(doc_id)

        cursor = self._conn.cursor()
        cursor.execute(
            f"UPDATE documents SET {', '.join(updates)} WHERE id = ?",
            values,
        )
        self._conn.commit()
        return True

    def delete_document(self, doc_id: str) -> bool:
        """Delete a document and its chunks."""
        cursor = self._conn.cursor()

        # Delete chunks first
        cursor.execute("DELETE FROM chunks WHERE document_id = ?", (doc_id,))

        # Delete document
        cursor.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        deleted = cursor.rowcount > 0

        self._conn.commit()
        return deleted

    def list_documents(self) -> list[dict]:
        """List all documents."""
        cursor = self._conn.cursor()
        cursor.execute("SELECT * FROM documents ORDER BY created DESC")

        documents = []
        for row in cursor.fetchall():
            documents.append({
                "id": row["id"],
                "path": row["path"],
                "title": row["title"],
                "tags": json.loads(row["tags"]),
                "metadata": json.loads(row["metadata"]),
                "created": row["created"],
                "updated": row["updated"],
            })
        return documents

    def add_chunk(
        self,
        chunk_id: str,
        document_id: str,
        chunk_index: int,
        text: str,
        embedding: np.ndarray,
        metadata: dict,
    ) -> None:
        """Add a chunk to the store."""
        cursor = self._conn.cursor()

        # Store embedding as bytes
        embedding_bytes = embedding.astype(np.float32).tobytes()

        cursor.execute(
            """
            INSERT OR REPLACE INTO chunks (id, document_id, chunk_index, text, embedding, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (chunk_id, document_id, chunk_index, text, embedding_bytes, json.dumps(metadata)),
        )
        self._conn.commit()

    def get_chunks_for_document(self, document_id: str) -> list[dict]:
        """Get all chunks for a document."""
        cursor = self._conn.cursor()
        cursor.execute(
            "SELECT * FROM chunks WHERE document_id = ? ORDER BY chunk_index",
            (document_id,),
        )

        chunks = []
        for row in cursor.fetchall():
            embedding = np.frombuffer(row["embedding"], dtype=np.float32)
            chunks.append({
                "id": row["id"],
                "document_id": row["document_id"],
                "chunk_index": row["chunk_index"],
                "text": row["text"],
                "embedding": embedding,
                "metadata": json.loads(row["metadata"]),
            })
        return chunks

    def get_all_chunks(self) -> list[dict]:
        """Get all chunks from all documents."""
        cursor = self._conn.cursor()
        cursor.execute("SELECT * FROM chunks ORDER BY document_id, chunk_index")

        chunks = []
        for row in cursor.fetchall():
            embedding = np.frombuffer(row["embedding"], dtype=np.float32)
            chunks.append({
                "id": row["id"],
                "document_id": row["document_id"],
                "chunk_index": row["chunk_index"],
                "text": row["text"],
                "embedding": embedding,
                "metadata": json.loads(row["metadata"]),
            })
        return chunks

    def search_chunks(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        threshold: float = 0.0,
    ) -> list[dict]:
        """Search for similar chunks using cosine similarity."""
        all_chunks = self.get_all_chunks()

        if not all_chunks:
            return []

        # Calculate similarities
        results = []
        query_norm = np.linalg.norm(query_embedding)

        for chunk in all_chunks:
            chunk_embedding = chunk["embedding"]
            chunk_norm = np.linalg.norm(chunk_embedding)

            if query_norm > 0 and chunk_norm > 0:
                similarity = float(np.dot(query_embedding, chunk_embedding) / (query_norm * chunk_norm))
            else:
                similarity = 0.0

            if similarity >= threshold:
                results.append({
                    "chunk": chunk,
                    "score": similarity,
                })

        # Sort by similarity
        results.sort(key=lambda x: x["score"], reverse=True)

        return results[:top_k]

    def get_metadata(self, key: str) -> Optional[str]:
        """Get a metadata value."""
        cursor = self._conn.cursor()
        cursor.execute("SELECT value FROM metadata WHERE key = ?", (key,))
        row = cursor.fetchone()
        return row["value"] if row else None

    def set_metadata(self, key: str, value: str) -> None:
        """Set a metadata value."""
        cursor = self._conn.cursor()
        cursor.execute(
            "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
            (key, value),
        )
        self._conn.commit()

    def get_chunk_count(self) -> int:
        """Get total number of chunks."""
        cursor = self._conn.cursor()
        cursor.execute("SELECT COUNT(*) as count FROM chunks")
        return cursor.fetchone()["count"]

    def get_document_count(self) -> int:
        """Get total number of documents."""
        cursor = self._conn.cursor()
        cursor.execute("SELECT COUNT(*) as count FROM documents")
        return cursor.fetchone()["count"]


class DuckDBMetadataStore(BaseMetadataStore):
    """DuckDB-based metadata storage with vector search capabilities."""

    def __init__(self, db_path: str):
        """
        Initialize DuckDB store.

        Args:
            db_path: Path to DuckDB database file
        """
        if not DUCKDB_AVAILABLE:
            raise ImportError("DuckDB is not installed")

        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = duckdb.connect(str(self.db_path))
        self.initialize()

    def initialize(self) -> None:
        """Initialize the database schema."""
        # Documents table
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id VARCHAR PRIMARY KEY,
                path VARCHAR,
                title VARCHAR,
                tags VARCHAR,
                metadata VARCHAR,
                created TIMESTAMP,
                updated TIMESTAMP
            )
        """)

        # Chunks table
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id VARCHAR PRIMARY KEY,
                document_id VARCHAR,
                chunk_index INTEGER,
                text VARCHAR,
                embedding BLOB,
                metadata VARCHAR
            )
        """)

        # Metadata table
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS metadata (
                key VARCHAR PRIMARY KEY,
                value VARCHAR
            )
        """)

        logger.info(f"DuckDB database initialized at {self.db_path}")

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    def add_document(
        self,
        doc_id: str,
        path: str,
        title: str,
        tags: list[str],
        metadata: dict,
    ) -> None:
        """Add a document to the store."""
        now = datetime.now()
        self._conn.execute(
            """
            INSERT OR REPLACE INTO documents (id, path, title, tags, metadata, created, updated)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (doc_id, path, title, json.dumps(tags), json.dumps(metadata), now, now),
        )

    def get_document(self, doc_id: str) -> Optional[dict]:
        """Get a document by ID."""
        result = self._conn.execute(
            "SELECT * FROM documents WHERE id = ?", (doc_id,)
        ).fetchone()

        if result:
            return {
                "id": result[0],
                "path": result[1],
                "title": result[2],
                "tags": json.loads(result[3]),
                "metadata": json.loads(result[4]),
                "created": str(result[5]),
                "updated": str(result[6]),
            }
        return None

    def update_document(self, doc_id: str, **kwargs) -> bool:
        """Update a document."""
        doc = self.get_document(doc_id)
        if not doc:
            return False

        updates = []
        values = []

        for key, value in kwargs.items():
            if key in ("tags", "metadata"):
                value = json.dumps(value)
            updates.append(f"{key} = ?")
            values.append(value)

        updates.append("updated = ?")
        values.append(datetime.now())
        values.append(doc_id)

        self._conn.execute(
            f"UPDATE documents SET {', '.join(updates)} WHERE id = ?",
            values,
        )
        return True

    def delete_document(self, doc_id: str) -> bool:
        """Delete a document and its chunks."""
        self._conn.execute("DELETE FROM chunks WHERE document_id = ?", (doc_id,))
        result = self._conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        return result.rowcount > 0 if hasattr(result, 'rowcount') else True

    def list_documents(self) -> list[dict]:
        """List all documents."""
        results = self._conn.execute(
            "SELECT * FROM documents ORDER BY created DESC"
        ).fetchall()

        documents = []
        for row in results:
            documents.append({
                "id": row[0],
                "path": row[1],
                "title": row[2],
                "tags": json.loads(row[3]),
                "metadata": json.loads(row[4]),
                "created": str(row[5]),
                "updated": str(row[6]),
            })
        return documents

    def add_chunk(
        self,
        chunk_id: str,
        document_id: str,
        chunk_index: int,
        text: str,
        embedding: np.ndarray,
        metadata: dict,
    ) -> None:
        """Add a chunk to the store."""
        embedding_bytes = embedding.astype(np.float32).tobytes()

        self._conn.execute(
            """
            INSERT OR REPLACE INTO chunks (id, document_id, chunk_index, text, embedding, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (chunk_id, document_id, chunk_index, text, embedding_bytes, json.dumps(metadata)),
        )

    def get_chunks_for_document(self, document_id: str) -> list[dict]:
        """Get all chunks for a document."""
        results = self._conn.execute(
            "SELECT * FROM chunks WHERE document_id = ? ORDER BY chunk_index",
            (document_id,),
        ).fetchall()

        chunks = []
        for row in results:
            embedding = np.frombuffer(row[4], dtype=np.float32)
            chunks.append({
                "id": row[0],
                "document_id": row[1],
                "chunk_index": row[2],
                "text": row[3],
                "embedding": embedding,
                "metadata": json.loads(row[5]),
            })
        return chunks

    def get_all_chunks(self) -> list[dict]:
        """Get all chunks from all documents."""
        results = self._conn.execute(
            "SELECT * FROM chunks ORDER BY document_id, chunk_index"
        ).fetchall()

        chunks = []
        for row in results:
            embedding = np.frombuffer(row[4], dtype=np.float32)
            chunks.append({
                "id": row[0],
                "document_id": row[1],
                "chunk_index": row[2],
                "text": row[3],
                "embedding": embedding,
                "metadata": json.loads(row[5]),
            })
        return chunks

    def search_chunks(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        threshold: float = 0.0,
    ) -> list[dict]:
        """Search for similar chunks using cosine similarity."""
        all_chunks = self.get_all_chunks()

        if not all_chunks:
            return []

        results = []
        query_norm = np.linalg.norm(query_embedding)

        for chunk in all_chunks:
            chunk_embedding = chunk["embedding"]
            chunk_norm = np.linalg.norm(chunk_embedding)

            if query_norm > 0 and chunk_norm > 0:
                similarity = float(np.dot(query_embedding, chunk_embedding) / (query_norm * chunk_norm))
            else:
                similarity = 0.0

            if similarity >= threshold:
                results.append({
                    "chunk": chunk,
                    "score": similarity,
                })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def get_metadata(self, key: str) -> Optional[str]:
        """Get a metadata value."""
        result = self._conn.execute(
            "SELECT value FROM metadata WHERE key = ?", (key,)
        ).fetchone()
        return result[0] if result else None

    def set_metadata(self, key: str, value: str) -> None:
        """Set a metadata value."""
        self._conn.execute(
            "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
            (key, value),
        )

    def get_chunk_count(self) -> int:
        """Get total number of chunks."""
        return self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    def get_document_count(self) -> int:
        """Get total number of documents."""
        return self._conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]


def create_metadata_store(db_path: str, prefer_duckdb: bool = True) -> BaseMetadataStore:
    """
    Create a metadata store with automatic backend selection.

    Args:
        db_path: Path to database file
        prefer_duckdb: Prefer DuckDB over SQLite if available

    Returns:
        Metadata store instance
    """
    if prefer_duckdb and DUCKDB_AVAILABLE:
        try:
            return DuckDBMetadataStore(db_path + ".duckdb")
        except Exception as e:
            logger.warning(f"Failed to create DuckDB store: {e}, falling back to SQLite")

    return SQLiteMetadataStore(db_path + ".sqlite")


# Global metadata values
METADATA_KEYS = {
    "last_ingest_time": "last_ingest_time",
    "embeddings_version": "embeddings_version",
    "schema_version": "schema_version",
    "aero_version": "aero_version",
}

SCHEMA_VERSION = "2.0.0"
