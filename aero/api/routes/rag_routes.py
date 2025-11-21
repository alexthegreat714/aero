"""
RAG API routes for Aero Agent.

Provides endpoints for document ingestion, search, and management.
"""

import logging
import tempfile
from pathlib import Path
from typing import Optional, List

from fastapi import APIRouter, HTTPException, UploadFile, File, Query, BackgroundTasks
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter()


# -----------------------------------------------------------------------------
# Request/Response Models
# -----------------------------------------------------------------------------


class DocumentCreate(BaseModel):
    """Request body for creating a document."""

    content: str = Field(..., description="Document text content")
    metadata: Optional[dict] = Field(default=None, description="Optional metadata")
    source: Optional[str] = Field(default=None, description="Source identifier")
    tags: Optional[List[str]] = Field(default=None, description="Document tags")


class DocumentUpdate(BaseModel):
    """Request body for updating a document."""

    content: Optional[str] = None
    metadata: Optional[dict] = None
    tags: Optional[List[str]] = None


class SearchQuery(BaseModel):
    """Request body for search queries."""

    query: str = Field(..., description="Search query text")
    top_k: int = Field(default=5, ge=1, le=100, description="Number of results")
    threshold: float = Field(default=0.0, ge=0.0, le=1.0, description="Minimum similarity")
    filters: Optional[dict] = Field(default=None, description="Metadata filters")


class SearchResult(BaseModel):
    """Search result item."""

    chunk_id: str
    document_id: str
    content: str
    score: float
    rank: int
    metadata: dict


class DocumentResponse(BaseModel):
    """Document response model."""

    id: str
    content: str
    source: Optional[str]
    metadata: dict
    chunk_count: int = 0


class IngestResponse(BaseModel):
    """Response for document ingestion."""

    document_id: str
    chunks_created: int
    source: str
    status: str


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------


def _get_rag_components():
    """Get RAG components with lazy loading."""
    try:
        from aero.rag.store import get_default_store
        from aero.rag.embedder import get_default_embedder
        from aero.rag.reader import DocumentReader
        from aero.rag.chunker import TextChunker
        from aero.rag.preprocess import TextPreprocessor

        return {
            "store": get_default_store(),
            "embedder": get_default_embedder(),
            "reader": DocumentReader(),
            "chunker": TextChunker(),
            "preprocessor": TextPreprocessor(),
        }
    except Exception as e:
        logger.error(f"Failed to initialize RAG components: {e}")
        raise HTTPException(status_code=503, detail="RAG system not available")


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------


@router.get("/")
async def rag_status():
    """Get RAG system status."""
    try:
        components = _get_rag_components()
        store = components["store"]
        embedder = components["embedder"]

        return {
            "status": "ok",
            "module": "rag",
            "document_count": store.count(),
            "embedder": embedder.name,
            "embedder_dimension": embedder.dimension,
        }
    except HTTPException:
        raise
    except Exception as e:
        return {
            "status": "degraded",
            "module": "rag",
            "error": str(e),
        }


@router.post("/documents", response_model=IngestResponse)
async def add_document(doc: DocumentCreate):
    """
    Add a document to the RAG store.

    The document will be:
    1. Preprocessed (cleaned, normalized)
    2. Chunked into smaller pieces
    3. Embedded for semantic search
    4. Stored in the vector database

    Returns the document ID and chunk count.
    """
    try:
        components = _get_rag_components()
        store = components["store"]
        embedder = components["embedder"]
        chunker = components["chunker"]
        preprocessor = components["preprocessor"]

        # Preprocess content
        preprocessed = preprocessor.preprocess(doc.content)
        cleaned_text = preprocessor.clean_for_embedding(preprocessed.text)

        # Create chunks
        chunks = chunker.chunk(cleaned_text)

        # Generate embeddings for chunks
        chunk_data = []
        for i, chunk_text in enumerate(chunks):
            embedding = embedder.embed(chunk_text)
            chunk_data.append({
                "content": chunk_text,
                "embedding": embedding.tolist(),
                "metadata": {"chunk_index": i, **(doc.metadata or {})},
            })

        # Add to store
        metadata = doc.metadata or {}
        if doc.tags:
            metadata["tags"] = doc.tags

        doc_id = store.add_document(
            content=doc.content,
            metadata=metadata,
            source=doc.source,
            chunks=chunk_data,
        )

        logger.info(f"Added document {doc_id} with {len(chunks)} chunks")

        return IngestResponse(
            document_id=doc_id,
            chunks_created=len(chunks),
            source=doc.source or "inline",
            status="created",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to add document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/documents")
async def list_documents(
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
):
    """List documents in the RAG store with pagination."""
    try:
        components = _get_rag_components()
        store = components["store"]

        # Try to get list_documents method if available
        if hasattr(store, "list_documents"):
            documents = store.list_documents(limit=limit, offset=offset)
            doc_list = [
                {
                    "id": doc.id,
                    "content": doc.content[:200] + "..." if len(doc.content) > 200 else doc.content,
                    "source": doc.source,
                    "metadata": doc.metadata,
                }
                for doc in documents
            ]
        else:
            doc_list = []

        return {
            "status": "ok",
            "documents": doc_list,
            "total": store.count(),
            "limit": limit,
            "offset": offset,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to list documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/documents/{doc_id}")
async def get_document(doc_id: str):
    """Get a specific document by ID."""
    try:
        components = _get_rag_components()
        store = components["store"]

        document = store.get_document(doc_id)

        if document is None:
            raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")

        # Get chunks if available
        chunk_count = 0
        if hasattr(store, "get_chunks"):
            chunks = store.get_chunks(doc_id)
            chunk_count = len(chunks)

        return DocumentResponse(
            id=document.id,
            content=document.content,
            source=document.source,
            metadata=document.metadata,
            chunk_count=chunk_count,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    """Delete a document from the RAG store."""
    try:
        components = _get_rag_components()
        store = components["store"]

        deleted = store.delete_document(doc_id)

        if not deleted:
            raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")

        logger.info(f"Deleted document {doc_id}")

        return {
            "status": "deleted",
            "document_id": doc_id,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search")
async def search_documents(query: SearchQuery):
    """
    Search for relevant documents using semantic similarity.

    Returns ranked list of chunks matching the query.
    """
    try:
        components = _get_rag_components()
        store = components["store"]
        embedder = components["embedder"]

        # Generate query embedding
        query_embedding = embedder.embed(query.query)

        # Search
        results = store.search(
            query_embedding=query_embedding,
            top_k=query.top_k,
            threshold=query.threshold,
            filters=query.filters,
        )

        # Format results
        search_results = []
        for result in results:
            search_results.append(SearchResult(
                chunk_id=result.chunk.id,
                document_id=result.chunk.document_id,
                content=result.chunk.content,
                score=result.score,
                rank=result.rank,
                metadata=result.chunk.metadata,
            ))

        return {
            "status": "ok",
            "query": query.query,
            "results": [r.model_dump() for r in search_results],
            "total": len(search_results),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    tags: Optional[str] = Query(default=None, description="Comma-separated tags"),
):
    """
    Upload a document file to the RAG store.

    Supports: .txt, .md, .pdf, .docx, .html, .json, .yaml
    Images (.jpg, .png) are processed via OCR.
    """
    try:
        components = _get_rag_components()
        reader = components["reader"]

        # Check file extension
        filename = file.filename or "unknown"
        suffix = Path(filename).suffix.lower()

        supported = reader.supported_formats()
        if suffix not in supported:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format: {suffix}. Supported: {supported}",
            )

        # Save to temp file and read
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        try:
            # Parse document
            parsed = reader.read(tmp_path)

            # Create document request
            metadata = parsed.metadata.copy()
            metadata["original_filename"] = filename
            metadata["format"] = parsed.format

            if parsed.ocr_used:
                metadata["ocr_used"] = True

            tag_list = None
            if tags:
                tag_list = [t.strip() for t in tags.split(",")]

            doc_request = DocumentCreate(
                content=parsed.content,
                metadata=metadata,
                source=filename,
                tags=tag_list,
            )

            # Use add_document endpoint logic
            return await add_document(doc_request)

        finally:
            # Clean up temp file
            Path(tmp_path).unlink(missing_ok=True)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ingest/batch")
async def batch_ingest(
    files: List[UploadFile] = File(...),
    background_tasks: BackgroundTasks = None,
):
    """
    Batch upload multiple documents.

    Files are processed and ingested into the RAG store.
    """
    results = []
    errors = []

    for file in files:
        try:
            result = await upload_document(file)
            results.append({
                "filename": file.filename,
                "document_id": result.document_id,
                "status": "success",
            })
        except HTTPException as e:
            errors.append({
                "filename": file.filename,
                "error": e.detail,
                "status": "failed",
            })
        except Exception as e:
            errors.append({
                "filename": file.filename,
                "error": str(e),
                "status": "failed",
            })

    return {
        "status": "completed",
        "successful": len(results),
        "failed": len(errors),
        "results": results,
        "errors": errors,
    }


@router.get("/embeddings/info")
async def embeddings_info():
    """Get information about the embedding system."""
    try:
        from aero.rag.embedder import detect_embedding_backends, get_default_embedder

        backends = detect_embedding_backends()
        embedder = get_default_embedder()

        return {
            "status": "ok",
            "current_backend": embedder.name,
            "dimension": embedder.dimension,
            "available_backends": {
                "ollama": backends["ollama"],
                "sentence_transformers": backends["sentence_transformers"],
                "hash_fallback": backends["hash"],
            },
            "details": backends.get("details", {}),
        }

    except Exception as e:
        logger.error(f"Failed to get embedding info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/clear")
async def clear_store():
    """Clear all documents from the RAG store. Use with caution."""
    try:
        components = _get_rag_components()
        store = components["store"]

        if hasattr(store, "clear"):
            store.clear()

        logger.warning("RAG store cleared")

        return {
            "status": "cleared",
            "message": "All documents have been removed",
        }

    except Exception as e:
        logger.error(f"Failed to clear store: {e}")
        raise HTTPException(status_code=500, detail=str(e))
