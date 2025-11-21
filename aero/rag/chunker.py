"""
Document chunking for Aero Agent RAG system.

Provides intelligent text chunking with overlap and context preservation.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Callable, Optional

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """Represents a text chunk."""

    text: str
    index: int
    metadata: dict = field(default_factory=dict)
    start_char: int = 0
    end_char: int = 0


class TextChunker:
    """
    Text chunker for RAG documents.

    Provides configurable chunking with:
    - Token-based or character-based chunking
    - Sliding window overlap
    - Section header preservation
    - Equation context preservation

    Example:
        chunker = TextChunker(chunk_size=512, overlap=0.2)
        chunks = chunker.chunk("Long document text...")
        for chunk in chunks:
            print(f"Chunk {chunk.index}: {len(chunk.text)} chars")
    """

    def __init__(
        self,
        chunk_size: int = 512,
        overlap: float = 0.2,
        min_chunk_size: int = 100,
        max_chunk_size: int = 2048,
        preserve_sentences: bool = True,
        preserve_paragraphs: bool = True,
    ):
        """
        Initialize the chunker.

        Args:
            chunk_size: Target chunk size in tokens (approximate)
            overlap: Overlap ratio between chunks (0.0 to 0.5)
            min_chunk_size: Minimum chunk size in characters
            max_chunk_size: Maximum chunk size in characters
            preserve_sentences: Try to break at sentence boundaries
            preserve_paragraphs: Try to preserve paragraph structure
        """
        self.chunk_size = chunk_size
        self.overlap = min(max(overlap, 0.0), 0.5)
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.preserve_sentences = preserve_sentences
        self.preserve_paragraphs = preserve_paragraphs

        # Approximate chars per token (for English)
        self._chars_per_token = 4

        # Sentence boundary pattern
        self._sentence_end = re.compile(r'[.!?]+[\s\n]+|(?<=\n)\n+')

        # Paragraph boundary pattern
        self._paragraph_end = re.compile(r'\n\s*\n')

    @property
    def target_chars(self) -> int:
        """Target chunk size in characters."""
        return self.chunk_size * self._chars_per_token

    @property
    def overlap_chars(self) -> int:
        """Overlap size in characters."""
        return int(self.target_chars * self.overlap)

    def chunk(self, text: str, metadata: Optional[dict] = None) -> list[Chunk]:
        """
        Chunk text into smaller pieces.

        Args:
            text: Text to chunk
            metadata: Optional base metadata for all chunks

        Returns:
            List of Chunk objects
        """
        if not text or not text.strip():
            return []

        base_metadata = metadata or {}

        # Detect section structure
        sections = self._detect_sections(text)

        if sections:
            # Chunk by sections first
            return self._chunk_by_sections(text, sections, base_metadata)
        else:
            # Simple chunking
            return self._simple_chunk(text, base_metadata)

    def _simple_chunk(
        self,
        text: str,
        base_metadata: dict,
        section_context: Optional[str] = None,
    ) -> list[Chunk]:
        """Simple sliding window chunking."""
        chunks = []
        text_len = len(text)

        if text_len <= self.target_chars:
            # Text is small enough for single chunk
            meta = {**base_metadata}
            if section_context:
                meta["section"] = section_context

            return [Chunk(
                text=text.strip(),
                index=0,
                metadata=meta,
                start_char=0,
                end_char=text_len,
            )]

        # Calculate step size
        step = self.target_chars - self.overlap_chars
        if step <= 0:
            step = self.target_chars // 2

        position = 0
        chunk_index = 0

        while position < text_len:
            # Calculate end position
            end = min(position + self.target_chars, text_len)

            # Try to break at natural boundary
            if end < text_len:
                end = self._find_break_point(text, position, end)

            # Extract chunk text
            chunk_text = text[position:end].strip()

            if len(chunk_text) >= self.min_chunk_size:
                meta = {**base_metadata, "chunk_method": "sliding_window"}
                if section_context:
                    meta["section"] = section_context

                chunks.append(Chunk(
                    text=chunk_text,
                    index=chunk_index,
                    metadata=meta,
                    start_char=position,
                    end_char=end,
                ))
                chunk_index += 1

            # Move position
            position = end - self.overlap_chars
            if position >= end:
                position = end

            # Ensure progress
            if position <= chunks[-1].start_char if chunks else position == 0:
                position = end

        return chunks

    def _find_break_point(self, text: str, start: int, end: int) -> int:
        """Find the best break point near the target end position."""
        search_start = max(start, end - self.overlap_chars)
        search_text = text[search_start:end + self.overlap_chars]

        if self.preserve_paragraphs:
            # Look for paragraph break
            match = None
            for m in self._paragraph_end.finditer(search_text):
                match = m
            if match:
                return search_start + match.end()

        if self.preserve_sentences:
            # Look for sentence break
            match = None
            for m in self._sentence_end.finditer(search_text):
                match = m
            if match:
                return search_start + match.end()

        # Fall back to end position
        return end

    def _detect_sections(self, text: str) -> list[dict]:
        """Detect section headers in text."""
        sections = []

        # Common header patterns
        patterns = [
            (r"^(\d+(?:\.\d+)*)\s+([A-Z][^\n]{2,})$", "numbered"),
            (r"^(#{1,6})\s+([^\n]+)$", "markdown"),
            (r"^([A-Z][A-Z\s]{4,})$", "uppercase"),
        ]

        lines = text.split("\n")
        char_pos = 0

        for i, line in enumerate(lines):
            for pattern, section_type in patterns:
                match = re.match(pattern, line.strip())
                if match:
                    sections.append({
                        "line": i,
                        "char_pos": char_pos,
                        "text": line.strip(),
                        "type": section_type,
                    })
                    break
            char_pos += len(line) + 1  # +1 for newline

        return sections

    def _chunk_by_sections(
        self,
        text: str,
        sections: list[dict],
        base_metadata: dict,
    ) -> list[Chunk]:
        """Chunk text respecting section boundaries."""
        chunks = []
        chunk_index = 0

        for i, section in enumerate(sections):
            # Get section content
            start = section["char_pos"]
            if i + 1 < len(sections):
                end = sections[i + 1]["char_pos"]
            else:
                end = len(text)

            section_text = text[start:end]
            section_title = section["text"]

            # Chunk the section content
            section_chunks = self._simple_chunk(
                section_text,
                {**base_metadata, "section": section_title},
                section_context=section_title,
            )

            # Renumber chunks
            for chunk in section_chunks:
                chunk.index = chunk_index
                chunk.start_char += start
                chunk.end_char += start
                chunks.append(chunk)
                chunk_index += 1

        return chunks

    def chunk_with_equations(
        self,
        text: str,
        equations: list[str],
        metadata: Optional[dict] = None,
    ) -> list[Chunk]:
        """
        Chunk text while preserving equation context.

        Equations are kept with their surrounding text when possible.

        Args:
            text: Text with equation placeholders
            equations: List of extracted equations
            metadata: Base metadata

        Returns:
            List of chunks with equation context
        """
        chunks = self.chunk(text, metadata)

        # Add equation information to relevant chunks
        for chunk in chunks:
            equation_indices = []
            for i, eq in enumerate(equations):
                placeholder = f"[EQUATION_{i}]"
                if placeholder in chunk.text or "[EQUATION]" in chunk.text:
                    equation_indices.append(i)

            if equation_indices:
                chunk.metadata["equations"] = equation_indices
                chunk.metadata["has_equations"] = True

        return chunks


def chunk_text(
    text: str,
    chunk_size: int = 512,
    overlap: float = 0.2,
    **kwargs,
) -> list[dict]:
    """
    Convenience function to chunk text.

    Args:
        text: Text to chunk
        chunk_size: Target chunk size in tokens
        overlap: Overlap ratio
        **kwargs: Additional arguments for TextChunker

    Returns:
        List of chunk dictionaries
    """
    chunker = TextChunker(chunk_size=chunk_size, overlap=overlap, **kwargs)
    chunks = chunker.chunk(text)

    return [
        {
            "text": chunk.text,
            "index": chunk.index,
            "metadata": chunk.metadata,
        }
        for chunk in chunks
    ]
