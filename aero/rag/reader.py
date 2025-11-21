"""
Document reader for Aero Agent RAG system.

Handles reading and parsing various document formats.
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ParsedDocument:
    """Represents a parsed document."""

    content: str
    metadata: dict
    source: str
    format: str
    pages: Optional[int] = None


class BaseReader(ABC):
    """Base class for document readers."""

    @property
    @abstractmethod
    def supported_formats(self) -> list[str]:
        """List of supported file extensions."""
        raise NotImplementedError

    @abstractmethod
    def read(self, path: Path) -> ParsedDocument:
        """Read and parse a document."""
        raise NotImplementedError

    def can_read(self, path: Path) -> bool:
        """Check if this reader can handle the file."""
        return path.suffix.lower() in self.supported_formats


class TextReader(BaseReader):
    """Reader for plain text files."""

    @property
    def supported_formats(self) -> list[str]:
        return [".txt", ".md", ".rst", ".csv", ".json", ".yaml", ".yml"]

    def read(self, path: Path) -> ParsedDocument:
        """
        Read a text file.

        Args:
            path: Path to the file

        Returns:
            ParsedDocument with content
        """
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        return ParsedDocument(
            content=content,
            metadata={
                "filename": path.name,
                "size": path.stat().st_size,
            },
            source=str(path),
            format=path.suffix.lower(),
        )


class PDFReader(BaseReader):
    """
    Reader for PDF files.

    Requires PyPDF2 or pdfplumber to be installed.
    Falls back to placeholder if not available.
    """

    def __init__(self):
        """Initialize the PDF reader."""
        self._backend: Optional[str] = None

        # Try to detect available PDF backend
        try:
            import PyPDF2
            self._backend = "pypdf2"
            logger.debug("Using PyPDF2 backend for PDF reading")
        except ImportError:
            try:
                import pdfplumber
                self._backend = "pdfplumber"
                logger.debug("Using pdfplumber backend for PDF reading")
            except ImportError:
                logger.warning(
                    "No PDF backend available. "
                    "Install PyPDF2 or pdfplumber for PDF support."
                )

    @property
    def supported_formats(self) -> list[str]:
        return [".pdf"]

    def read(self, path: Path) -> ParsedDocument:
        """
        Read a PDF file.

        Args:
            path: Path to the PDF file

        Returns:
            ParsedDocument with extracted text
        """
        if self._backend == "pypdf2":
            return self._read_pypdf2(path)
        elif self._backend == "pdfplumber":
            return self._read_pdfplumber(path)
        else:
            return ParsedDocument(
                content=f"[PDF content not extracted - no backend available: {path.name}]",
                metadata={"filename": path.name, "error": "no_backend"},
                source=str(path),
                format=".pdf",
            )

    def _read_pypdf2(self, path: Path) -> ParsedDocument:
        """Read PDF using PyPDF2."""
        import PyPDF2

        text_parts = []
        page_count = 0

        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            page_count = len(reader.pages)

            for page in reader.pages:
                text = page.extract_text()
                if text:
                    text_parts.append(text)

        return ParsedDocument(
            content="\n\n".join(text_parts),
            metadata={
                "filename": path.name,
                "backend": "pypdf2",
            },
            source=str(path),
            format=".pdf",
            pages=page_count,
        )

    def _read_pdfplumber(self, path: Path) -> ParsedDocument:
        """Read PDF using pdfplumber."""
        import pdfplumber

        text_parts = []
        page_count = 0

        with pdfplumber.open(path) as pdf:
            page_count = len(pdf.pages)

            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    text_parts.append(text)

        return ParsedDocument(
            content="\n\n".join(text_parts),
            metadata={
                "filename": path.name,
                "backend": "pdfplumber",
            },
            source=str(path),
            format=".pdf",
            pages=page_count,
        )


class DocumentReader:
    """
    Unified document reader that handles multiple formats.

    Automatically selects appropriate reader based on file extension.

    Example:
        reader = DocumentReader()
        doc = reader.read(Path("document.pdf"))
        print(doc.content)
    """

    def __init__(self):
        """Initialize the document reader."""
        self._readers: list[BaseReader] = [
            TextReader(),
            PDFReader(),
        ]

    def read(self, path: str | Path) -> ParsedDocument:
        """
        Read a document from file.

        Args:
            path: Path to the document

        Returns:
            ParsedDocument with content

        Raises:
            ValueError: If file format is not supported
            FileNotFoundError: If file does not exist
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        for reader in self._readers:
            if reader.can_read(path):
                logger.debug(f"Reading {path} with {reader.__class__.__name__}")
                return reader.read(path)

        raise ValueError(f"Unsupported file format: {path.suffix}")

    def read_text(self, text: str, source: str = "inline") -> ParsedDocument:
        """
        Create a ParsedDocument from raw text.

        Args:
            text: Raw text content
            source: Source identifier

        Returns:
            ParsedDocument
        """
        return ParsedDocument(
            content=text,
            metadata={"source_type": "inline"},
            source=source,
            format="text",
        )

    def supported_formats(self) -> list[str]:
        """Get list of all supported file formats."""
        formats = []
        for reader in self._readers:
            formats.extend(reader.supported_formats)
        return formats

    def add_reader(self, reader: BaseReader) -> None:
        """
        Add a custom reader.

        Args:
            reader: Reader instance to add
        """
        self._readers.append(reader)
        logger.info(f"Added reader for formats: {reader.supported_formats}")
