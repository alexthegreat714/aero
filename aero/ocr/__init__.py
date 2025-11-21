"""
OCR module for Aero Agent.

Provides optical character recognition with multiple backend support:
- Tesseract OCR
- DeepSeek OCR (via Ollama)
- EasyOCR

Example:
    from aero.ocr import get_ocr_text
    text = get_ocr_text("document.pdf")
"""

import logging
from pathlib import Path
from typing import Optional

from aero.ocr.registry import (
    detect_available_backends,
    detect_tesseract,
    detect_deepseek,
    detect_easyocr,
    get_best_backend,
)
from aero.ocr.tesseract_backend import TesseractBackend
from aero.ocr.deepseek_backend import DeepSeekBackend
from aero.ocr.easyocr_backend import EasyOCRBackend

logger = logging.getLogger(__name__)

# Cached OCR instance
_ocr_instance: Optional[object] = None


def get_ocr_text(
    path: str,
    backend: Optional[str] = None,
    language: str = "eng",
) -> str:
    """
    Extract text from an image or scanned document using OCR.

    Automatically selects the best available OCR backend if not specified.
    Supports: images (.jpg, .png, .tiff, .bmp, .gif), scanned PDFs.

    Args:
        path: Path to the image or PDF file
        backend: OCR backend to use ('tesseract', 'deepseek', 'easyocr', or None for auto)
        language: Language code for OCR (default: 'eng')

    Returns:
        Extracted text content

    Raises:
        FileNotFoundError: If the file does not exist
        RuntimeError: If no OCR backend is available
        ValueError: If the file format is not supported

    Example:
        text = get_ocr_text("scanned_document.pdf")
        text = get_ocr_text("image.png", backend="tesseract")
    """
    global _ocr_instance

    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    # Determine backend
    if backend is None:
        backend = get_best_backend()

    if backend is None:
        raise RuntimeError(
            "No OCR backend available. Install Tesseract, EasyOCR, or configure DeepSeek via Ollama."
        )

    # Get or create OCR instance
    ocr = _get_ocr_backend(backend, language)

    # Process file based on type
    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        return _ocr_pdf(file_path, ocr)
    elif suffix in [".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".gif"]:
        return ocr.process_image(str(file_path))
    else:
        raise ValueError(f"Unsupported file format for OCR: {suffix}")


def _get_ocr_backend(backend: str, language: str = "eng") -> object:
    """Get an OCR backend instance."""
    global _ocr_instance

    # Return cached instance if same backend
    if _ocr_instance is not None:
        backend_name = getattr(_ocr_instance, "name", "")
        if backend_name == backend:
            return _ocr_instance

    # Create new instance
    if backend == "tesseract":
        _ocr_instance = TesseractBackend(language=language)
    elif backend == "deepseek":
        _ocr_instance = DeepSeekBackend()
    elif backend == "easyocr":
        _ocr_instance = EasyOCRBackend(languages=[language[:2]])
    else:
        raise ValueError(f"Unknown OCR backend: {backend}")

    return _ocr_instance


def _ocr_pdf(pdf_path: Path, ocr: object) -> str:
    """
    Extract text from a PDF using OCR.

    Converts each page to an image and runs OCR.
    """
    text_parts = []

    try:
        # Try pdf2image first
        from pdf2image import convert_from_path

        images = convert_from_path(str(pdf_path), dpi=300)

        for i, image in enumerate(images):
            # Save temporary image
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                image.save(tmp.name, "PNG")
                page_text = ocr.process_image(tmp.name)
                text_parts.append(f"--- Page {i + 1} ---\n{page_text}")

                # Clean up
                Path(tmp.name).unlink(missing_ok=True)

    except ImportError:
        # Fallback: try PyMuPDF (fitz)
        try:
            import fitz

            doc = fitz.open(str(pdf_path))
            for page_num in range(len(doc)):
                page = doc[page_num]

                # Render page to image
                mat = fitz.Matrix(2, 2)  # 2x zoom for better OCR
                pix = page.get_pixmap(matrix=mat)

                import tempfile
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    pix.save(tmp.name)
                    page_text = ocr.process_image(tmp.name)
                    text_parts.append(f"--- Page {page_num + 1} ---\n{page_text}")

                    # Clean up
                    Path(tmp.name).unlink(missing_ok=True)

            doc.close()

        except ImportError:
            logger.warning("Neither pdf2image nor PyMuPDF available for PDF OCR")
            raise RuntimeError(
                "Cannot perform OCR on PDF: install pdf2image or PyMuPDF"
            )

    return "\n\n".join(text_parts)


def clear_ocr_cache() -> None:
    """Clear the cached OCR instance."""
    global _ocr_instance
    _ocr_instance = None


def ocr_available() -> bool:
    """Check if any OCR backend is available."""
    return get_best_backend() is not None


__all__ = [
    # Main function
    "get_ocr_text",
    "ocr_available",
    "clear_ocr_cache",
    # Detection functions
    "detect_available_backends",
    "detect_tesseract",
    "detect_deepseek",
    "detect_easyocr",
    "get_best_backend",
    # Backend classes
    "TesseractBackend",
    "DeepSeekBackend",
    "EasyOCRBackend",
]
