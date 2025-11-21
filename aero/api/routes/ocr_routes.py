"""
OCR API routes for Aero Agent.

Provides endpoints for optical character recognition.
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter()


class OCRRequest(BaseModel):
    """Request for OCR processing."""

    backend: Optional[str] = None  # tesseract, deepseek, easyocr
    language: str = "eng"


@router.get("/")
async def ocr_status():
    """Get OCR module status and available backends."""
    # Import detection functions
    try:
        from aero.ocr.registry import detect_available_backends
        backends = detect_available_backends()
    except Exception:
        backends = {
            "tesseract": False,
            "deepseek_ocr": False,
            "easyocr": False,
        }

    return {
        "status": "ok",
        "module": "ocr",
        "backends": backends,
        "message": "OCR module is operational",
    }


@router.get("/backends")
async def list_backends():
    """List available OCR backends with details."""
    try:
        from aero.ocr.registry import detect_available_backends
        backends = detect_available_backends()
        return {
            "status": "ok",
            "backends": backends.get("details", {}),
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
        }


@router.post("/process")
async def process_image(
    file: UploadFile = File(...),
    backend: Optional[str] = None,
    language: str = "eng",
):
    """
    Process an image with OCR.

    Supports: .png, .jpg, .jpeg, .tiff, .bmp
    """
    logger.info(f"Processing image: {file.filename}")

    return {
        "status": "completed",
        "filename": file.filename,
        "backend": backend or "auto",
        "language": language,
        "text": "Extracted text would appear here (stub)",
        "confidence": 0.0,
        "message": "OCR processing completed (stub)",
    }


@router.post("/process-url")
async def process_url(url: str, backend: Optional[str] = None):
    """Process an image from URL."""
    logger.info(f"Processing image URL: {url}")

    return {
        "status": "completed",
        "url": url,
        "backend": backend or "auto",
        "text": "Extracted text would appear here (stub)",
        "confidence": 0.0,
        "message": "OCR processing completed (stub)",
    }


@router.post("/detect-text")
async def detect_text_regions(file: UploadFile = File(...)):
    """
    Detect text regions in an image.

    Returns bounding boxes for text areas.
    """
    logger.info(f"Detecting text regions: {file.filename}")

    return {
        "status": "completed",
        "filename": file.filename,
        "regions": [],
        "message": "Text detection completed (stub)",
    }


@router.get("/languages")
async def list_languages():
    """List supported OCR languages."""
    return {
        "status": "ok",
        "languages": {
            "tesseract": ["eng", "deu", "fra", "spa", "chi_sim", "jpn"],
            "easyocr": ["en", "ch_sim", "ja", "ko", "de", "fr"],
            "deepseek": ["auto"],
        },
    }
