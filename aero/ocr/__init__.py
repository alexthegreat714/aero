"""
OCR module for Aero Agent.

Provides optical character recognition with multiple backend support:
- Tesseract OCR
- DeepSeek OCR (via Ollama)
- EasyOCR
"""

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

__all__ = [
    "detect_available_backends",
    "detect_tesseract",
    "detect_deepseek",
    "detect_easyocr",
    "get_best_backend",
    "TesseractBackend",
    "DeepSeekBackend",
    "EasyOCRBackend",
]
