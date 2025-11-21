"""
Tesseract OCR backend for Aero Agent.

Provides OCR using Tesseract via pytesseract.
"""

import logging
from typing import Optional
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Try to import pytesseract
try:
    import pytesseract
    PYTESSERACT_AVAILABLE = True
except ImportError:
    PYTESSERACT_AVAILABLE = False
    logger.warning("pytesseract not installed - Tesseract backend disabled")


@dataclass
class OCRResult:
    """Container for OCR results."""

    text: str
    confidence: float
    language: str
    backend: str
    metadata: dict


class TesseractBackend:
    """
    Tesseract OCR backend.

    Provides OCR functionality using Tesseract via pytesseract.

    Example:
        backend = TesseractBackend()
        result = backend.ocr("image.png")
        print(result.text)
    """

    def __init__(
        self,
        lang: str = "eng",
        tesseract_cmd: Optional[str] = None,
        config: str = "",
    ):
        """
        Initialize the Tesseract backend.

        Args:
            lang: Language code (e.g., "eng", "deu", "fra")
            tesseract_cmd: Path to tesseract executable (auto-detect if None)
            config: Additional tesseract config options
        """
        self.lang = lang
        self.config = config

        if tesseract_cmd and PYTESSERACT_AVAILABLE:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

        self._available = PYTESSERACT_AVAILABLE

        if self._available:
            logger.info(f"Tesseract backend initialized (lang={lang})")
        else:
            logger.warning("Tesseract backend not available")

    @property
    def is_available(self) -> bool:
        """Check if backend is available."""
        return self._available

    def ocr(
        self,
        image: str | Path | np.ndarray | Image.Image,
        lang: Optional[str] = None,
    ) -> OCRResult:
        """
        Perform OCR on an image.

        Args:
            image: Image path, numpy array, or PIL Image
            lang: Override language for this call

        Returns:
            OCRResult with extracted text
        """
        if not self._available:
            return OCRResult(
                text="",
                confidence=0.0,
                language=lang or self.lang,
                backend="tesseract",
                metadata={"error": "pytesseract not available"},
            )

        # Load image if needed
        if isinstance(image, (str, Path)):
            image = Image.open(image)
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        lang = lang or self.lang

        try:
            # Get OCR text
            text = pytesseract.image_to_string(
                image,
                lang=lang,
                config=self.config,
            )

            # Get confidence data
            data = pytesseract.image_to_data(
                image,
                lang=lang,
                config=self.config,
                output_type=pytesseract.Output.DICT,
            )

            # Calculate average confidence
            confidences = [
                int(c) for c in data["conf"]
                if str(c).isdigit() and int(c) > 0
            ]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0

            return OCRResult(
                text=text.strip(),
                confidence=avg_confidence / 100.0,
                language=lang,
                backend="tesseract",
                metadata={
                    "word_count": len([t for t in data["text"] if t.strip()]),
                },
            )

        except Exception as e:
            logger.error(f"Tesseract OCR failed: {e}")
            return OCRResult(
                text="",
                confidence=0.0,
                language=lang,
                backend="tesseract",
                metadata={"error": str(e)},
            )

    def ocr_to_boxes(
        self,
        image: str | Path | np.ndarray | Image.Image,
        lang: Optional[str] = None,
    ) -> list[dict]:
        """
        Perform OCR and return bounding boxes for each word.

        Args:
            image: Image path, numpy array, or PIL Image
            lang: Override language

        Returns:
            List of dictionaries with text and bounding box info
        """
        if not self._available:
            return []

        # Load image if needed
        if isinstance(image, (str, Path)):
            image = Image.open(image)
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        lang = lang or self.lang

        try:
            data = pytesseract.image_to_data(
                image,
                lang=lang,
                config=self.config,
                output_type=pytesseract.Output.DICT,
            )

            boxes = []
            for i, text in enumerate(data["text"]):
                if text.strip():
                    boxes.append({
                        "text": text,
                        "left": data["left"][i],
                        "top": data["top"][i],
                        "width": data["width"][i],
                        "height": data["height"][i],
                        "confidence": data["conf"][i] / 100.0,
                    })

            return boxes

        except Exception as e:
            logger.error(f"Tesseract box detection failed: {e}")
            return []

    def get_languages(self) -> list[str]:
        """
        Get list of available languages.

        Returns:
            List of language codes
        """
        if not self._available:
            return []

        try:
            langs = pytesseract.get_languages()
            return [l for l in langs if l != "osd"]
        except Exception:
            return ["eng"]
