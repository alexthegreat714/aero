"""
EasyOCR backend for Aero Agent.

Provides OCR using the EasyOCR library.
"""

import logging
from typing import Optional
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Try to import easyocr
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    logger.warning("easyocr not installed - EasyOCR backend disabled")


@dataclass
class OCRResult:
    """Container for OCR results."""

    text: str
    confidence: float
    language: str
    backend: str
    metadata: dict


class EasyOCRBackend:
    """
    EasyOCR backend.

    Provides OCR functionality using the EasyOCR library with GPU support.

    Example:
        backend = EasyOCRBackend(languages=["en"])
        result = backend.ocr("image.png")
        print(result.text)
    """

    def __init__(
        self,
        languages: list[str] = None,
        gpu: bool = True,
    ):
        """
        Initialize the EasyOCR backend.

        Args:
            languages: List of language codes (default: ["en"])
            gpu: Whether to use GPU if available
        """
        self.languages = languages or ["en"]
        self.gpu = gpu

        self._reader = None
        self._available = EASYOCR_AVAILABLE

        if self._available:
            self._initialize_reader()
        else:
            logger.warning("EasyOCR backend not available")

    def _initialize_reader(self) -> None:
        """Initialize the EasyOCR reader."""
        try:
            self._reader = easyocr.Reader(
                self.languages,
                gpu=self.gpu,
            )
            logger.info(f"EasyOCR initialized (languages={self.languages}, gpu={self.gpu})")
        except Exception as e:
            logger.error(f"Failed to initialize EasyOCR: {e}")
            self._available = False

    @property
    def is_available(self) -> bool:
        """Check if backend is available."""
        return self._available and self._reader is not None

    def ocr(
        self,
        image: str | Path | np.ndarray | Image.Image,
        detail: int = 0,
    ) -> OCRResult:
        """
        Perform OCR on an image.

        Args:
            image: Image path, numpy array, or PIL Image
            detail: Detail level (0 = text only, 1 = with boxes)

        Returns:
            OCRResult with extracted text
        """
        if not self.is_available:
            return OCRResult(
                text="",
                confidence=0.0,
                language=",".join(self.languages),
                backend="easyocr",
                metadata={"error": "EasyOCR not available"},
            )

        # Convert PIL Image to numpy array
        if isinstance(image, Image.Image):
            image = np.array(image)
        elif isinstance(image, (str, Path)):
            image = str(image)

        try:
            results = self._reader.readtext(image, detail=1)

            # Extract text and calculate average confidence
            texts = []
            confidences = []

            for bbox, text, conf in results:
                texts.append(text)
                confidences.append(conf)

            full_text = " ".join(texts)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0

            return OCRResult(
                text=full_text.strip(),
                confidence=avg_confidence,
                language=",".join(self.languages),
                backend="easyocr",
                metadata={
                    "word_count": len(results),
                    "gpu_used": self.gpu,
                },
            )

        except Exception as e:
            logger.error(f"EasyOCR failed: {e}")
            return OCRResult(
                text="",
                confidence=0.0,
                language=",".join(self.languages),
                backend="easyocr",
                metadata={"error": str(e)},
            )

    def ocr_to_boxes(
        self,
        image: str | Path | np.ndarray | Image.Image,
    ) -> list[dict]:
        """
        Perform OCR and return bounding boxes.

        Args:
            image: Image path, numpy array, or PIL Image

        Returns:
            List of dictionaries with text and bounding box info
        """
        if not self.is_available:
            return []

        # Convert PIL Image to numpy array
        if isinstance(image, Image.Image):
            image = np.array(image)
        elif isinstance(image, (str, Path)):
            image = str(image)

        try:
            results = self._reader.readtext(image, detail=1)

            boxes = []
            for bbox, text, conf in results:
                # bbox is a list of 4 points
                x_coords = [p[0] for p in bbox]
                y_coords = [p[1] for p in bbox]

                boxes.append({
                    "text": text,
                    "left": min(x_coords),
                    "top": min(y_coords),
                    "width": max(x_coords) - min(x_coords),
                    "height": max(y_coords) - min(y_coords),
                    "confidence": conf,
                    "bbox": bbox,
                })

            return boxes

        except Exception as e:
            logger.error(f"EasyOCR box detection failed: {e}")
            return []

    def add_language(self, lang: str) -> None:
        """
        Add a language and reinitialize the reader.

        Args:
            lang: Language code to add
        """
        if lang not in self.languages:
            self.languages.append(lang)
            self._initialize_reader()

    @staticmethod
    def supported_languages() -> list[str]:
        """
        Get list of supported languages.

        Returns:
            List of language codes
        """
        # Common supported languages
        return [
            "en", "ch_sim", "ch_tra", "ja", "ko",
            "th", "vi", "fr", "de", "es", "it",
            "pt", "ru", "ar", "hi", "bn",
        ]
