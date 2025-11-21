"""
DeepSeek OCR backend for Aero Agent.

Provides OCR using DeepSeek vision models via Ollama.
"""

import base64
import logging
import subprocess
from typing import Optional
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
import requests

logger = logging.getLogger(__name__)


@dataclass
class OCRResult:
    """Container for OCR results."""

    text: str
    confidence: float
    language: str
    backend: str
    metadata: dict


class DeepSeekBackend:
    """
    DeepSeek OCR backend via Ollama.

    Uses DeepSeek vision models to extract text from images.

    Example:
        backend = DeepSeekBackend()
        if backend.is_available:
            result = backend.ocr("image.png")
            print(result.text)
    """

    def __init__(
        self,
        model: str = "deepseek-vl",
        ollama_host: str = "http://localhost:11434",
    ):
        """
        Initialize the DeepSeek backend.

        Args:
            model: Ollama model name
            ollama_host: Ollama API host
        """
        self.model = model
        self.ollama_host = ollama_host.rstrip("/")

        self._available = self._check_availability()

        if self._available:
            logger.info(f"DeepSeek backend initialized (model={model})")
        else:
            logger.warning("DeepSeek backend not available")

    def _check_availability(self) -> bool:
        """Check if the backend is available."""
        try:
            # Check if Ollama is running
            response = requests.get(
                f"{self.ollama_host}/api/tags",
                timeout=5,
            )

            if response.status_code != 200:
                return False

            # Check if model is available
            data = response.json()
            models = [m.get("name", "").split(":")[0] for m in data.get("models", [])]

            return self.model in models or any(self.model in m for m in models)

        except Exception as e:
            logger.debug(f"DeepSeek availability check failed: {e}")
            return False

    @property
    def is_available(self) -> bool:
        """Check if backend is available."""
        return self._available

    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string."""
        import io

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def ocr(
        self,
        image: str | Path | np.ndarray | Image.Image,
        prompt: Optional[str] = None,
    ) -> OCRResult:
        """
        Perform OCR on an image using DeepSeek.

        Args:
            image: Image path, numpy array, or PIL Image
            prompt: Custom prompt (default: extract all text)

        Returns:
            OCRResult with extracted text
        """
        if not self._available:
            return OCRResult(
                text="",
                confidence=0.0,
                language="unknown",
                backend="deepseek",
                metadata={"error": "DeepSeek backend not available"},
            )

        # Load image if needed
        if isinstance(image, (str, Path)):
            pil_image = Image.open(image)
        elif isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
        else:
            pil_image = image

        # Convert to base64
        image_b64 = self._image_to_base64(pil_image)

        # Default prompt for OCR
        if prompt is None:
            prompt = (
                "Extract all text from this image. "
                "Return only the extracted text, preserving the layout as much as possible. "
                "Do not add any explanations or commentary."
            )

        try:
            response = requests.post(
                f"{self.ollama_host}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "images": [image_b64],
                    "stream": False,
                },
                timeout=60,
            )

            if response.status_code == 200:
                data = response.json()
                text = data.get("response", "")

                return OCRResult(
                    text=text.strip(),
                    confidence=0.9,  # DeepSeek doesn't provide confidence
                    language="auto",
                    backend="deepseek",
                    metadata={
                        "model": self.model,
                        "total_duration": data.get("total_duration"),
                    },
                )
            else:
                logger.error(f"DeepSeek API error: {response.status_code}")
                return OCRResult(
                    text="",
                    confidence=0.0,
                    language="unknown",
                    backend="deepseek",
                    metadata={"error": f"API error: {response.status_code}"},
                )

        except Exception as e:
            logger.error(f"DeepSeek OCR failed: {e}")
            return OCRResult(
                text="",
                confidence=0.0,
                language="unknown",
                backend="deepseek",
                metadata={"error": str(e)},
            )

    def analyze_document(
        self,
        image: str | Path | np.ndarray | Image.Image,
        query: str,
    ) -> str:
        """
        Analyze a document image with a specific query.

        Args:
            image: Image path, numpy array, or PIL Image
            query: Question about the document

        Returns:
            Analysis result text
        """
        result = self.ocr(image, prompt=query)
        return result.text

    def list_models(self) -> list[str]:
        """
        List available Ollama models.

        Returns:
            List of model names
        """
        try:
            response = requests.get(
                f"{self.ollama_host}/api/tags",
                timeout=5,
            )

            if response.status_code == 200:
                data = response.json()
                return [m.get("name", "") for m in data.get("models", [])]

        except Exception as e:
            logger.debug(f"Could not list models: {e}")

        return []
