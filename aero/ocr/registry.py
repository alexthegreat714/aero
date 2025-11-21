"""
OCR backend detection and registry for Aero Agent.

Provides automatic detection of available OCR backends:
- Tesseract OCR
- DeepSeek OCR (via Ollama)
- EasyOCR
"""

import logging
import shutil
import subprocess
from typing import Optional

logger = logging.getLogger(__name__)


def detect_tesseract() -> dict:
    """
    Detect if Tesseract OCR is available.

    Returns:
        Dictionary with:
            - available: bool
            - version: str or None
            - path: str or None
    """
    result = {
        "available": False,
        "version": None,
        "path": None,
    }

    # Check if tesseract is in PATH
    tesseract_path = shutil.which("tesseract")

    if tesseract_path:
        result["path"] = tesseract_path
        result["available"] = True

        # Try to get version
        try:
            proc = subprocess.run(
                ["tesseract", "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            # Parse version from output (first line)
            output = proc.stdout or proc.stderr
            if output:
                first_line = output.strip().split("\n")[0]
                result["version"] = first_line
        except Exception as e:
            logger.debug(f"Could not get Tesseract version: {e}")

    logger.debug(f"Tesseract detection: {result}")
    return result


def detect_deepseek() -> dict:
    """
    Detect if DeepSeek OCR model is available via Ollama.

    Checks if Ollama is installed and if the deepseek-ocr model is available.

    Returns:
        Dictionary with:
            - available: bool
            - ollama_installed: bool
            - model_name: str or None
    """
    result = {
        "available": False,
        "ollama_installed": False,
        "model_name": None,
    }

    # Check if ollama is in PATH
    ollama_path = shutil.which("ollama")

    if not ollama_path:
        logger.debug("Ollama not found in PATH")
        return result

    result["ollama_installed"] = True

    # Try to list models and find deepseek-ocr
    try:
        proc = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if proc.returncode == 0:
            output = proc.stdout.lower()

            # Check for various deepseek OCR model names
            model_names = [
                "deepseek-ocr",
                "deepseek-vl",
                "deepseek-vision",
            ]

            for model_name in model_names:
                if model_name in output:
                    result["available"] = True
                    result["model_name"] = model_name
                    break

    except subprocess.TimeoutExpired:
        logger.warning("Ollama list command timed out")
    except Exception as e:
        logger.debug(f"Could not check Ollama models: {e}")

    logger.debug(f"DeepSeek OCR detection: {result}")
    return result


def detect_easyocr() -> dict:
    """
    Detect if EasyOCR is available.

    Returns:
        Dictionary with:
            - available: bool
            - version: str or None
            - languages: list of available languages
    """
    result = {
        "available": False,
        "version": None,
        "languages": [],
    }

    try:
        import easyocr
        result["available"] = True

        # Try to get version
        try:
            result["version"] = easyocr.__version__
        except AttributeError:
            pass

        # Default supported languages
        result["languages"] = ["en", "ch_sim", "ch_tra", "ja", "ko"]

    except ImportError:
        logger.debug("EasyOCR not installed")

    logger.debug(f"EasyOCR detection: {result}")
    return result


def detect_available_backends() -> dict:
    """
    Detect all available OCR backends.

    Returns:
        Dictionary with:
            - tesseract: bool
            - deepseek_ocr: bool
            - easyocr: bool
            - details: dict with full detection results
    """
    tesseract = detect_tesseract()
    deepseek = detect_deepseek()
    easyocr = detect_easyocr()

    return {
        "tesseract": tesseract["available"],
        "deepseek_ocr": deepseek["available"],
        "easyocr": easyocr["available"],
        "details": {
            "tesseract": tesseract,
            "deepseek_ocr": deepseek,
            "easyocr": easyocr,
        },
    }


def get_best_backend() -> Optional[str]:
    """
    Get the best available OCR backend.

    Priority order:
    1. DeepSeek OCR (best quality for complex documents)
    2. EasyOCR (good quality, easy to use)
    3. Tesseract (most widely available)

    Returns:
        Backend name or None if no backend available
    """
    backends = detect_available_backends()

    if backends["deepseek_ocr"]:
        return "deepseek"
    elif backends["easyocr"]:
        return "easyocr"
    elif backends["tesseract"]:
        return "tesseract"
    else:
        return None


def get_ollama_models() -> list[str]:
    """
    Get list of available Ollama models.

    Returns:
        List of model names
    """
    models = []

    ollama_path = shutil.which("ollama")
    if not ollama_path:
        return models

    try:
        proc = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if proc.returncode == 0:
            lines = proc.stdout.strip().split("\n")
            # Skip header line
            for line in lines[1:]:
                if line.strip():
                    # Model name is first column
                    model_name = line.split()[0]
                    models.append(model_name)

    except Exception as e:
        logger.debug(f"Could not list Ollama models: {e}")

    return models
