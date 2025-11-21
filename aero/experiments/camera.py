"""
Camera capture utilities for Aero Agent.

Provides webcam and camera control for experiments.
"""

import logging
from typing import Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Try to import OpenCV
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logger.warning("OpenCV not available - camera features disabled")


@dataclass
class CameraCapture:
    """Represents a captured image frame."""

    image: np.ndarray
    timestamp: datetime
    frame_number: int
    metadata: dict

    @property
    def shape(self) -> Tuple[int, ...]:
        """Get image dimensions."""
        return self.image.shape

    @property
    def width(self) -> int:
        """Get image width."""
        return self.image.shape[1]

    @property
    def height(self) -> int:
        """Get image height."""
        return self.image.shape[0]

    def save(self, path: str) -> bool:
        """
        Save the capture to file.

        Args:
            path: Output file path

        Returns:
            True if successful
        """
        if not CV2_AVAILABLE:
            logger.error("Cannot save - OpenCV not available")
            return False

        try:
            cv2.imwrite(path, self.image)
            logger.debug(f"Saved capture to {path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save capture: {e}")
            return False


class Camera:
    """
    Camera interface for Aero experiments.

    Provides:
    - Webcam capture
    - Video recording
    - Frame processing

    Example:
        camera = Camera(index=0)
        if camera.open():
            capture = camera.capture()
            capture.save("frame.png")
            camera.close()
    """

    def __init__(
        self,
        index: int = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
    ):
        """
        Initialize the camera.

        Args:
            index: Camera device index
            width: Capture width
            height: Capture height
            fps: Frames per second
        """
        self.index = index
        self.width = width
        self.height = height
        self.fps = fps

        self._cap = None
        self._frame_count = 0
        self._is_open = False

    @property
    def is_open(self) -> bool:
        """Check if camera is open."""
        return self._is_open

    def open(self) -> bool:
        """
        Open the camera device.

        Returns:
            True if successful
        """
        if not CV2_AVAILABLE:
            logger.error("OpenCV not available - cannot open camera")
            return False

        try:
            self._cap = cv2.VideoCapture(self.index)

            if not self._cap.isOpened():
                logger.error(f"Failed to open camera at index {self.index}")
                return False

            # Set resolution
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self._cap.set(cv2.CAP_PROP_FPS, self.fps)

            self._is_open = True
            self._frame_count = 0
            logger.info(f"Camera opened: index={self.index}, {self.width}x{self.height}")
            return True

        except Exception as e:
            logger.error(f"Error opening camera: {e}")
            return False

    def close(self) -> None:
        """Close the camera device."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

        self._is_open = False
        logger.info("Camera closed")

    def capture(self) -> Optional[CameraCapture]:
        """
        Capture a single frame.

        Returns:
            CameraCapture or None if failed
        """
        if not self._is_open or self._cap is None:
            logger.error("Camera not open")
            return None

        ret, frame = self._cap.read()

        if not ret:
            logger.error("Failed to capture frame")
            return None

        self._frame_count += 1

        return CameraCapture(
            image=frame,
            timestamp=datetime.now(),
            frame_number=self._frame_count,
            metadata={
                "camera_index": self.index,
                "width": frame.shape[1],
                "height": frame.shape[0],
            },
        )

    def capture_sequence(
        self,
        count: int,
        interval_ms: int = 0,
    ) -> list[CameraCapture]:
        """
        Capture a sequence of frames.

        Args:
            count: Number of frames to capture
            interval_ms: Delay between frames in milliseconds

        Returns:
            List of CameraCapture objects
        """
        if not CV2_AVAILABLE:
            return []

        captures = []

        for _ in range(count):
            cap = self.capture()
            if cap:
                captures.append(cap)

            if interval_ms > 0:
                cv2.waitKey(interval_ms)

        return captures

    def __enter__(self):
        """Context manager entry."""
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


def is_camera_available(index: int = 0) -> bool:
    """
    Check if a camera is available at the given index.

    Args:
        index: Camera device index

    Returns:
        True if camera is available
    """
    if not CV2_AVAILABLE:
        return False

    try:
        cap = cv2.VideoCapture(index)
        available = cap.isOpened()
        cap.release()
        return available
    except Exception:
        return False


def list_cameras(max_index: int = 10) -> list[int]:
    """
    List available camera indices.

    Args:
        max_index: Maximum index to check

    Returns:
        List of available camera indices
    """
    available = []

    for i in range(max_index):
        if is_camera_available(i):
            available.append(i)

    return available
