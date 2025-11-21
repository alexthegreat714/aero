"""
Synthetic experiment generation for Aero Agent.

Provides utilities for generating synthetic data for testing
experiments without physical hardware (webcam, sensors, etc.).
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SyntheticExperiment:
    """
    Synthetic experiment generator for testing.

    Generates fake sensor data, images, and flow fields
    for testing the experiment pipeline without hardware.
    """

    name: str = "synthetic"
    seed: Optional[int] = None
    _rng: np.random.Generator = field(default=None, repr=False)

    def __post_init__(self):
        """Initialize random generator."""
        self._rng = np.random.default_rng(self.seed)

    def generate_timeseries(
        self,
        length: int = 100,
        pattern: str = "sine",
        noise_level: float = 0.1,
        frequency: float = 1.0,
        amplitude: float = 1.0,
        offset: float = 0.0,
    ) -> Tuple[np.ndarray, List[datetime]]:
        """
        Generate synthetic time series data.

        Args:
            length: Number of data points
            pattern: Pattern type (sine, cosine, linear, exponential, step, random)
            noise_level: Amount of noise to add (fraction of amplitude)
            frequency: Frequency for periodic patterns
            amplitude: Amplitude of signal
            offset: DC offset

        Returns:
            Tuple of (values array, timestamps list)
        """
        t = np.linspace(0, 2 * np.pi * frequency, length)

        if pattern == "sine":
            values = amplitude * np.sin(t) + offset
        elif pattern == "cosine":
            values = amplitude * np.cos(t) + offset
        elif pattern == "linear":
            values = amplitude * np.linspace(0, 1, length) + offset
        elif pattern == "exponential":
            values = amplitude * np.exp(-t / (2 * np.pi * frequency)) + offset
        elif pattern == "step":
            values = np.zeros(length) + offset
            values[length // 2:] = amplitude
        elif pattern == "sawtooth":
            values = amplitude * (t % (2 * np.pi)) / (2 * np.pi) + offset
        elif pattern == "random":
            values = amplitude * self._rng.random(length) + offset
        else:
            values = np.zeros(length) + offset

        # Add noise
        if noise_level > 0:
            noise = self._rng.normal(0, noise_level * amplitude, length)
            values = values + noise

        # Generate timestamps
        base_time = datetime.now()
        timestamps = [base_time + timedelta(seconds=i * 0.1) for i in range(length)]

        return values, timestamps

    def generate_image(
        self,
        width: int = 640,
        height: int = 480,
        pattern: str = "gradient",
        channels: int = 3,
    ) -> np.ndarray:
        """
        Generate a synthetic test image.

        Args:
            width: Image width
            height: Image height
            pattern: Pattern type (gradient, checkerboard, circle, noise, solid)
            channels: Number of color channels (1=grayscale, 3=color)

        Returns:
            Image as numpy array (height, width) or (height, width, 3)
        """
        if pattern == "gradient":
            # Horizontal gradient
            x = np.linspace(0, 255, width)
            image = np.tile(x, (height, 1))

        elif pattern == "gradient_v":
            # Vertical gradient
            y = np.linspace(0, 255, height)
            image = np.tile(y.reshape(-1, 1), (1, width))

        elif pattern == "checkerboard":
            # Checkerboard pattern
            block_size = 32
            x = np.arange(width) // block_size
            y = np.arange(height) // block_size
            xv, yv = np.meshgrid(x, y)
            image = ((xv + yv) % 2) * 255

        elif pattern == "circle":
            # Circle in center
            y, x = np.ogrid[:height, :width]
            cx, cy = width // 2, height // 2
            r = min(width, height) // 4
            mask = ((x - cx) ** 2 + (y - cy) ** 2) <= r ** 2
            image = mask.astype(np.float64) * 255

        elif pattern == "noise":
            # Random noise
            image = self._rng.random((height, width)) * 255

        elif pattern == "solid":
            # Solid gray
            image = np.ones((height, width)) * 128

        elif pattern == "bars":
            # Vertical bars
            bar_width = 20
            x = np.arange(width) // bar_width
            image = (x % 2) * 255
            image = np.tile(image, (height, 1))

        else:
            image = np.zeros((height, width))

        image = image.astype(np.uint8)

        # Convert to color if needed
        if channels == 3:
            image = np.stack([image, image, image], axis=2)

        return image

    def generate_image_sequence(
        self,
        num_frames: int = 10,
        width: int = 640,
        height: int = 480,
        motion_type: str = "translate",
        motion_speed: float = 5.0,
    ) -> List[np.ndarray]:
        """
        Generate a sequence of images with simulated motion.

        Args:
            num_frames: Number of frames to generate
            width: Image width
            height: Image height
            motion_type: Type of motion (translate, rotate, scale, oscillate)
            motion_speed: Speed of motion in pixels/frame

        Returns:
            List of image arrays
        """
        frames = []

        # Create base pattern (circle)
        base = self.generate_image(width, height, "circle", channels=3)

        for i in range(num_frames):
            frame = np.zeros((height, width, 3), dtype=np.uint8)

            if motion_type == "translate":
                # Horizontal translation
                offset = int(i * motion_speed) % width
                frame[:, offset:] = base[:, :width - offset]
                frame[:, :offset] = base[:, width - offset:]

            elif motion_type == "oscillate":
                # Oscillating motion
                offset = int(motion_speed * np.sin(i * 0.5) * 10)
                if offset >= 0:
                    frame[:, offset:] = base[:, :width - offset]
                else:
                    frame[:, :width + offset] = base[:, -offset:]

            elif motion_type == "grow":
                # Growing circle
                scale = 1.0 + i * 0.1
                cy, cx = height // 2, width // 2
                r = int(min(width, height) // 4 * scale)
                y, x = np.ogrid[:height, :width]
                mask = ((x - cx) ** 2 + (y - cy) ** 2) <= r ** 2
                frame[mask] = 255

            else:
                frame = base.copy()

            frames.append(frame)

        return frames

    def generate_flow_field(
        self,
        width: int = 64,
        height: int = 64,
        flow_type: str = "uniform",
        magnitude: float = 5.0,
    ) -> np.ndarray:
        """
        Generate a synthetic optical flow field.

        Args:
            width: Field width
            height: Field height
            flow_type: Type of flow (uniform, radial, vortex, random)
            magnitude: Flow magnitude

        Returns:
            Flow field array (height, width, 2) with (u, v) components
        """
        flow = np.zeros((height, width, 2), dtype=np.float32)

        if flow_type == "uniform":
            # Uniform rightward flow
            flow[:, :, 0] = magnitude
            flow[:, :, 1] = 0

        elif flow_type == "radial":
            # Radial outward flow from center
            cy, cx = height // 2, width // 2
            y, x = np.mgrid[:height, :width]
            dx = x - cx
            dy = y - cy
            r = np.sqrt(dx ** 2 + dy ** 2) + 1e-6
            flow[:, :, 0] = magnitude * dx / r
            flow[:, :, 1] = magnitude * dy / r

        elif flow_type == "vortex":
            # Rotational/vortex flow
            cy, cx = height // 2, width // 2
            y, x = np.mgrid[:height, :width]
            dx = x - cx
            dy = y - cy
            r = np.sqrt(dx ** 2 + dy ** 2) + 1e-6
            flow[:, :, 0] = -magnitude * dy / r
            flow[:, :, 1] = magnitude * dx / r

        elif flow_type == "shear":
            # Shear flow
            y = np.arange(height).reshape(-1, 1)
            flow[:, :, 0] = magnitude * (y / height - 0.5)

        elif flow_type == "random":
            # Random flow
            flow[:, :, 0] = self._rng.uniform(-magnitude, magnitude, (height, width))
            flow[:, :, 1] = self._rng.uniform(-magnitude, magnitude, (height, width))

        else:
            flow[:, :, 0] = magnitude

        return flow


# =============================================================================
# Module-level convenience functions
# =============================================================================


def generate_synthetic_flow(
    width: int = 64,
    height: int = 64,
    flow_type: str = "uniform",
    magnitude: float = 5.0,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Generate synthetic optical flow data for testing.

    Args:
        width: Field width
        height: Field height
        flow_type: Type of flow (uniform, radial, vortex, shear, random)
        magnitude: Flow magnitude
        seed: Random seed for reproducibility

    Returns:
        Dictionary with:
        - flow: Flow field array (height, width, 2)
        - magnitude_field: Magnitude at each point
        - mean_magnitude: Average magnitude
        - flow_type: Type of flow generated
    """
    exp = SyntheticExperiment(seed=seed)
    flow = exp.generate_flow_field(width, height, flow_type, magnitude)

    mag = np.sqrt(flow[:, :, 0] ** 2 + flow[:, :, 1] ** 2)

    return {
        "flow": flow,
        "magnitude_field": mag,
        "mean_magnitude": float(np.mean(mag)),
        "max_magnitude": float(np.max(mag)),
        "flow_type": flow_type,
        "shape": (height, width),
    }


def generate_synthetic_timeseries(
    length: int = 100,
    pattern: str = "sine",
    noise_level: float = 0.1,
    frequency: float = 1.0,
    amplitude: float = 1.0,
    offset: float = 0.0,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Generate synthetic time series data for testing.

    Args:
        length: Number of data points
        pattern: Pattern type (sine, cosine, linear, exponential, step, random)
        noise_level: Amount of noise (fraction of amplitude)
        frequency: Frequency for periodic patterns
        amplitude: Signal amplitude
        offset: DC offset
        seed: Random seed

    Returns:
        Dictionary with:
        - values: Data values
        - timestamps: Timestamp list
        - pattern: Pattern type
        - statistics: Mean, std, min, max
    """
    exp = SyntheticExperiment(seed=seed)
    values, timestamps = exp.generate_timeseries(
        length, pattern, noise_level, frequency, amplitude, offset
    )

    return {
        "values": values,
        "timestamps": timestamps,
        "pattern": pattern,
        "length": length,
        "statistics": {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        },
    }


def generate_test_image(
    width: int = 640,
    height: int = 480,
    pattern: str = "gradient",
    channels: int = 3,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate a test image for experiments.

    Args:
        width: Image width
        height: Image height
        pattern: Pattern type (gradient, checkerboard, circle, noise, solid, bars)
        channels: Number of color channels (1 or 3)
        seed: Random seed

    Returns:
        Image array
    """
    exp = SyntheticExperiment(seed=seed)
    return exp.generate_image(width, height, pattern, channels)


def generate_test_image_sequence(
    num_frames: int = 10,
    width: int = 640,
    height: int = 480,
    motion_type: str = "translate",
    motion_speed: float = 5.0,
    seed: Optional[int] = None,
) -> List[np.ndarray]:
    """
    Generate a test image sequence with motion.

    Args:
        num_frames: Number of frames
        width: Image width
        height: Image height
        motion_type: Type of motion (translate, oscillate, grow)
        motion_speed: Speed of motion

    Returns:
        List of image arrays
    """
    exp = SyntheticExperiment(seed=seed)
    return exp.generate_image_sequence(num_frames, width, height, motion_type, motion_speed)
