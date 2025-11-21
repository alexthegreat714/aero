"""
Data analysis utilities for Aero experiments.

Provides tools for analyzing sensor and image data.
"""

import logging
from typing import Any, Optional, Tuple
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AnalysisResult:
    """Container for analysis results."""

    name: str
    value: Any
    metadata: dict

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "value": self.value,
            "metadata": self.metadata,
        }


class DataAnalyzer:
    """
    Data analysis utilities for experiments.

    Provides:
    - Statistical analysis
    - Signal processing
    - Image analysis (basic)

    Example:
        analyzer = DataAnalyzer()
        stats = analyzer.compute_statistics(data)
        print(f"Mean: {stats['mean']}, Std: {stats['std']}")
    """

    def __init__(self):
        """Initialize the analyzer."""
        self._logger = logging.getLogger("aero.analyzer")

    def compute_statistics(self, data: np.ndarray) -> dict:
        """
        Compute basic statistics for data.

        Args:
            data: Input data array

        Returns:
            Dictionary with statistics
        """
        data = np.asarray(data)

        return {
            "count": len(data),
            "mean": float(np.mean(data)),
            "std": float(np.std(data)),
            "min": float(np.min(data)),
            "max": float(np.max(data)),
            "median": float(np.median(data)),
            "q25": float(np.percentile(data, 25)),
            "q75": float(np.percentile(data, 75)),
        }

    def moving_average(
        self,
        data: np.ndarray,
        window_size: int = 5,
    ) -> np.ndarray:
        """
        Compute moving average.

        Args:
            data: Input data
            window_size: Size of moving window

        Returns:
            Smoothed data
        """
        data = np.asarray(data)
        kernel = np.ones(window_size) / window_size
        return np.convolve(data, kernel, mode="valid")

    def detect_peaks(
        self,
        data: np.ndarray,
        threshold: Optional[float] = None,
        min_distance: int = 1,
    ) -> np.ndarray:
        """
        Detect peaks in data.

        Args:
            data: Input data
            threshold: Minimum peak height (default: mean + std)
            min_distance: Minimum distance between peaks

        Returns:
            Array of peak indices
        """
        data = np.asarray(data)

        if threshold is None:
            threshold = np.mean(data) + np.std(data)

        # Simple peak detection
        peaks = []

        for i in range(1, len(data) - 1):
            if data[i] > data[i - 1] and data[i] > data[i + 1]:
                if data[i] >= threshold:
                    # Check minimum distance
                    if not peaks or (i - peaks[-1]) >= min_distance:
                        peaks.append(i)

        return np.array(peaks)

    def compute_fft(
        self,
        data: np.ndarray,
        sample_rate: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute FFT of data.

        Args:
            data: Input data
            sample_rate: Sampling rate in Hz

        Returns:
            Tuple of (frequencies, magnitudes)
        """
        data = np.asarray(data)
        n = len(data)

        fft_result = np.fft.fft(data)
        frequencies = np.fft.fftfreq(n, d=1.0 / sample_rate)

        # Get positive frequencies only
        positive_mask = frequencies >= 0
        frequencies = frequencies[positive_mask]
        magnitudes = np.abs(fft_result[positive_mask])

        return frequencies, magnitudes

    def compute_correlation(
        self,
        data1: np.ndarray,
        data2: np.ndarray,
    ) -> float:
        """
        Compute Pearson correlation coefficient.

        Args:
            data1: First data array
            data2: Second data array

        Returns:
            Correlation coefficient
        """
        return float(np.corrcoef(data1, data2)[0, 1])

    def interpolate(
        self,
        x: np.ndarray,
        y: np.ndarray,
        x_new: np.ndarray,
        method: str = "linear",
    ) -> np.ndarray:
        """
        Interpolate data to new x values.

        Args:
            x: Original x values
            y: Original y values
            x_new: New x values
            method: Interpolation method ("linear", "cubic")

        Returns:
            Interpolated y values
        """
        from scipy.interpolate import interp1d

        if method == "linear":
            kind = "linear"
        elif method == "cubic":
            kind = "cubic"
        else:
            kind = "linear"

        f = interp1d(x, y, kind=kind, fill_value="extrapolate")
        return f(x_new)

    def compute_derivative(
        self,
        data: np.ndarray,
        dx: float = 1.0,
    ) -> np.ndarray:
        """
        Compute numerical derivative.

        Args:
            data: Input data
            dx: Step size

        Returns:
            Derivative values
        """
        return np.gradient(data, dx)

    def compute_integral(
        self,
        data: np.ndarray,
        dx: float = 1.0,
    ) -> float:
        """
        Compute numerical integral using trapezoidal rule.

        Args:
            data: Input data
            dx: Step size

        Returns:
            Integral value
        """
        return float(np.trapz(data, dx=dx))

    def analyze_image(self, image: np.ndarray) -> dict:
        """
        Basic image analysis.

        Args:
            image: Input image array

        Returns:
            Dictionary with image statistics
        """
        if len(image.shape) == 3:
            # Color image
            return {
                "shape": image.shape,
                "dtype": str(image.dtype),
                "mean_rgb": [float(np.mean(image[:, :, c])) for c in range(3)],
                "std_rgb": [float(np.std(image[:, :, c])) for c in range(3)],
                "min": int(np.min(image)),
                "max": int(np.max(image)),
            }
        else:
            # Grayscale
            return {
                "shape": image.shape,
                "dtype": str(image.dtype),
                "mean": float(np.mean(image)),
                "std": float(np.std(image)),
                "min": int(np.min(image)),
                "max": int(np.max(image)),
            }

    def extract_profile(
        self,
        image: np.ndarray,
        start: Tuple[int, int],
        end: Tuple[int, int],
        num_points: int = 100,
    ) -> np.ndarray:
        """
        Extract intensity profile along a line in an image.

        Args:
            image: Input image
            start: Start point (x, y)
            end: End point (x, y)
            num_points: Number of sample points

        Returns:
            Array of intensity values
        """
        x = np.linspace(start[0], end[0], num_points).astype(int)
        y = np.linspace(start[1], end[1], num_points).astype(int)

        # Clip to image bounds
        x = np.clip(x, 0, image.shape[1] - 1)
        y = np.clip(y, 0, image.shape[0] - 1)

        if len(image.shape) == 3:
            # Return grayscale equivalent for color images
            return np.mean(image[y, x], axis=1)
        else:
            return image[y, x]
