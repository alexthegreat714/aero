"""
Data analysis utilities for Aero experiments.

Provides tools for analyzing sensor and image data,
including frame analysis, optical flow, and motion detection.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

# Try to import OpenCV
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logger.warning("OpenCV not available - frame analysis features limited")


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


# =============================================================================
# Module-level frame analysis functions
# =============================================================================


def analyze_frame(image: np.ndarray) -> Dict[str, Any]:
    """
    Comprehensive frame analysis.

    Extracts statistics, histogram, and edge information
    from a single frame.

    Args:
        image: Input image (grayscale or color)

    Returns:
        Dictionary with analysis results including:
        - shape: Image dimensions
        - statistics: Mean, std, min, max per channel
        - histogram: Intensity distribution
        - edges: Edge statistics
    """
    result: Dict[str, Any] = {
        "shape": image.shape,
        "dtype": str(image.dtype),
    }

    # Convert to grayscale for analysis if color
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if CV2_AVAILABLE else np.mean(image, axis=2)
        result["color"] = True
        result["channels"] = image.shape[2]

        # Per-channel statistics
        result["mean_per_channel"] = [float(np.mean(image[:, :, c])) for c in range(image.shape[2])]
        result["std_per_channel"] = [float(np.std(image[:, :, c])) for c in range(image.shape[2])]
    else:
        gray = image
        result["color"] = False
        result["channels"] = 1

    # Overall statistics
    result["mean"] = float(np.mean(gray))
    result["std"] = float(np.std(gray))
    result["min"] = int(np.min(gray))
    result["max"] = int(np.max(gray))

    # Histogram
    hist, _ = np.histogram(gray.flatten(), bins=256, range=(0, 256))
    result["histogram_peak"] = int(np.argmax(hist))
    result["histogram_spread"] = float(np.std(hist))

    # Edge detection if OpenCV available
    if CV2_AVAILABLE:
        edges = cv2.Canny(gray.astype(np.uint8), 50, 150)
        result["edge_pixels"] = int(np.sum(edges > 0))
        result["edge_density"] = float(np.sum(edges > 0) / edges.size)

    return result


def compute_optical_flow(
    frame1: np.ndarray,
    frame2: np.ndarray,
    pyr_scale: float = 0.5,
    levels: int = 3,
    winsize: int = 15,
    iterations: int = 3,
    poly_n: int = 5,
    poly_sigma: float = 1.2,
) -> Optional[Dict[str, Any]]:
    """
    Compute dense optical flow between two frames using Farneback method.

    Args:
        frame1: First frame (grayscale or color)
        frame2: Second frame (grayscale or color)
        pyr_scale: Scale factor for pyramid (default 0.5)
        levels: Number of pyramid levels
        winsize: Averaging window size
        iterations: Number of iterations per level
        poly_n: Size of pixel neighborhood
        poly_sigma: Standard deviation of Gaussian for polynomial expansion

    Returns:
        Dictionary with flow results:
        - flow: Dense optical flow array (height, width, 2)
        - magnitude: Flow magnitude statistics
        - angle: Flow direction statistics
        - motion_detected: Boolean indicating significant motion
        Returns None if OpenCV not available
    """
    if not CV2_AVAILABLE:
        logger.warning("OpenCV not available - cannot compute optical flow")
        return None

    # Convert to grayscale if needed
    if len(frame1.shape) == 3:
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    else:
        gray1 = frame1

    if len(frame2.shape) == 3:
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    else:
        gray2 = frame2

    # Compute flow
    flow = cv2.calcOpticalFlowFarneback(
        gray1.astype(np.uint8),
        gray2.astype(np.uint8),
        None,
        pyr_scale,
        levels,
        winsize,
        iterations,
        poly_n,
        poly_sigma,
        0,
    )

    # Compute magnitude and angle
    magnitude, angle = cv2.cartToPolar(flow[:, :, 0], flow[:, :, 1])

    # Statistics
    result = {
        "flow": flow,
        "magnitude": {
            "mean": float(np.mean(magnitude)),
            "max": float(np.max(magnitude)),
            "std": float(np.std(magnitude)),
        },
        "angle": {
            "mean": float(np.mean(angle)),
            "std": float(np.std(angle)),
        },
        "motion_detected": float(np.mean(magnitude)) > 1.0,
    }

    return result


def detect_motion(
    frame1: np.ndarray,
    frame2: np.ndarray,
    threshold: int = 25,
    min_area: int = 500,
) -> Dict[str, Any]:
    """
    Detect motion between two frames using frame differencing.

    Args:
        frame1: First frame
        frame2: Second frame
        threshold: Pixel difference threshold
        min_area: Minimum contour area to consider as motion

    Returns:
        Dictionary with:
        - motion_detected: Boolean
        - motion_area: Total area of motion regions
        - motion_regions: Number of distinct motion regions
        - motion_percentage: Percentage of frame with motion
        - diff_mean: Mean absolute difference
    """
    # Convert to grayscale if needed
    if len(frame1.shape) == 3:
        if CV2_AVAILABLE:
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        else:
            gray1 = np.mean(frame1, axis=2).astype(np.uint8)
            gray2 = np.mean(frame2, axis=2).astype(np.uint8)
    else:
        gray1 = frame1.astype(np.uint8)
        gray2 = frame2.astype(np.uint8)

    # Compute difference
    diff = np.abs(gray1.astype(np.int16) - gray2.astype(np.int16))
    diff_mean = float(np.mean(diff))

    result = {
        "motion_detected": False,
        "motion_area": 0,
        "motion_regions": 0,
        "motion_percentage": 0.0,
        "diff_mean": diff_mean,
    }

    if CV2_AVAILABLE:
        # Threshold the difference
        _, thresh = cv2.threshold(diff.astype(np.uint8), threshold, 255, cv2.THRESH_BINARY)

        # Dilate to fill gaps
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.dilate(thresh, kernel, iterations=2)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter by area
        motion_area = 0
        motion_regions = 0

        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= min_area:
                motion_area += area
                motion_regions += 1

        total_area = gray1.shape[0] * gray1.shape[1]

        result["motion_detected"] = motion_regions > 0
        result["motion_area"] = motion_area
        result["motion_regions"] = motion_regions
        result["motion_percentage"] = 100.0 * motion_area / total_area

    else:
        # Simple threshold-based detection
        motion_pixels = np.sum(diff > threshold)
        total_pixels = diff.size

        result["motion_detected"] = motion_pixels > min_area
        result["motion_area"] = int(motion_pixels)
        result["motion_percentage"] = 100.0 * motion_pixels / total_pixels

    return result


def compute_histogram(
    image: np.ndarray,
    bins: int = 256,
    normalize: bool = True,
) -> Dict[str, Any]:
    """
    Compute image histogram.

    Args:
        image: Input image (grayscale or color)
        bins: Number of histogram bins
        normalize: Whether to normalize histogram

    Returns:
        Dictionary with:
        - histogram: Histogram values (per channel for color)
        - bins: Bin edges
        - peak_value: Most common intensity
        - mean_intensity: Mean intensity
    """
    result: Dict[str, Any] = {}

    if len(image.shape) == 3 and CV2_AVAILABLE:
        # Color image - compute per channel
        histograms = []
        peaks = []

        for c in range(image.shape[2]):
            hist = cv2.calcHist([image], [c], None, [bins], [0, 256])
            if normalize:
                hist = hist / hist.sum()
            histograms.append(hist.flatten().tolist())
            peaks.append(int(np.argmax(hist)))

        result["histogram"] = histograms
        result["peak_per_channel"] = peaks
        result["mean_per_channel"] = [float(np.mean(image[:, :, c])) for c in range(image.shape[2])]

    else:
        # Grayscale
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image

        hist, bin_edges = np.histogram(gray.flatten(), bins=bins, range=(0, 256))
        if normalize:
            hist = hist / hist.sum()

        result["histogram"] = hist.tolist()
        result["bins"] = bin_edges.tolist()
        result["peak_value"] = int(np.argmax(hist))
        result["mean_intensity"] = float(np.mean(gray))

    return result


def detect_edges(
    image: np.ndarray,
    low_threshold: int = 50,
    high_threshold: int = 150,
    aperture_size: int = 3,
) -> Dict[str, Any]:
    """
    Detect edges using Canny edge detector.

    Args:
        image: Input image
        low_threshold: Lower threshold for hysteresis
        high_threshold: Upper threshold for hysteresis
        aperture_size: Aperture size for Sobel operator

    Returns:
        Dictionary with:
        - edges: Binary edge image (if OpenCV available)
        - edge_count: Number of edge pixels
        - edge_density: Proportion of edge pixels
        - edge_mean_position: Mean position of edges
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        if CV2_AVAILABLE:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = np.mean(image, axis=2).astype(np.uint8)
    else:
        gray = image.astype(np.uint8)

    result: Dict[str, Any] = {}

    if CV2_AVAILABLE:
        edges = cv2.Canny(gray, low_threshold, high_threshold, apertureSize=aperture_size)

        edge_pixels = np.where(edges > 0)
        edge_count = len(edge_pixels[0])

        result["edges"] = edges
        result["edge_count"] = edge_count
        result["edge_density"] = edge_count / edges.size

        if edge_count > 0:
            result["edge_mean_position"] = [
                float(np.mean(edge_pixels[0])),
                float(np.mean(edge_pixels[1])),
            ]
        else:
            result["edge_mean_position"] = [0.0, 0.0]

    else:
        # Simple gradient-based edge detection
        gy, gx = np.gradient(gray.astype(float))
        gradient_mag = np.sqrt(gx ** 2 + gy ** 2)
        edges = (gradient_mag > low_threshold).astype(np.uint8) * 255

        edge_count = np.sum(edges > 0)

        result["edges"] = edges
        result["edge_count"] = int(edge_count)
        result["edge_density"] = float(edge_count / edges.size)

    return result


def compute_frame_difference_stats(
    frames: List[np.ndarray],
) -> Dict[str, Any]:
    """
    Compute statistics on frame-to-frame differences.

    Args:
        frames: List of frames (at least 2)

    Returns:
        Dictionary with:
        - mean_diff: Mean difference between consecutive frames
        - max_diff: Maximum difference
        - motion_frames: Number of frames with significant motion
        - total_motion: Sum of all motion
    """
    if len(frames) < 2:
        return {
            "mean_diff": 0.0,
            "max_diff": 0.0,
            "motion_frames": 0,
            "total_motion": 0.0,
        }

    diffs = []
    motion_count = 0

    for i in range(1, len(frames)):
        result = detect_motion(frames[i - 1], frames[i])
        diffs.append(result["diff_mean"])

        if result["motion_detected"]:
            motion_count += 1

    return {
        "mean_diff": float(np.mean(diffs)),
        "max_diff": float(np.max(diffs)),
        "motion_frames": motion_count,
        "total_motion": float(np.sum(diffs)),
    }
