"""
Time series utilities for Aero Agent data lake.

Provides functions for packing, unpacking, and analyzing time series data.
"""

import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def pack_timeseries(
    t: List[float],
    values: List[float],
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Pack time series data into a storable format.

    Args:
        t: Time values
        values: Data values
        metadata: Optional metadata dictionary

    Returns:
        Dictionary with packed time series
    """
    return {
        "t": t if isinstance(t, list) else list(t),
        "values": values if isinstance(values, list) else list(values),
        "metadata": metadata or {},
        "length": len(values),
    }


def unpack_timeseries(record: Dict[str, Any]) -> Tuple[List[float], List[float], Dict[str, Any]]:
    """
    Unpack time series data from stored format.

    Args:
        record: Stored time series record

    Returns:
        Tuple of (t, values, metadata)
    """
    t = record.get("t", [])
    values = record.get("values", [])
    metadata = record.get("metadata", {})

    # Handle JSON strings
    if isinstance(t, str):
        t = json.loads(t)
    if isinstance(values, str):
        values = json.loads(values)
    if isinstance(metadata, str):
        metadata = json.loads(metadata)

    return t, values, metadata


def compute_basic_ts_stats(
    t: List[float],
    values: List[float],
) -> Dict[str, Any]:
    """
    Compute basic statistics for a time series.

    Args:
        t: Time values
        values: Data values

    Returns:
        Dictionary with statistics including:
        - mean, std, min, max
        - trend direction
        - duration
        - sample count
    """
    if not values:
        return {
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "trend": "flat",
            "duration": 0.0,
            "count": 0,
        }

    values_arr = np.array(values)

    stats = {
        "mean": float(np.mean(values_arr)),
        "std": float(np.std(values_arr)),
        "min": float(np.min(values_arr)),
        "max": float(np.max(values_arr)),
        "count": len(values),
    }

    # Compute duration
    if t and len(t) >= 2:
        stats["duration"] = float(t[-1] - t[0])
    else:
        stats["duration"] = 0.0

    # Compute trend direction
    if len(values) >= 2:
        first_half_mean = np.mean(values_arr[:len(values_arr) // 2])
        second_half_mean = np.mean(values_arr[len(values_arr) // 2:])
        diff = second_half_mean - first_half_mean

        if abs(diff) < stats["std"] * 0.1:
            stats["trend"] = "flat"
        elif diff > 0:
            stats["trend"] = "increasing"
        else:
            stats["trend"] = "decreasing"
    else:
        stats["trend"] = "flat"

    # Additional metrics
    if len(values) >= 2:
        # Rate of change
        if stats["duration"] > 0:
            stats["rate_of_change"] = (values[-1] - values[0]) / stats["duration"]
        else:
            stats["rate_of_change"] = 0.0

        # Coefficient of variation
        if stats["mean"] != 0:
            stats["cv"] = stats["std"] / abs(stats["mean"])
        else:
            stats["cv"] = 0.0

    return stats


def resample_timeseries(
    t: List[float],
    values: List[float],
    num_points: int,
) -> Tuple[List[float], List[float]]:
    """
    Resample a time series to a fixed number of points.

    Args:
        t: Original time values
        values: Original data values
        num_points: Target number of points

    Returns:
        Tuple of (new_t, new_values)
    """
    if len(t) < 2 or num_points < 2:
        return t, values

    t_arr = np.array(t)
    values_arr = np.array(values)

    new_t = np.linspace(t_arr[0], t_arr[-1], num_points)
    new_values = np.interp(new_t, t_arr, values_arr)

    return new_t.tolist(), new_values.tolist()


def detect_anomalies(
    values: List[float],
    threshold: float = 3.0,
) -> List[int]:
    """
    Detect anomalies in a time series using z-score.

    Args:
        values: Data values
        threshold: Z-score threshold for anomaly detection

    Returns:
        List of indices where anomalies were detected
    """
    if len(values) < 3:
        return []

    values_arr = np.array(values)
    mean = np.mean(values_arr)
    std = np.std(values_arr)

    if std == 0:
        return []

    z_scores = np.abs((values_arr - mean) / std)
    anomaly_indices = np.where(z_scores > threshold)[0]

    return anomaly_indices.tolist()


def compute_fft_features(
    values: List[float],
    sample_rate: float = 1.0,
) -> Dict[str, Any]:
    """
    Compute FFT-based features for a time series.

    Args:
        values: Data values
        sample_rate: Sampling rate in Hz

    Returns:
        Dictionary with FFT features
    """
    if len(values) < 4:
        return {
            "dominant_frequency": 0.0,
            "dominant_magnitude": 0.0,
            "spectral_centroid": 0.0,
        }

    values_arr = np.array(values)
    n = len(values_arr)

    # Compute FFT
    fft_result = np.fft.fft(values_arr)
    frequencies = np.fft.fftfreq(n, d=1.0 / sample_rate)

    # Get positive frequencies only
    positive_mask = frequencies > 0
    positive_freqs = frequencies[positive_mask]
    positive_mags = np.abs(fft_result[positive_mask])

    if len(positive_mags) == 0:
        return {
            "dominant_frequency": 0.0,
            "dominant_magnitude": 0.0,
            "spectral_centroid": 0.0,
        }

    # Find dominant frequency
    dominant_idx = np.argmax(positive_mags)
    dominant_freq = positive_freqs[dominant_idx]
    dominant_mag = positive_mags[dominant_idx]

    # Compute spectral centroid
    total_mag = np.sum(positive_mags)
    if total_mag > 0:
        spectral_centroid = np.sum(positive_freqs * positive_mags) / total_mag
    else:
        spectral_centroid = 0.0

    return {
        "dominant_frequency": float(dominant_freq),
        "dominant_magnitude": float(dominant_mag),
        "spectral_centroid": float(spectral_centroid),
    }


def smooth_timeseries(
    values: List[float],
    window_size: int = 5,
) -> List[float]:
    """
    Smooth a time series using moving average.

    Args:
        values: Data values
        window_size: Size of smoothing window

    Returns:
        Smoothed values
    """
    if len(values) < window_size:
        return values

    values_arr = np.array(values)
    kernel = np.ones(window_size) / window_size
    smoothed = np.convolve(values_arr, kernel, mode='same')

    return smoothed.tolist()
