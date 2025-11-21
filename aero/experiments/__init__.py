"""
Experiments module for Aero Agent.

Provides utilities for physical experiments including:
- Camera capture and control
- Sensor data acquisition
- Data analysis
"""

from aero.experiments.camera import Camera, CameraCapture
from aero.experiments.sensors import SensorReader, SensorData
from aero.experiments.analyzer import DataAnalyzer

__all__ = [
    "Camera",
    "CameraCapture",
    "SensorReader",
    "SensorData",
    "DataAnalyzer",
]
