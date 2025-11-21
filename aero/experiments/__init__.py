"""
Experiments module for Aero Agent.

Provides utilities for physical experiments including:
- Camera capture and control
- Sensor data acquisition
- Data analysis
- Synthetic experiments for testing
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
import json

from aero.experiments.camera import (
    Camera,
    CameraCapture,
    is_camera_available,
    is_webcam_available,
    capture_frame,
    capture_sequence,
    list_cameras,
    CV2_AVAILABLE,
)
from aero.experiments.sensors import (
    SensorReader,
    SensorData,
    BaseSensor,
    MockSensor,
    FileSensor,
    read_csv_timeseries,
    read_json_timeseries,
)
from aero.experiments.analyzer import (
    DataAnalyzer,
    AnalysisResult,
    analyze_frame,
    compute_optical_flow,
    detect_motion,
    compute_histogram,
    detect_edges,
)
from aero.experiments.synthetic import (
    SyntheticExperiment,
    generate_synthetic_flow,
    generate_synthetic_timeseries,
    generate_test_image,
)


class ExperimentType(str, Enum):
    """Types of experiments supported."""

    CAMERA_SNAPSHOT = "camera_snapshot"
    CAMERA_FLOW_SEQUENCE = "camera_flow_sequence"
    VIDEO_FILE_ANALYSIS = "video_file_analysis"
    IMAGE_SEQUENCE_ANALYSIS = "image_sequence_analysis"
    SENSOR_TIMESERIES = "sensor_timeseries"
    SYNTHETIC_FLOW = "synthetic_flow"
    SYNTHETIC_TIMESERIES = "synthetic_timeseries"


@dataclass
class ExperimentResult:
    """
    Unified result container for all experiment types.

    Attributes:
        experiment_type: Type of experiment performed
        data: Primary result data (numpy arrays, values, etc.)
        metadata: Additional information about the experiment
        timestamp: When the experiment was performed
        success: Whether the experiment completed successfully
        error: Error message if experiment failed
    """

    experiment_type: str
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    success: bool = True
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (JSON-serializable)."""
        # Convert numpy arrays to lists
        data_serializable = {}
        for key, value in self.data.items():
            if hasattr(value, "tolist"):
                data_serializable[key] = value.tolist()
            elif isinstance(value, datetime):
                data_serializable[key] = value.isoformat()
            else:
                data_serializable[key] = value

        return {
            "experiment_type": self.experiment_type,
            "data": data_serializable,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
            "success": self.success,
            "error": self.error,
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2, default=str)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExperimentResult":
        """Create from dictionary."""
        timestamp = d.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        elif timestamp is None:
            timestamp = datetime.now()

        return cls(
            experiment_type=d.get("experiment_type", "unknown"),
            data=d.get("data", {}),
            metadata=d.get("metadata", {}),
            timestamp=timestamp,
            success=d.get("success", True),
            error=d.get("error"),
        )

    @classmethod
    def failure(cls, experiment_type: str, error: str) -> "ExperimentResult":
        """Create a failure result."""
        return cls(
            experiment_type=experiment_type,
            success=False,
            error=error,
        )


__all__ = [
    # Result container
    "ExperimentResult",
    "ExperimentType",
    # Camera
    "Camera",
    "CameraCapture",
    "is_camera_available",
    "is_webcam_available",
    "capture_frame",
    "capture_sequence",
    "list_cameras",
    "CV2_AVAILABLE",
    # Sensors
    "SensorReader",
    "SensorData",
    "BaseSensor",
    "MockSensor",
    "FileSensor",
    "read_csv_timeseries",
    "read_json_timeseries",
    # Analyzer
    "DataAnalyzer",
    "AnalysisResult",
    "analyze_frame",
    "compute_optical_flow",
    "detect_motion",
    "compute_histogram",
    "detect_edges",
    # Synthetic
    "SyntheticExperiment",
    "generate_synthetic_flow",
    "generate_synthetic_timeseries",
    "generate_test_image",
]
