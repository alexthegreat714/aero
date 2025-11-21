"""
Experiment API routes for Aero Agent.

Provides endpoints for running experiments including:
- Camera capture and optical flow
- Sensor time series
- Video/image analysis
- Synthetic experiments for testing
"""

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter()


# =============================================================================
# Request/Response Models
# =============================================================================


class CameraSnapshotRequest(BaseModel):
    """Request for camera snapshot experiment."""

    camera_index: int = Field(default=0, description="Camera device index")
    width: int = Field(default=640, ge=64, le=4096)
    height: int = Field(default=480, ge=64, le=4096)


class CameraFlowRequest(BaseModel):
    """Request for camera optical flow experiment."""

    camera_index: int = Field(default=0, description="Camera device index")
    num_frames: int = Field(default=10, ge=2, le=100)
    interval_ms: int = Field(default=100, ge=10, le=5000)
    width: int = Field(default=640, ge=64, le=4096)
    height: int = Field(default=480, ge=64, le=4096)


class VideoAnalysisRequest(BaseModel):
    """Request for video file analysis."""

    video_path: str = Field(..., description="Path to video file")
    max_frames: int = Field(default=100, ge=1, le=1000)
    skip_frames: int = Field(default=0, ge=0, le=100)


class SensorTimeseriesRequest(BaseModel):
    """Request for sensor timeseries experiment."""

    file_path: str = Field(..., description="Path to CSV or JSON file")
    value_column: Optional[str] = Field(default=None, description="Column name for values")
    time_column: Optional[str] = Field(default=None, description="Column name for timestamps")


class SyntheticFlowRequest(BaseModel):
    """Request for synthetic flow experiment."""

    width: int = Field(default=64, ge=16, le=512)
    height: int = Field(default=64, ge=16, le=512)
    flow_type: str = Field(default="uniform", description="Flow type: uniform, radial, vortex, shear, random")
    magnitude: float = Field(default=5.0, ge=0.0, le=100.0)
    seed: Optional[int] = Field(default=None, description="Random seed for reproducibility")


class SyntheticTimeseriesRequest(BaseModel):
    """Request for synthetic timeseries experiment."""

    length: int = Field(default=100, ge=10, le=10000)
    pattern: str = Field(default="sine", description="Pattern: sine, cosine, linear, exponential, step, random")
    noise_level: float = Field(default=0.1, ge=0.0, le=1.0)
    frequency: float = Field(default=1.0, ge=0.1, le=100.0)
    amplitude: float = Field(default=1.0, ge=0.0, le=1000.0)
    offset: float = Field(default=0.0)
    seed: Optional[int] = Field(default=None)


class ImageSequenceRequest(BaseModel):
    """Request for generating test image sequence."""

    num_frames: int = Field(default=10, ge=2, le=100)
    width: int = Field(default=640, ge=64, le=4096)
    height: int = Field(default=480, ge=64, le=4096)
    motion_type: str = Field(default="translate", description="Motion: translate, oscillate, grow")
    motion_speed: float = Field(default=5.0, ge=0.0, le=100.0)


# =============================================================================
# Status and Info Endpoints
# =============================================================================


@router.get("/")
async def experiments_status():
    """Get experiment system status."""
    # Check for OpenCV availability
    try:
        from aero.experiments import CV2_AVAILABLE, is_webcam_available
        webcam_available = is_webcam_available(0) if CV2_AVAILABLE else False
    except ImportError:
        CV2_AVAILABLE = False
        webcam_available = False

    return {
        "status": "ok",
        "module": "experiments",
        "opencv_available": CV2_AVAILABLE,
        "webcam_available": webcam_available,
        "experiment_types": [
            "camera_snapshot",
            "camera_flow_sequence",
            "video_file_analysis",
            "image_sequence_analysis",
            "sensor_timeseries",
            "synthetic_flow",
            "synthetic_timeseries",
        ],
        "message": "Experiment system ready",
    }


@router.get("/capabilities")
async def get_capabilities():
    """Get available experiment capabilities based on system."""
    capabilities = {
        "synthetic": True,  # Always available
        "sensor_file": True,  # Always available (file-based)
    }

    try:
        from aero.experiments import CV2_AVAILABLE, is_webcam_available
        capabilities["opencv"] = CV2_AVAILABLE
        capabilities["webcam"] = is_webcam_available(0) if CV2_AVAILABLE else False
        capabilities["video_analysis"] = CV2_AVAILABLE
        capabilities["optical_flow"] = CV2_AVAILABLE
    except ImportError:
        capabilities["opencv"] = False
        capabilities["webcam"] = False
        capabilities["video_analysis"] = False
        capabilities["optical_flow"] = False

    return {
        "status": "ok",
        "capabilities": capabilities,
    }


# =============================================================================
# Camera Endpoints
# =============================================================================


@router.post("/camera/snapshot")
async def camera_snapshot(request: CameraSnapshotRequest):
    """
    Capture a single frame from webcam.

    Returns frame analysis including statistics and edge detection.
    """
    try:
        from aero.experiments import (
            capture_frame,
            analyze_frame,
            is_webcam_available,
            ExperimentResult,
        )

        if not is_webcam_available(request.camera_index):
            return {
                "status": "error",
                "error": f"Webcam not available at index {request.camera_index}",
            }

        capture = capture_frame(
            index=request.camera_index,
            width=request.width,
            height=request.height,
        )

        if capture is None:
            return {
                "status": "error",
                "error": "Failed to capture frame",
            }

        analysis = analyze_frame(capture.image)

        return {
            "status": "ok",
            "experiment_type": "camera_snapshot",
            "shape": capture.shape,
            "analysis": analysis,
            "frame_number": capture.frame_number,
        }

    except Exception as e:
        logger.exception(f"Camera snapshot error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/camera/flow")
async def camera_flow_sequence(request: CameraFlowRequest):
    """
    Capture frame sequence and compute optical flow.

    Returns flow analysis between consecutive frames.
    """
    try:
        from aero.pipeline.planner import (
            ExperimentConfig,
            ExperimentType,
            execute_experiment,
        )

        config = ExperimentConfig(
            exp_type=ExperimentType.CAMERA_FLOW_SEQUENCE,
            hypothesis_id="api_request",
            source=str(request.camera_index),
            parameters={
                "num_frames": request.num_frames,
                "interval_ms": request.interval_ms,
                "width": request.width,
                "height": request.height,
            },
        )

        result = execute_experiment(config)

        return {
            "status": "ok" if result.success else "error",
            **result.to_dict(),
        }

    except Exception as e:
        logger.exception(f"Camera flow error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Video/Image Endpoints
# =============================================================================


@router.post("/video/analyze")
async def analyze_video(request: VideoAnalysisRequest):
    """
    Analyze a video file for motion and patterns.

    Returns frame-by-frame analysis and motion statistics.
    """
    try:
        from aero.pipeline.planner import (
            ExperimentConfig,
            ExperimentType,
            execute_experiment,
        )

        config = ExperimentConfig(
            exp_type=ExperimentType.VIDEO_FILE_ANALYSIS,
            hypothesis_id="api_request",
            source=request.video_path,
            parameters={
                "max_frames": request.max_frames,
                "skip_frames": request.skip_frames,
            },
        )

        result = execute_experiment(config)

        return {
            "status": "ok" if result.success else "error",
            **result.to_dict(),
        }

    except Exception as e:
        logger.exception(f"Video analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Sensor Endpoints
# =============================================================================


@router.post("/sensor/timeseries")
async def read_sensor_timeseries(request: SensorTimeseriesRequest):
    """
    Read and analyze sensor time series from a file.

    Supports CSV and JSON formats with auto-detection of columns.
    """
    try:
        from aero.pipeline.planner import (
            ExperimentConfig,
            ExperimentType,
            execute_experiment,
        )

        config = ExperimentConfig(
            exp_type=ExperimentType.SENSOR_TIMESERIES,
            hypothesis_id="api_request",
            source=request.file_path,
            parameters={
                "value_column": request.value_column,
                "time_column": request.time_column,
            },
        )

        result = execute_experiment(config)

        return {
            "status": "ok" if result.success else "error",
            **result.to_dict(),
        }

    except Exception as e:
        logger.exception(f"Sensor timeseries error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Synthetic Experiment Endpoints
# =============================================================================


@router.post("/synthetic/flow")
async def synthetic_flow(request: SyntheticFlowRequest):
    """
    Generate synthetic optical flow data for testing.

    Useful for testing flow analysis without webcam hardware.
    """
    try:
        from aero.pipeline.planner import (
            ExperimentConfig,
            ExperimentType,
            execute_experiment,
        )

        config = ExperimentConfig(
            exp_type=ExperimentType.SYNTHETIC_FLOW,
            hypothesis_id="synthetic_test",
            source="",
            parameters={
                "width": request.width,
                "height": request.height,
                "flow_type": request.flow_type,
                "magnitude": request.magnitude,
                "seed": request.seed,
            },
        )

        result = execute_experiment(config)

        return {
            "status": "ok" if result.success else "error",
            **result.to_dict(),
        }

    except Exception as e:
        logger.exception(f"Synthetic flow error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/synthetic/timeseries")
async def synthetic_timeseries(request: SyntheticTimeseriesRequest):
    """
    Generate synthetic time series data for testing.

    Supports various patterns: sine, cosine, linear, exponential, step, random.
    """
    try:
        from aero.pipeline.planner import (
            ExperimentConfig,
            ExperimentType,
            execute_experiment,
        )

        config = ExperimentConfig(
            exp_type=ExperimentType.SYNTHETIC_TIMESERIES,
            hypothesis_id="synthetic_test",
            source="",
            parameters={
                "length": request.length,
                "pattern": request.pattern,
                "noise_level": request.noise_level,
                "frequency": request.frequency,
                "amplitude": request.amplitude,
                "offset": request.offset,
                "seed": request.seed,
            },
        )

        result = execute_experiment(config)

        return {
            "status": "ok" if result.success else "error",
            **result.to_dict(),
        }

    except Exception as e:
        logger.exception(f"Synthetic timeseries error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/synthetic/image-sequence")
async def synthetic_image_sequence(request: ImageSequenceRequest):
    """
    Generate a synthetic image sequence with motion.

    Useful for testing motion detection and flow analysis.
    """
    try:
        from aero.experiments.synthetic import generate_test_image_sequence
        from aero.experiments.analyzer import compute_frame_difference_stats

        frames = generate_test_image_sequence(
            num_frames=request.num_frames,
            width=request.width,
            height=request.height,
            motion_type=request.motion_type,
            motion_speed=request.motion_speed,
        )

        # Compute motion statistics
        motion_stats = compute_frame_difference_stats(frames)

        return {
            "status": "ok",
            "experiment_type": "synthetic_image_sequence",
            "num_frames": len(frames),
            "frame_shape": frames[0].shape if frames else None,
            "motion_type": request.motion_type,
            "motion_statistics": motion_stats,
        }

    except Exception as e:
        logger.exception(f"Synthetic image sequence error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Analysis Endpoints
# =============================================================================


@router.post("/analyze/frame")
async def analyze_single_frame(image_path: str):
    """
    Analyze a single image file.

    Returns statistics, histogram, and edge information.
    """
    try:
        from aero.experiments import CV2_AVAILABLE

        if not CV2_AVAILABLE:
            return {
                "status": "error",
                "error": "OpenCV not available for image analysis",
            }

        import cv2
        from aero.experiments.analyzer import analyze_frame

        image = cv2.imread(image_path)

        if image is None:
            return {
                "status": "error",
                "error": f"Could not load image: {image_path}",
            }

        analysis = analyze_frame(image)

        return {
            "status": "ok",
            "path": image_path,
            "analysis": analysis,
        }

    except Exception as e:
        logger.exception(f"Frame analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/flow-types")
async def list_flow_types():
    """List available synthetic flow types."""
    return {
        "status": "ok",
        "flow_types": [
            {"name": "uniform", "description": "Uniform rightward flow"},
            {"name": "radial", "description": "Radial outward flow from center"},
            {"name": "vortex", "description": "Rotational vortex flow"},
            {"name": "shear", "description": "Horizontal shear flow"},
            {"name": "random", "description": "Random flow field"},
        ],
    }


@router.get("/patterns")
async def list_timeseries_patterns():
    """List available synthetic timeseries patterns."""
    return {
        "status": "ok",
        "patterns": [
            {"name": "sine", "description": "Sinusoidal wave"},
            {"name": "cosine", "description": "Cosine wave"},
            {"name": "linear", "description": "Linear ramp"},
            {"name": "exponential", "description": "Exponential decay"},
            {"name": "step", "description": "Step function"},
            {"name": "sawtooth", "description": "Sawtooth wave"},
            {"name": "random", "description": "Random noise"},
        ],
    }
