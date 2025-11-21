"""
Tests for the experiments module.

These tests run without hardware dependencies (webcam, sensors)
by using synthetic data generators.
"""

import json
import pytest
import tempfile
import numpy as np
from datetime import datetime
from pathlib import Path


class TestExperimentResult:
    """Tests for ExperimentResult dataclass."""

    def test_create_experiment_result(self):
        """Test creating an ExperimentResult."""
        from aero.experiments import ExperimentResult

        result = ExperimentResult(
            experiment_type="test",
            data={"values": [1, 2, 3]},
            metadata={"test": True},
        )

        assert result.experiment_type == "test"
        assert result.success is True
        assert result.error is None
        assert result.data["values"] == [1, 2, 3]

    def test_experiment_result_to_dict(self):
        """Test converting ExperimentResult to dictionary."""
        from aero.experiments import ExperimentResult

        result = ExperimentResult(
            experiment_type="test",
            data={"value": 42},
            metadata={"source": "test"},
        )

        d = result.to_dict()
        assert d["experiment_type"] == "test"
        assert d["data"]["value"] == 42
        assert d["success"] is True
        assert "timestamp" in d

    def test_experiment_result_to_json(self):
        """Test converting ExperimentResult to JSON."""
        from aero.experiments import ExperimentResult

        result = ExperimentResult(
            experiment_type="test",
            data={"array": np.array([1, 2, 3])},
        )

        json_str = result.to_json()
        parsed = json.loads(json_str)
        assert parsed["experiment_type"] == "test"
        assert parsed["data"]["array"] == [1, 2, 3]

    def test_experiment_result_from_dict(self):
        """Test creating ExperimentResult from dictionary."""
        from aero.experiments import ExperimentResult

        d = {
            "experiment_type": "test",
            "data": {"value": 100},
            "metadata": {},
            "success": True,
            "timestamp": datetime.now().isoformat(),
        }

        result = ExperimentResult.from_dict(d)
        assert result.experiment_type == "test"
        assert result.data["value"] == 100

    def test_experiment_result_failure(self):
        """Test creating a failure result."""
        from aero.experiments import ExperimentResult

        result = ExperimentResult.failure("test", "Something went wrong")
        assert result.success is False
        assert result.error == "Something went wrong"


class TestSyntheticExperiments:
    """Tests for synthetic experiment generation."""

    def test_generate_synthetic_flow(self):
        """Test generating synthetic optical flow."""
        from aero.experiments.synthetic import generate_synthetic_flow

        result = generate_synthetic_flow(
            width=32,
            height=32,
            flow_type="uniform",
            magnitude=5.0,
            seed=42,
        )

        assert "flow" in result
        assert result["shape"] == (32, 32)
        assert result["flow_type"] == "uniform"
        assert result["mean_magnitude"] > 0

    def test_synthetic_flow_types(self):
        """Test all synthetic flow types."""
        from aero.experiments.synthetic import generate_synthetic_flow

        flow_types = ["uniform", "radial", "vortex", "shear", "random"]

        for flow_type in flow_types:
            result = generate_synthetic_flow(
                width=16,
                height=16,
                flow_type=flow_type,
                magnitude=3.0,
            )
            assert result["flow_type"] == flow_type
            assert result["mean_magnitude"] > 0

    def test_generate_synthetic_timeseries(self):
        """Test generating synthetic time series."""
        from aero.experiments.synthetic import generate_synthetic_timeseries

        result = generate_synthetic_timeseries(
            length=100,
            pattern="sine",
            noise_level=0.1,
            frequency=2.0,
            amplitude=5.0,
            offset=10.0,
            seed=42,
        )

        assert len(result["values"]) == 100
        assert result["pattern"] == "sine"
        assert result["length"] == 100
        assert "mean" in result["statistics"]

    def test_timeseries_patterns(self):
        """Test all synthetic timeseries patterns."""
        from aero.experiments.synthetic import generate_synthetic_timeseries

        patterns = ["sine", "cosine", "linear", "exponential", "step", "random"]

        for pattern in patterns:
            result = generate_synthetic_timeseries(
                length=50,
                pattern=pattern,
                seed=42,
            )
            assert result["pattern"] == pattern
            assert len(result["values"]) == 50

    def test_generate_test_image(self):
        """Test generating test images."""
        from aero.experiments.synthetic import generate_test_image

        image = generate_test_image(
            width=100,
            height=80,
            pattern="gradient",
            channels=3,
        )

        assert image.shape == (80, 100, 3)
        assert image.dtype == np.uint8

    def test_image_patterns(self):
        """Test all test image patterns."""
        from aero.experiments.synthetic import generate_test_image

        patterns = ["gradient", "gradient_v", "checkerboard", "circle", "noise", "solid", "bars"]

        for pattern in patterns:
            image = generate_test_image(
                width=64,
                height=64,
                pattern=pattern,
                channels=1,
            )
            assert image.shape == (64, 64)

    def test_synthetic_experiment_class(self):
        """Test SyntheticExperiment class."""
        from aero.experiments.synthetic import SyntheticExperiment

        exp = SyntheticExperiment(name="test", seed=42)

        # Test timeseries generation
        values, timestamps = exp.generate_timeseries(
            length=50,
            pattern="sine",
        )
        assert len(values) == 50
        assert len(timestamps) == 50

        # Test image generation
        image = exp.generate_image(width=32, height=32, pattern="gradient")
        assert image.shape == (32, 32)

        # Test image sequence generation
        frames = exp.generate_image_sequence(
            num_frames=5,
            width=32,
            height=32,
        )
        assert len(frames) == 5

        # Test flow field generation
        flow = exp.generate_flow_field(width=16, height=16, flow_type="radial")
        assert flow.shape == (16, 16, 2)


class TestSensorTimeseries:
    """Tests for sensor file reading."""

    def test_read_csv_timeseries(self):
        """Test reading CSV time series."""
        from aero.experiments.sensors import read_csv_timeseries

        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("timestamp,value,sensor_id\n")
            f.write("2024-01-01T00:00:00,10.5,temp1\n")
            f.write("2024-01-01T00:00:01,11.2,temp1\n")
            f.write("2024-01-01T00:00:02,10.8,temp1\n")
            csv_path = f.name

        try:
            values, timestamps = read_csv_timeseries(csv_path)
            assert len(values) == 3
            assert len(timestamps) == 3
            assert values[0] == pytest.approx(10.5)
        finally:
            Path(csv_path).unlink()

    def test_read_json_timeseries_array(self):
        """Test reading JSON time series (array format)."""
        from aero.experiments.sensors import read_json_timeseries

        # Create temporary JSON file with array of values
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump([1.0, 2.0, 3.0, 4.0, 5.0], f)
            json_path = f.name

        try:
            values, timestamps = read_json_timeseries(json_path)
            assert len(values) == 5
            assert values == [1.0, 2.0, 3.0, 4.0, 5.0]
        finally:
            Path(json_path).unlink()

    def test_read_json_timeseries_objects(self):
        """Test reading JSON time series (array of objects)."""
        from aero.experiments.sensors import read_json_timeseries

        # Create temporary JSON file with array of objects
        data = [
            {"value": 10.0, "time": "2024-01-01T00:00:00"},
            {"value": 20.0, "time": "2024-01-01T00:01:00"},
            {"value": 30.0, "time": "2024-01-01T00:02:00"},
        ]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(data, f)
            json_path = f.name

        try:
            values, timestamps = read_json_timeseries(json_path)
            assert len(values) == 3
            assert values == [10.0, 20.0, 30.0]
        finally:
            Path(json_path).unlink()

    def test_file_sensor(self):
        """Test FileSensor class."""
        from aero.experiments.sensors import FileSensor

        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("value\n")
            f.write("1.0\n")
            f.write("2.0\n")
            f.write("3.0\n")
            csv_path = f.name

        try:
            sensor = FileSensor(
                sensor_id="test",
                file_path=csv_path,
                unit="units",
                loop=True,
            )

            assert sensor.connect() is True

            # Read values
            reading1 = sensor.read()
            assert reading1.value == pytest.approx(1.0)

            reading2 = sensor.read()
            assert reading2.value == pytest.approx(2.0)

            sensor.disconnect()
        finally:
            Path(csv_path).unlink()


class TestAnalyzer:
    """Tests for the analyzer module."""

    def test_analyze_frame(self):
        """Test frame analysis."""
        from aero.experiments.analyzer import analyze_frame

        # Create test image
        image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)

        result = analyze_frame(image)

        assert "shape" in result
        assert "mean" in result
        assert "std" in result
        assert result["shape"] == (100, 100)

    def test_analyze_frame_color(self):
        """Test color frame analysis."""
        from aero.experiments.analyzer import analyze_frame

        # Create test color image
        image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

        result = analyze_frame(image)

        assert result["color"] is True
        assert result["channels"] == 3
        assert "mean_per_channel" in result

    def test_detect_motion(self):
        """Test motion detection."""
        from aero.experiments.analyzer import detect_motion

        # Create two frames with difference
        frame1 = np.zeros((100, 100), dtype=np.uint8)
        frame2 = np.zeros((100, 100), dtype=np.uint8)
        frame2[40:60, 40:60] = 255  # Add white square

        result = detect_motion(frame1, frame2)

        assert "motion_detected" in result
        assert "diff_mean" in result
        assert result["diff_mean"] > 0

    def test_compute_histogram(self):
        """Test histogram computation."""
        from aero.experiments.analyzer import compute_histogram

        image = np.random.randint(0, 256, (64, 64), dtype=np.uint8)

        result = compute_histogram(image)

        assert "histogram" in result
        assert "peak_value" in result
        assert "mean_intensity" in result

    def test_detect_edges(self):
        """Test edge detection."""
        from aero.experiments.analyzer import detect_edges

        # Create image with clear edge
        image = np.zeros((100, 100), dtype=np.uint8)
        image[:, 50:] = 255

        result = detect_edges(image)

        assert "edge_count" in result
        assert "edge_density" in result
        assert result["edge_count"] > 0

    def test_frame_difference_stats(self):
        """Test frame difference statistics."""
        from aero.experiments.analyzer import compute_frame_difference_stats
        from aero.experiments.synthetic import SyntheticExperiment

        exp = SyntheticExperiment(seed=42)
        frames = exp.generate_image_sequence(num_frames=5, width=64, height=64)

        stats = compute_frame_difference_stats(frames)

        assert "mean_diff" in stats
        assert "max_diff" in stats
        assert "motion_frames" in stats


class TestDataAnalyzer:
    """Tests for the DataAnalyzer class."""

    def test_compute_statistics(self):
        """Test computing statistics."""
        from aero.experiments.analyzer import DataAnalyzer

        analyzer = DataAnalyzer()
        data = np.array([1, 2, 3, 4, 5])

        stats = analyzer.compute_statistics(data)

        assert stats["count"] == 5
        assert stats["mean"] == pytest.approx(3.0)
        assert stats["min"] == 1.0
        assert stats["max"] == 5.0

    def test_moving_average(self):
        """Test moving average."""
        from aero.experiments.analyzer import DataAnalyzer

        analyzer = DataAnalyzer()
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

        smoothed = analyzer.moving_average(data, window_size=3)

        assert len(smoothed) == 8  # Length is reduced by window - 1
        assert smoothed[0] == pytest.approx(2.0)  # (1+2+3)/3

    def test_detect_peaks(self):
        """Test peak detection."""
        from aero.experiments.analyzer import DataAnalyzer

        analyzer = DataAnalyzer()
        # Create data with clear peaks
        data = np.array([0, 1, 5, 1, 0, 1, 10, 1, 0])

        peaks = analyzer.detect_peaks(data, threshold=4)

        assert len(peaks) >= 1
        assert 2 in peaks or 6 in peaks  # Should find peaks at index 2 and 6

    def test_compute_fft(self):
        """Test FFT computation."""
        from aero.experiments.analyzer import DataAnalyzer

        analyzer = DataAnalyzer()
        # Create sinusoidal signal
        t = np.linspace(0, 1, 100)
        data = np.sin(2 * np.pi * 5 * t)  # 5 Hz signal

        frequencies, magnitudes = analyzer.compute_fft(data, sample_rate=100)

        assert len(frequencies) > 0
        assert len(magnitudes) == len(frequencies)

    def test_compute_derivative(self):
        """Test numerical derivative."""
        from aero.experiments.analyzer import DataAnalyzer

        analyzer = DataAnalyzer()
        # Linear data: y = 2x, derivative should be ~2
        data = np.array([0, 2, 4, 6, 8, 10])

        derivative = analyzer.compute_derivative(data, dx=1.0)

        assert len(derivative) == len(data)
        assert np.mean(derivative) == pytest.approx(2.0, rel=0.1)


class TestPlannerExperiments:
    """Tests for experiment planning and execution."""

    def test_execute_synthetic_flow(self):
        """Test executing synthetic flow experiment."""
        from aero.pipeline.planner import (
            ExperimentConfig,
            ExperimentType,
            execute_experiment,
        )

        config = ExperimentConfig(
            exp_type=ExperimentType.SYNTHETIC_FLOW,
            hypothesis_id="test",
            source="",
            parameters={
                "width": 32,
                "height": 32,
                "flow_type": "vortex",
                "magnitude": 3.0,
            },
        )

        result = execute_experiment(config)

        assert result.success is True
        assert result.experiment_type == "synthetic_flow"
        assert "mean_magnitude" in result.data

    def test_execute_synthetic_timeseries(self):
        """Test executing synthetic timeseries experiment."""
        from aero.pipeline.planner import (
            ExperimentConfig,
            ExperimentType,
            execute_experiment,
        )

        config = ExperimentConfig(
            exp_type=ExperimentType.SYNTHETIC_TIMESERIES,
            hypothesis_id="test",
            source="",
            parameters={
                "length": 50,
                "pattern": "sine",
                "amplitude": 2.0,
            },
        )

        result = execute_experiment(config)

        assert result.success is True
        assert result.experiment_type == "synthetic_timeseries"
        assert result.data["length"] == 50

    def test_create_experiment_config(self):
        """Test creating experiment config."""
        from aero.pipeline.planner import (
            ExperimentType,
            create_experiment_config,
        )

        config = create_experiment_config(
            exp_type=ExperimentType.SYNTHETIC_FLOW,
            hypothesis_id="test123",
            width=64,
            height=64,
        )

        assert config.exp_type == ExperimentType.SYNTHETIC_FLOW
        assert config.hypothesis_id == "test123"
        assert config.parameters["width"] == 64


class TestScientificReasoningLoopExperiments:
    """Tests for experiment integration in ScientificReasoningLoop."""

    def test_run_synthetic_experiment(self):
        """Test running synthetic experiment through loop."""
        from aero.pipeline.aero_loop import ScientificReasoningLoop, LoopConfig

        config = LoopConfig(max_iterations=1)
        loop = ScientificReasoningLoop(config=config)

        result = loop.run_synthetic_experiment(
            exp_type="synthetic_flow",
            width=16,
            height=16,
            flow_type="uniform",
        )

        assert result["success"] is True
        assert result["experiment_type"] == "synthetic_flow"

    def test_run_experiment_with_dict(self):
        """Test running experiment with dict config."""
        from aero.pipeline.aero_loop import ScientificReasoningLoop, LoopConfig

        config = LoopConfig(max_iterations=1)
        loop = ScientificReasoningLoop(config=config)

        result = loop.run_experiment({
            "exp_type": "synthetic_timeseries",
            "parameters": {
                "length": 30,
                "pattern": "cosine",
            },
        })

        assert result["success"] is True
        assert result["experiment_type"] == "synthetic_timeseries"


class TestExperimentTypes:
    """Tests for experiment type enum."""

    def test_experiment_type_values(self):
        """Test experiment type enum values."""
        from aero.experiments import ExperimentType

        assert ExperimentType.CAMERA_SNAPSHOT.value == "camera_snapshot"
        assert ExperimentType.SYNTHETIC_FLOW.value == "synthetic_flow"
        assert ExperimentType.SENSOR_TIMESERIES.value == "sensor_timeseries"

    def test_planner_experiment_types(self):
        """Test planner experiment types."""
        from aero.pipeline.planner import ExperimentType as PlannerExpType

        assert PlannerExpType.CAMERA_SNAPSHOT.value == "camera_snapshot"
        assert PlannerExpType.SYNTHETIC_FLOW.value == "synthetic_flow"


# Run tests with: pytest tests/test_experiments.py -v
