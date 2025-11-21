"""
Tests for the Data Lake module.

Tests DataStore, timeseries utilities, and query functions.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from datetime import datetime

import numpy as np

from aero.data import (
    DataStore,
    get_default_store,
    set_default_store,
    DUCKDB_AVAILABLE,
    pack_timeseries,
    unpack_timeseries,
    compute_basic_ts_stats,
    resample_timeseries,
    detect_anomalies,
    smooth_timeseries,
    list_recent_simulations,
    list_recent_experiments,
    find_simulations_by_type,
    find_experiments_by_type,
    find_simulations_by_tag,
    find_experiments_by_tag,
    find_by_hypothesis,
    get_simulation_with_timeseries,
    count_by_type,
    search_metadata,
)
from aero.sim.results import SimulationResult
from aero.experiments import ExperimentResult


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test data."""
    tmp = tempfile.mkdtemp()
    yield tmp
    shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture
def data_store(temp_dir):
    """Create a DataStore for testing."""
    db_path = str(Path(temp_dir) / "test.db")
    fields_dir = str(Path(temp_dir) / "fields")
    store = DataStore(db_path=db_path, fields_dir=fields_dir)
    return store


class TestDataStore:
    """Tests for DataStore class."""

    def test_initialization(self, data_store):
        """Test DataStore initialization."""
        assert data_store is not None
        assert data_store._conn is not None

    def test_save_and_get_simulation(self, data_store):
        """Test saving and retrieving a simulation result."""
        # Create a simulation result
        result = SimulationResult(
            fields={"temperature": np.array([1.0, 2.0, 3.0, 4.0, 5.0])},
            metadata={
                "simulation_type": "heat_1d",
                "status": "completed",
                "runtime_seconds": 1.5,
            },
            tags=["test", "heat"],
        )

        # Save it
        record_id = data_store.save_simulation_result(result)
        assert record_id is not None
        assert record_id == result.id

        # Retrieve it
        retrieved = data_store.get_simulation(record_id)
        assert retrieved is not None
        assert retrieved["id"] == record_id
        assert retrieved["type"] == "heat_1d"
        assert "test" in retrieved["tags"]

    def test_save_and_get_experiment(self, data_store):
        """Test saving and retrieving an experiment result."""
        # Create an experiment result
        result = ExperimentResult(
            experiment_type="synthetic_flow",
            data={"velocity": [1.0, 2.0, 3.0]},
            metadata={"source": "test"},
            success=True,
            tags=["test", "flow"],
        )

        # Save it
        record_id = data_store.save_experiment_result(result)
        assert record_id is not None
        assert record_id == result.id

        # Retrieve it
        retrieved = data_store.get_experiment(record_id)
        assert retrieved is not None
        assert retrieved["id"] == record_id
        assert retrieved["type"] == "synthetic_flow"
        assert "test" in retrieved["tags"]

    def test_save_field_array(self, data_store):
        """Test saving and loading field arrays."""
        # Create a simulation first
        result = SimulationResult(
            fields={"solution": np.random.rand(100, 100)},
            metadata={"simulation_type": "laplace_2d"},
        )
        sim_id = data_store.save_simulation_result(result)

        # Save field array separately
        large_field = np.random.rand(500, 500)
        field_id = data_store.save_field_array(
            parent_id=sim_id,
            parent_type="simulation",
            name="pressure",
            array=large_field,
        )

        assert field_id is not None

        # Load it back
        loaded = data_store.load_field_array(field_id)
        assert loaded is not None
        assert loaded.shape == (500, 500)
        assert np.allclose(loaded, large_field)

    def test_save_timeseries(self, data_store):
        """Test saving and retrieving timeseries data."""
        # Create a simulation first
        result = SimulationResult(
            fields={},
            metadata={"simulation_type": "heat_1d"},
        )
        sim_id = data_store.save_simulation_result(result)

        # Save timeseries
        t = list(np.linspace(0, 1, 100))
        values = list(np.sin(2 * np.pi * np.array(t)))

        ts_id = data_store.save_timeseries(
            parent_id=sim_id,
            parent_type="simulation",
            name="temperature_probe",
            t=t,
            values=values,
            metadata={"location": "center"},
        )

        assert ts_id is not None

        # List timeseries
        ts_list = data_store.list_timeseries(sim_id, "simulation")
        assert len(ts_list) == 1
        assert ts_list[0]["name"] == "temperature_probe"

    def test_delete_simulation(self, data_store):
        """Test deleting a simulation."""
        # Create and save
        result = SimulationResult(
            fields={},
            metadata={"simulation_type": "test"},
        )
        sim_id = data_store.save_simulation_result(result)

        # Verify it exists
        assert data_store.get_simulation(sim_id) is not None

        # Delete
        success = data_store.delete_simulation(sim_id)
        assert success is True

        # Verify it's gone
        assert data_store.get_simulation(sim_id) is None

    def test_delete_experiment(self, data_store):
        """Test deleting an experiment."""
        # Create and save
        result = ExperimentResult(
            experiment_type="test",
            data={},
        )
        exp_id = data_store.save_experiment_result(result)

        # Verify it exists
        assert data_store.get_experiment(exp_id) is not None

        # Delete
        success = data_store.delete_experiment(exp_id)
        assert success is True

        # Verify it's gone
        assert data_store.get_experiment(exp_id) is None

    def test_get_stats(self, data_store):
        """Test getting database statistics."""
        # Save some data
        for i in range(3):
            data_store.save_simulation_result(SimulationResult(
                fields={},
                metadata={"simulation_type": "heat_1d"},
            ))

        for i in range(2):
            data_store.save_experiment_result(ExperimentResult(
                experiment_type="test",
                data={},
            ))

        # Get stats
        stats = data_store.get_stats()
        assert stats["simulations"] == 3
        assert stats["experiments"] == 2

    def test_hypothesis_id_linking(self, data_store):
        """Test linking simulations and experiments to hypotheses."""
        hypothesis_id = "hyp_12345"

        # Save simulation with hypothesis
        sim_result = SimulationResult(
            fields={},
            metadata={"simulation_type": "heat_1d"},
        )
        sim_id = data_store.save_simulation_result(sim_result, hypothesis_id=hypothesis_id)

        # Save experiment with hypothesis
        exp_result = ExperimentResult(
            experiment_type="test",
            data={},
        )
        exp_id = data_store.save_experiment_result(exp_result, hypothesis_id=hypothesis_id)

        # Find by hypothesis
        results = find_by_hypothesis(data_store, hypothesis_id)
        assert len(results["simulations"]) == 1
        assert len(results["experiments"]) == 1
        assert results["simulations"][0]["id"] == sim_id
        assert results["experiments"][0]["id"] == exp_id


class TestTimeseriesUtilities:
    """Tests for timeseries utility functions."""

    def test_pack_unpack_timeseries(self):
        """Test packing and unpacking timeseries data."""
        t = [0.0, 0.1, 0.2, 0.3, 0.4]
        values = [1.0, 1.1, 0.9, 1.2, 0.8]
        metadata = {"unit": "m/s"}

        # Pack
        packed = pack_timeseries(t, values, metadata)
        assert "t" in packed
        assert "values" in packed
        assert packed["metadata"]["unit"] == "m/s"

        # Unpack
        t_out, values_out, meta_out = unpack_timeseries(packed)
        assert t_out == t
        assert values_out == values
        assert meta_out["unit"] == "m/s"

    def test_compute_basic_ts_stats(self):
        """Test computing basic timeseries statistics."""
        t = list(range(100))
        values = list(np.sin(np.array(t) * 0.1))

        stats = compute_basic_ts_stats(t, values)

        assert "mean" in stats
        assert "std" in stats
        assert "min" in stats
        assert "max" in stats
        assert "duration" in stats
        assert "count" in stats
        assert stats["count"] == 100

    def test_resample_timeseries(self):
        """Test resampling timeseries data."""
        t = list(np.linspace(0, 1, 100))
        values = list(np.sin(2 * np.pi * np.array(t)))

        # Resample to 20 points
        t_new, values_new = resample_timeseries(t, values, num_points=20)

        assert len(t_new) == 20
        assert len(values_new) == 20

    def test_detect_anomalies(self):
        """Test anomaly detection in timeseries."""
        t = list(range(100))
        values = [1.0] * 100
        # Add anomalies
        values[50] = 10.0  # Spike
        values[75] = -5.0  # Another anomaly

        anomalies = detect_anomalies(t, values, threshold=3.0)

        assert len(anomalies) > 0
        # Should detect at least the two obvious anomalies
        anomaly_indices = [a["index"] for a in anomalies]
        assert 50 in anomaly_indices
        assert 75 in anomaly_indices

    def test_smooth_timeseries(self):
        """Test smoothing timeseries data."""
        t = list(range(100))
        # Noisy signal
        values = list(np.sin(np.array(t) * 0.1) + np.random.randn(100) * 0.1)

        smoothed = smooth_timeseries(t, values, window_size=5)

        assert len(smoothed) == len(values)
        # Smoothed should have less variance
        assert np.std(smoothed) < np.std(values) * 1.1  # Allow some tolerance


class TestQueryFunctions:
    """Tests for query utility functions."""

    def test_list_recent_simulations(self, data_store):
        """Test listing recent simulations."""
        # Create several simulations
        for i in range(5):
            data_store.save_simulation_result(SimulationResult(
                fields={},
                metadata={"simulation_type": f"type_{i}"},
            ))

        results = list_recent_simulations(data_store, limit=3)
        assert len(results) == 3

    def test_list_recent_experiments(self, data_store):
        """Test listing recent experiments."""
        # Create several experiments
        for i in range(5):
            data_store.save_experiment_result(ExperimentResult(
                experiment_type=f"type_{i}",
                data={},
            ))

        results = list_recent_experiments(data_store, limit=3)
        assert len(results) == 3

    def test_find_simulations_by_type(self, data_store):
        """Test finding simulations by type."""
        # Create simulations of different types
        for i in range(3):
            data_store.save_simulation_result(SimulationResult(
                fields={},
                metadata={"simulation_type": "heat_1d"},
            ))
        for i in range(2):
            data_store.save_simulation_result(SimulationResult(
                fields={},
                metadata={"simulation_type": "laplace_2d"},
            ))

        heat_results = find_simulations_by_type(data_store, "heat_1d")
        laplace_results = find_simulations_by_type(data_store, "laplace_2d")

        assert len(heat_results) == 3
        assert len(laplace_results) == 2

    def test_find_experiments_by_type(self, data_store):
        """Test finding experiments by type."""
        for i in range(3):
            data_store.save_experiment_result(ExperimentResult(
                experiment_type="synthetic_flow",
                data={},
            ))
        for i in range(2):
            data_store.save_experiment_result(ExperimentResult(
                experiment_type="camera_snapshot",
                data={},
            ))

        flow_results = find_experiments_by_type(data_store, "synthetic_flow")
        camera_results = find_experiments_by_type(data_store, "camera_snapshot")

        assert len(flow_results) == 3
        assert len(camera_results) == 2

    def test_find_simulations_by_tag(self, data_store):
        """Test finding simulations by tag."""
        # Create simulations with different tags
        data_store.save_simulation_result(SimulationResult(
            fields={},
            metadata={"simulation_type": "test"},
            tags=["important", "heat"],
        ))
        data_store.save_simulation_result(SimulationResult(
            fields={},
            metadata={"simulation_type": "test"},
            tags=["important", "flow"],
        ))
        data_store.save_simulation_result(SimulationResult(
            fields={},
            metadata={"simulation_type": "test"},
            tags=["flow"],
        ))

        important = find_simulations_by_tag(data_store, "important")
        flow = find_simulations_by_tag(data_store, "flow")

        assert len(important) == 2
        assert len(flow) == 2

    def test_find_experiments_by_tag(self, data_store):
        """Test finding experiments by tag."""
        data_store.save_experiment_result(ExperimentResult(
            experiment_type="test",
            data={},
            tags=["calibration"],
        ))
        data_store.save_experiment_result(ExperimentResult(
            experiment_type="test",
            data={},
            tags=["production"],
        ))

        calibration = find_experiments_by_tag(data_store, "calibration")
        production = find_experiments_by_tag(data_store, "production")

        assert len(calibration) == 1
        assert len(production) == 1

    def test_count_by_type(self, data_store):
        """Test counting records by type."""
        # Create simulations
        for i in range(3):
            data_store.save_simulation_result(SimulationResult(
                fields={},
                metadata={"simulation_type": "heat_1d"},
            ))
        for i in range(2):
            data_store.save_simulation_result(SimulationResult(
                fields={},
                metadata={"simulation_type": "laplace_2d"},
            ))

        counts = count_by_type(data_store, "simulations")

        assert counts.get("heat_1d", 0) == 3
        assert counts.get("laplace_2d", 0) == 2

    def test_search_metadata(self, data_store):
        """Test searching by metadata."""
        data_store.save_simulation_result(SimulationResult(
            fields={},
            metadata={
                "simulation_type": "test",
                "solver": "jacobi",
            },
        ))
        data_store.save_simulation_result(SimulationResult(
            fields={},
            metadata={
                "simulation_type": "test",
                "solver": "gauss_seidel",
            },
        ))

        jacobi = search_metadata(data_store, "solver", "jacobi", table="simulations")
        gauss = search_metadata(data_store, "solver", "gauss", table="simulations")

        assert len(jacobi) == 1
        assert len(gauss) == 1

    def test_get_simulation_with_timeseries(self, data_store):
        """Test getting simulation with associated timeseries."""
        # Create simulation
        sim_result = SimulationResult(
            fields={},
            metadata={"simulation_type": "heat_1d"},
        )
        sim_id = data_store.save_simulation_result(sim_result)

        # Add timeseries
        data_store.save_timeseries(
            parent_id=sim_id,
            parent_type="simulation",
            name="probe_1",
            t=[0, 1, 2],
            values=[1.0, 1.1, 1.2],
        )
        data_store.save_timeseries(
            parent_id=sim_id,
            parent_type="simulation",
            name="probe_2",
            t=[0, 1, 2],
            values=[2.0, 2.1, 2.2],
        )

        # Get with timeseries
        result = get_simulation_with_timeseries(data_store, sim_id)

        assert result is not None
        assert "timeseries" in result
        assert len(result["timeseries"]) == 2


class TestSimulationResultDataLake:
    """Tests for SimulationResult integration with data lake."""

    def test_simulation_result_has_id(self):
        """Test that SimulationResult has a UUID."""
        result = SimulationResult(
            fields={},
            metadata={},
        )
        assert result.id is not None
        assert len(result.id) == 36  # UUID format

    def test_simulation_result_to_record(self):
        """Test SimulationResult.to_record method."""
        result = SimulationResult(
            fields={"temperature": np.array([1, 2, 3])},
            metadata={
                "simulation_type": "heat_1d",
                "config": {"alpha": 0.01},
                "status": "completed",
            },
            tags=["test"],
        )

        record = result.to_record()

        assert record["id"] == result.id
        assert record["type"] == "heat_1d"
        assert record["status"] == "completed"
        assert "test" in record["tags"]
        assert "config" in record
        assert "fields" in record

    def test_simulation_result_preserves_id_on_json_roundtrip(self):
        """Test that ID is preserved through JSON serialization."""
        original = SimulationResult(
            fields={"x": np.array([1, 2, 3])},
            metadata={"type": "test"},
            tags=["tag1"],
        )
        original_id = original.id

        # Roundtrip through JSON
        json_str = original.to_json()
        restored = SimulationResult.from_json(json_str)

        assert restored.id == original_id


class TestExperimentResultDataLake:
    """Tests for ExperimentResult integration with data lake."""

    def test_experiment_result_has_id(self):
        """Test that ExperimentResult has a UUID."""
        result = ExperimentResult(
            experiment_type="test",
            data={},
        )
        assert result.id is not None
        assert len(result.id) == 36  # UUID format

    def test_experiment_result_to_record(self):
        """Test ExperimentResult.to_record method."""
        result = ExperimentResult(
            experiment_type="synthetic_flow",
            data={"velocity": [1, 2, 3]},
            metadata={"source": "test"},
            tags=["experiment"],
        )

        record = result.to_record()

        assert record["id"] == result.id
        assert record["type"] == "synthetic_flow"
        assert "experiment" in record["tags"]
        assert record["status"] == "completed"

    def test_experiment_result_preserves_id_on_dict_roundtrip(self):
        """Test that ID is preserved through dict serialization."""
        original = ExperimentResult(
            experiment_type="test",
            data={"x": [1, 2, 3]},
            tags=["tag1"],
        )
        original_id = original.id

        # Roundtrip through dict
        d = original.to_dict()
        restored = ExperimentResult.from_dict(d)

        assert restored.id == original_id


class TestDefaultStore:
    """Tests for default store management."""

    def test_set_and_get_default_store(self, temp_dir):
        """Test setting and getting the default store."""
        db_path = str(Path(temp_dir) / "default.db")
        fields_dir = str(Path(temp_dir) / "fields")

        store = DataStore(db_path=db_path, fields_dir=fields_dir)
        set_default_store(store)

        retrieved = get_default_store()
        assert retrieved is store
