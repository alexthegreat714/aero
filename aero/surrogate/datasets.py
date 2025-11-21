"""
Dataset builders for Surrogate Modeling.

Builds PyTorch Datasets from DataStore records (simulations, experiments, timeseries).
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Check for PyTorch
try:
    import torch
    from torch.utils.data import Dataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    Dataset = object


if TORCH_AVAILABLE:

    class SimulationFieldDataset(Dataset):
        """
        Dataset built from simulation field data.

        Samples points from simulation outputs (e.g., temperature fields)
        for training surrogate models.

        Example:
            # For heat_1d: input (x, t) -> output u(x, t)
            dataset = SimulationFieldDataset(
                records=sim_records,
                fields_loader=lambda fid: store.load_field_array(fid),
                input_fields=["x", "t"],
                output_field="solution",
                sample_mode="points",
                num_samples=1000,
            )
        """

        def __init__(
            self,
            records: List[Dict[str, Any]],
            fields_loader: Callable[[str], Optional[np.ndarray]],
            input_fields: Optional[List[str]] = None,
            output_field: str = "solution",
            sample_mode: str = "points",
            num_samples: int = 1000,
            normalize: bool = True,
        ):
            """
            Initialize the dataset.

            Args:
                records: List of simulation records from DataStore
                fields_loader: Callback to load field arrays by ID
                input_fields: Input field names (default: infer from sim type)
                output_field: Output field name
                sample_mode: "points" samples random points, "all" uses all data
                num_samples: Number of samples to draw per simulation
                normalize: Whether to normalize inputs/outputs
            """
            self.records = records
            self.fields_loader = fields_loader
            self.output_field = output_field
            self.sample_mode = sample_mode
            self.num_samples = num_samples
            self.normalize = normalize

            # Build dataset
            self.inputs: List[np.ndarray] = []
            self.outputs: List[np.ndarray] = []

            self._build_dataset(input_fields)

            # Normalize if requested
            if normalize and len(self.inputs) > 0:
                self._compute_normalization()

            logger.info(
                f"SimulationFieldDataset: {len(self.inputs)} samples from "
                f"{len(records)} records"
            )

        def _build_dataset(self, input_fields: Optional[List[str]]) -> None:
            """Build dataset from records."""
            for record in self.records:
                try:
                    self._process_record(record, input_fields)
                except Exception as e:
                    logger.warning(f"Error processing record {record.get('id', '?')}: {e}")

        def _process_record(
            self,
            record: Dict[str, Any],
            input_fields: Optional[List[str]],
        ) -> None:
            """Process a single simulation record."""
            sim_type = record.get("type", "unknown")

            # Get solution field
            fields_meta = record.get("fields_meta", [])
            solution = None

            # Try to load solution from fields
            for field_meta in fields_meta:
                if field_meta.get("name") == self.output_field:
                    solution = self.fields_loader(field_meta["id"])
                    break

            # Fallback: check if solution is in record directly
            if solution is None:
                raw_fields = record.get("fields", {})
                if self.output_field in raw_fields:
                    sol_data = raw_fields[self.output_field]
                    if isinstance(sol_data, list):
                        solution = np.array(sol_data)
                    elif isinstance(sol_data, np.ndarray):
                        solution = sol_data

            if solution is None:
                return

            # Generate input coordinates based on simulation type
            if sim_type == "heat_1d":
                self._sample_heat_1d(solution, record)
            elif sim_type == "laplace_2d":
                self._sample_laplace_2d(solution, record)
            else:
                # Generic: assume 1D or 2D array
                self._sample_generic(solution, record)

        def _sample_heat_1d(self, solution: np.ndarray, record: Dict) -> None:
            """Sample from 1D heat equation solution."""
            # solution shape: (nx,) or (nx, nt) for time-dependent
            if solution.ndim == 1:
                nx = len(solution)
                x = np.linspace(0, 1, nx)

                if self.sample_mode == "points":
                    indices = np.random.choice(nx, min(self.num_samples, nx), replace=False)
                else:
                    indices = np.arange(nx)

                for i in indices:
                    self.inputs.append(np.array([x[i]]))
                    self.outputs.append(np.array([solution[i]]))

            else:
                # Time-dependent: (nx, nt)
                nx, nt = solution.shape
                x = np.linspace(0, 1, nx)
                t = np.linspace(0, 1, nt)

                if self.sample_mode == "points":
                    n_total = min(self.num_samples, nx * nt)
                    for _ in range(n_total):
                        i = np.random.randint(nx)
                        j = np.random.randint(nt)
                        self.inputs.append(np.array([x[i], t[j]]))
                        self.outputs.append(np.array([solution[i, j]]))
                else:
                    for i in range(nx):
                        for j in range(nt):
                            self.inputs.append(np.array([x[i], t[j]]))
                            self.outputs.append(np.array([solution[i, j]]))

        def _sample_laplace_2d(self, solution: np.ndarray, record: Dict) -> None:
            """Sample from 2D Laplace solution."""
            if solution.ndim != 2:
                return

            nx, ny = solution.shape
            x = np.linspace(0, 1, nx)
            y = np.linspace(0, 1, ny)

            if self.sample_mode == "points":
                n_total = min(self.num_samples, nx * ny)
                for _ in range(n_total):
                    i = np.random.randint(nx)
                    j = np.random.randint(ny)
                    self.inputs.append(np.array([x[i], y[j]]))
                    self.outputs.append(np.array([solution[i, j]]))
            else:
                for i in range(nx):
                    for j in range(ny):
                        self.inputs.append(np.array([x[i], y[j]]))
                        self.outputs.append(np.array([solution[i, j]]))

        def _sample_generic(self, solution: np.ndarray, record: Dict) -> None:
            """Sample from generic solution array."""
            flat = solution.flatten()
            n = len(flat)
            coords = np.linspace(0, 1, n)

            if self.sample_mode == "points":
                indices = np.random.choice(n, min(self.num_samples, n), replace=False)
            else:
                indices = np.arange(n)

            for i in indices:
                self.inputs.append(np.array([coords[i]]))
                self.outputs.append(np.array([flat[i]]))

        def _compute_normalization(self) -> None:
            """Compute normalization statistics."""
            inputs_arr = np.array(self.inputs)
            outputs_arr = np.array(self.outputs)

            self.input_mean = inputs_arr.mean(axis=0)
            self.input_std = inputs_arr.std(axis=0) + 1e-8

            self.output_mean = outputs_arr.mean(axis=0)
            self.output_std = outputs_arr.std(axis=0) + 1e-8

        def __len__(self) -> int:
            return len(self.inputs)

        def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
            x = self.inputs[idx].astype(np.float32)
            y = self.outputs[idx].astype(np.float32)

            if self.normalize:
                x = (x - self.input_mean) / self.input_std
                y = (y - self.output_mean) / self.output_std

            return torch.from_numpy(x), torch.from_numpy(y)

        def get_normalization_params(self) -> Dict[str, np.ndarray]:
            """Get normalization parameters for inference."""
            if not self.normalize:
                return {}
            return {
                "input_mean": self.input_mean,
                "input_std": self.input_std,
                "output_mean": self.output_mean,
                "output_std": self.output_std,
            }


    class TimeSeriesDataset(Dataset):
        """
        Dataset for time series prediction.

        Uses sliding windows to create input-output pairs for
        sequence prediction tasks.

        Example:
            dataset = TimeSeriesDataset(
                ts_records=timeseries_records,
                window_size=10,
                prediction_horizon=1,
            )
        """

        def __init__(
            self,
            ts_records: List[Dict[str, Any]],
            window_size: int = 10,
            prediction_horizon: int = 1,
            normalize: bool = True,
        ):
            """
            Initialize the dataset.

            Args:
                ts_records: List of timeseries records
                window_size: Number of past values to use as input
                prediction_horizon: Number of future values to predict
                normalize: Whether to normalize values
            """
            self.window_size = window_size
            self.prediction_horizon = prediction_horizon
            self.normalize = normalize

            self.inputs: List[np.ndarray] = []
            self.outputs: List[np.ndarray] = []

            self._build_dataset(ts_records)

            if normalize and len(self.inputs) > 0:
                self._compute_normalization()

            logger.info(
                f"TimeSeriesDataset: {len(self.inputs)} samples, "
                f"window={window_size}, horizon={prediction_horizon}"
            )

        def _build_dataset(self, ts_records: List[Dict]) -> None:
            """Build dataset from timeseries records."""
            for record in ts_records:
                try:
                    values = record.get("values", [])
                    if isinstance(values, str):
                        import json
                        values = json.loads(values)

                    if len(values) < self.window_size + self.prediction_horizon:
                        continue

                    values = np.array(values, dtype=np.float32)

                    # Create sliding windows
                    for i in range(len(values) - self.window_size - self.prediction_horizon + 1):
                        x = values[i:i + self.window_size]
                        y = values[i + self.window_size:i + self.window_size + self.prediction_horizon]
                        self.inputs.append(x)
                        self.outputs.append(y)

                except Exception as e:
                    logger.warning(f"Error processing timeseries: {e}")

        def _compute_normalization(self) -> None:
            """Compute normalization statistics."""
            all_values = np.concatenate([np.concatenate([x, y]) for x, y in zip(self.inputs, self.outputs)])
            self.mean = all_values.mean()
            self.std = all_values.std() + 1e-8

        def __len__(self) -> int:
            return len(self.inputs)

        def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
            x = self.inputs[idx]
            y = self.outputs[idx]

            if self.normalize:
                x = (x - self.mean) / self.std
                y = (y - self.mean) / self.std

            return torch.from_numpy(x), torch.from_numpy(y)


    class TabularDataset(Dataset):
        """
        Generic tabular dataset for regression tasks.

        Simple dataset from numpy arrays of inputs and outputs.

        Example:
            X = np.random.randn(1000, 3)
            y = np.random.randn(1000, 1)
            dataset = TabularDataset(X, y)
        """

        def __init__(
            self,
            inputs: np.ndarray,
            outputs: np.ndarray,
            normalize: bool = True,
        ):
            """
            Initialize the dataset.

            Args:
                inputs: Input features (N, input_dim)
                outputs: Output targets (N, output_dim)
                normalize: Whether to normalize
            """
            self.inputs = inputs.astype(np.float32)
            self.outputs = outputs.astype(np.float32)
            self.normalize = normalize

            if normalize:
                self.input_mean = self.inputs.mean(axis=0)
                self.input_std = self.inputs.std(axis=0) + 1e-8
                self.output_mean = self.outputs.mean(axis=0)
                self.output_std = self.outputs.std(axis=0) + 1e-8

        def __len__(self) -> int:
            return len(self.inputs)

        def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
            x = self.inputs[idx]
            y = self.outputs[idx]

            if self.normalize:
                x = (x - self.input_mean) / self.input_std
                y = (y - self.output_mean) / self.output_std

            return torch.from_numpy(x), torch.from_numpy(y)

        def get_normalization_params(self) -> Dict[str, np.ndarray]:
            """Get normalization parameters."""
            if not self.normalize:
                return {}
            return {
                "input_mean": self.input_mean,
                "input_std": self.input_std,
                "output_mean": self.output_mean,
                "output_std": self.output_std,
            }


def build_dataset_from_simulations(
    store: "DataStore",
    sim_type: str,
    output_field: str = "solution",
    num_samples: int = 1000,
    max_records: int = 50,
    normalize: bool = True,
) -> "SimulationFieldDataset":
    """
    Build a dataset from simulations of a specific type.

    Args:
        store: DataStore instance
        sim_type: Simulation type (e.g., "heat_1d", "laplace_2d")
        output_field: Name of the output field
        num_samples: Samples per simulation
        max_records: Maximum number of simulations to use
        normalize: Whether to normalize data

    Returns:
        SimulationFieldDataset
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for datasets")

    from aero.data.query import get_simulation_records_for_type

    records = get_simulation_records_for_type(store, sim_type, limit=max_records)

    # Add field metadata to records
    for record in records:
        fields_meta = store.list_fields(record["id"], "simulation")
        record["fields_meta"] = fields_meta

    def fields_loader(field_id: str) -> Optional[np.ndarray]:
        return store.load_field_array(field_id)

    return SimulationFieldDataset(
        records=records,
        fields_loader=fields_loader,
        output_field=output_field,
        sample_mode="points",
        num_samples=num_samples,
        normalize=normalize,
    )


def build_dataset_from_timeseries(
    store: "DataStore",
    tag: Optional[str] = None,
    window_size: int = 10,
    prediction_horizon: int = 1,
    max_records: int = 50,
    normalize: bool = True,
) -> "TimeSeriesDataset":
    """
    Build a dataset from timeseries records.

    Args:
        store: DataStore instance
        tag: Optional tag to filter timeseries
        window_size: Input window size
        prediction_horizon: Prediction horizon
        max_records: Maximum records to use
        normalize: Whether to normalize

    Returns:
        TimeSeriesDataset
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for datasets")

    from aero.data.query import get_timeseries_for_tag

    if tag:
        ts_records = get_timeseries_for_tag(store, tag, limit=max_records)
    else:
        # Get all timeseries
        ts_records = []
        # Query from simulations
        conn = store._conn
        cursor = conn.execute(
            "SELECT * FROM timeseries ORDER BY created DESC LIMIT ?",
            (max_records,)
        )
        for row in cursor.fetchall():
            ts_records.append(dict(row))

    return TimeSeriesDataset(
        ts_records=ts_records,
        window_size=window_size,
        prediction_horizon=prediction_horizon,
        normalize=normalize,
    )


else:
    # Dummy implementations
    class SimulationFieldDataset:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch is required")

    class TimeSeriesDataset:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch is required")

    class TabularDataset:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch is required")

    def build_dataset_from_simulations(*args, **kwargs):
        raise RuntimeError("PyTorch is required")

    def build_dataset_from_timeseries(*args, **kwargs):
        raise RuntimeError("PyTorch is required")
