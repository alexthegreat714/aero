"""
Sensor data acquisition for Aero Agent.

Provides interfaces for reading sensor data in experiments.
"""

import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque
import threading

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SensorData:
    """Container for sensor readings."""

    sensor_id: str
    value: Any
    timestamp: datetime = field(default_factory=datetime.now)
    unit: str = ""
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "sensor_id": self.sensor_id,
            "value": self.value,
            "timestamp": self.timestamp.isoformat(),
            "unit": self.unit,
            "metadata": self.metadata,
        }


class BaseSensor(ABC):
    """Base class for sensor interfaces."""

    def __init__(self, sensor_id: str):
        """
        Initialize the sensor.

        Args:
            sensor_id: Unique identifier for this sensor
        """
        self.sensor_id = sensor_id
        self._is_connected = False

    @property
    @abstractmethod
    def unit(self) -> str:
        """Measurement unit."""
        raise NotImplementedError

    @abstractmethod
    def connect(self) -> bool:
        """Connect to the sensor."""
        raise NotImplementedError

    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from the sensor."""
        raise NotImplementedError

    @abstractmethod
    def read(self) -> SensorData:
        """Read current sensor value."""
        raise NotImplementedError


class MockSensor(BaseSensor):
    """
    Mock sensor for testing.

    Generates random values within a specified range.
    """

    def __init__(
        self,
        sensor_id: str,
        min_value: float = 0.0,
        max_value: float = 100.0,
        unit: str = "units",
    ):
        """
        Initialize the mock sensor.

        Args:
            sensor_id: Sensor identifier
            min_value: Minimum generated value
            max_value: Maximum generated value
            unit: Measurement unit
        """
        super().__init__(sensor_id)
        self.min_value = min_value
        self.max_value = max_value
        self._unit = unit

    @property
    def unit(self) -> str:
        return self._unit

    def connect(self) -> bool:
        """Connect (always succeeds for mock)."""
        self._is_connected = True
        logger.info(f"Mock sensor '{self.sensor_id}' connected")
        return True

    def disconnect(self) -> None:
        """Disconnect the mock sensor."""
        self._is_connected = False
        logger.info(f"Mock sensor '{self.sensor_id}' disconnected")

    def read(self) -> SensorData:
        """Generate a random reading."""
        value = np.random.uniform(self.min_value, self.max_value)

        return SensorData(
            sensor_id=self.sensor_id,
            value=value,
            unit=self._unit,
            metadata={"type": "mock"},
        )


class SensorReader:
    """
    Multi-sensor data acquisition system.

    Provides:
    - Multiple sensor management
    - Continuous polling
    - Data buffering
    - Callback support

    Example:
        reader = SensorReader()
        reader.add_sensor(MockSensor("temp", 20, 30, "°C"))
        reader.start_polling(interval=0.1)
        time.sleep(5)
        data = reader.get_buffer("temp")
        reader.stop_polling()
    """

    def __init__(self, buffer_size: int = 1000):
        """
        Initialize the sensor reader.

        Args:
            buffer_size: Maximum readings to buffer per sensor
        """
        self.buffer_size = buffer_size
        self._sensors: dict[str, BaseSensor] = {}
        self._buffers: dict[str, deque] = {}
        self._callbacks: list[Callable[[SensorData], None]] = []

        self._polling = False
        self._poll_thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()

    def add_sensor(self, sensor: BaseSensor) -> None:
        """
        Add a sensor to the reader.

        Args:
            sensor: Sensor instance
        """
        with self._lock:
            self._sensors[sensor.sensor_id] = sensor
            self._buffers[sensor.sensor_id] = deque(maxlen=self.buffer_size)
            logger.info(f"Added sensor: {sensor.sensor_id}")

    def remove_sensor(self, sensor_id: str) -> bool:
        """
        Remove a sensor from the reader.

        Args:
            sensor_id: Sensor identifier

        Returns:
            True if removed
        """
        with self._lock:
            if sensor_id in self._sensors:
                del self._sensors[sensor_id]
                del self._buffers[sensor_id]
                logger.info(f"Removed sensor: {sensor_id}")
                return True
            return False

    def connect_all(self) -> dict[str, bool]:
        """
        Connect all sensors.

        Returns:
            Dictionary of sensor_id -> success
        """
        results = {}
        with self._lock:
            for sensor_id, sensor in self._sensors.items():
                results[sensor_id] = sensor.connect()
        return results

    def disconnect_all(self) -> None:
        """Disconnect all sensors."""
        with self._lock:
            for sensor in self._sensors.values():
                sensor.disconnect()

    def read_sensor(self, sensor_id: str) -> Optional[SensorData]:
        """
        Read from a specific sensor.

        Args:
            sensor_id: Sensor identifier

        Returns:
            SensorData or None
        """
        with self._lock:
            if sensor_id not in self._sensors:
                return None

            data = self._sensors[sensor_id].read()
            self._buffers[sensor_id].append(data)

            for callback in self._callbacks:
                try:
                    callback(data)
                except Exception as e:
                    logger.error(f"Callback error: {e}")

            return data

    def read_all(self) -> dict[str, SensorData]:
        """
        Read from all sensors.

        Returns:
            Dictionary of sensor_id -> SensorData
        """
        results = {}
        with self._lock:
            for sensor_id in self._sensors:
                data = self.read_sensor(sensor_id)
                if data:
                    results[sensor_id] = data
        return results

    def get_buffer(self, sensor_id: str) -> list[SensorData]:
        """
        Get buffered readings for a sensor.

        Args:
            sensor_id: Sensor identifier

        Returns:
            List of SensorData
        """
        with self._lock:
            if sensor_id in self._buffers:
                return list(self._buffers[sensor_id])
            return []

    def clear_buffer(self, sensor_id: Optional[str] = None) -> None:
        """
        Clear sensor buffers.

        Args:
            sensor_id: Specific sensor or None for all
        """
        with self._lock:
            if sensor_id:
                if sensor_id in self._buffers:
                    self._buffers[sensor_id].clear()
            else:
                for buffer in self._buffers.values():
                    buffer.clear()

    def add_callback(self, callback: Callable[[SensorData], None]) -> None:
        """Add a callback for new readings."""
        self._callbacks.append(callback)

    def start_polling(self, interval: float = 0.1) -> None:
        """
        Start continuous polling.

        Args:
            interval: Polling interval in seconds
        """
        if self._polling:
            return

        self._polling = True
        self._poll_thread = threading.Thread(
            target=self._poll_loop,
            args=(interval,),
            daemon=True,
        )
        self._poll_thread.start()
        logger.info(f"Started polling at {interval}s interval")

    def stop_polling(self) -> None:
        """Stop continuous polling."""
        self._polling = False
        if self._poll_thread:
            self._poll_thread.join(timeout=2.0)
        logger.info("Stopped polling")

    def _poll_loop(self, interval: float) -> None:
        """Internal polling loop."""
        while self._polling:
            self.read_all()
            time.sleep(interval)

    @property
    def sensor_ids(self) -> list[str]:
        """List of registered sensor IDs."""
        with self._lock:
            return list(self._sensors.keys())


# =============================================================================
# File-based Sensors
# =============================================================================


class FileSensor(BaseSensor):
    """
    Sensor that reads from a file (CSV, JSON, or text).

    Simulates real-time sensor by reading values sequentially.
    Useful for replaying recorded sensor data.
    """

    def __init__(
        self,
        sensor_id: str,
        file_path: str,
        unit: str = "units",
        column: Optional[str] = None,
        loop: bool = True,
    ):
        """
        Initialize the file sensor.

        Args:
            sensor_id: Sensor identifier
            file_path: Path to data file
            unit: Measurement unit
            column: Column name for CSV/JSON (None = first data column)
            loop: Loop back to start when reaching end
        """
        super().__init__(sensor_id)
        self.file_path = file_path
        self._unit = unit
        self.column = column
        self.loop = loop

        self._data: list[float] = []
        self._timestamps: list[datetime] = []
        self._index: int = 0

    @property
    def unit(self) -> str:
        return self._unit

    def connect(self) -> bool:
        """Load data from file."""
        try:
            self._load_data()
            self._is_connected = True
            logger.info(f"FileSensor '{self.sensor_id}' loaded {len(self._data)} values from {self.file_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load file sensor data: {e}")
            return False

    def disconnect(self) -> None:
        """Clear data and disconnect."""
        self._data = []
        self._timestamps = []
        self._index = 0
        self._is_connected = False
        logger.info(f"FileSensor '{self.sensor_id}' disconnected")

    def read(self) -> SensorData:
        """Read next value from file data."""
        if not self._data:
            return SensorData(
                sensor_id=self.sensor_id,
                value=0.0,
                unit=self._unit,
                metadata={"error": "no_data"},
            )

        value = self._data[self._index]
        timestamp = self._timestamps[self._index] if self._timestamps else datetime.now()

        self._index += 1

        if self._index >= len(self._data):
            if self.loop:
                self._index = 0
            else:
                self._index = len(self._data) - 1

        return SensorData(
            sensor_id=self.sensor_id,
            value=value,
            timestamp=timestamp,
            unit=self._unit,
            metadata={
                "source": self.file_path,
                "index": self._index,
                "total": len(self._data),
            },
        )

    def _load_data(self) -> None:
        """Load data from file based on extension."""
        import csv
        import json
        from pathlib import Path

        path = Path(self.file_path)

        if path.suffix.lower() == ".csv":
            self._load_csv()
        elif path.suffix.lower() == ".json":
            self._load_json()
        else:
            self._load_text()

    def _load_csv(self) -> None:
        """Load data from CSV file."""
        import csv

        with open(self.file_path, "r", newline="") as f:
            reader = csv.DictReader(f)

            if not reader.fieldnames:
                raise ValueError("CSV file has no columns")

            # Determine value column
            if self.column and self.column in reader.fieldnames:
                value_col = self.column
            else:
                # Use first numeric-looking column
                value_col = reader.fieldnames[0]
                for name in reader.fieldnames:
                    if name.lower() not in ("time", "timestamp", "date", "index"):
                        value_col = name
                        break

            # Check for timestamp column
            time_col = None
            for name in reader.fieldnames:
                if name.lower() in ("time", "timestamp", "date", "datetime"):
                    time_col = name
                    break

            for row in reader:
                try:
                    value = float(row[value_col])
                    self._data.append(value)

                    if time_col and row.get(time_col):
                        try:
                            ts = datetime.fromisoformat(row[time_col])
                            self._timestamps.append(ts)
                        except ValueError:
                            self._timestamps.append(datetime.now())
                    else:
                        self._timestamps.append(datetime.now())

                except (ValueError, KeyError):
                    continue

    def _load_json(self) -> None:
        """Load data from JSON file."""
        import json

        with open(self.file_path, "r") as f:
            data = json.load(f)

        # Handle different JSON structures
        if isinstance(data, list):
            # Array of values or objects
            for item in data:
                if isinstance(item, (int, float)):
                    self._data.append(float(item))
                    self._timestamps.append(datetime.now())
                elif isinstance(item, dict):
                    # Extract value
                    value = None
                    if self.column and self.column in item:
                        value = item[self.column]
                    else:
                        for key in ["value", "v", "data", "reading"]:
                            if key in item:
                                value = item[key]
                                break
                        if value is None and item:
                            value = list(item.values())[0]

                    if value is not None:
                        try:
                            self._data.append(float(value))
                        except (TypeError, ValueError):
                            continue

                    # Extract timestamp
                    ts = datetime.now()
                    for key in ["time", "timestamp", "t", "datetime"]:
                        if key in item:
                            try:
                                ts = datetime.fromisoformat(str(item[key]))
                            except ValueError:
                                pass
                            break
                    self._timestamps.append(ts)

        elif isinstance(data, dict):
            # Dictionary with values array
            values = data.get("values", data.get("data", data.get("readings", [])))
            if isinstance(values, list):
                for v in values:
                    try:
                        self._data.append(float(v))
                        self._timestamps.append(datetime.now())
                    except (TypeError, ValueError):
                        continue

    def _load_text(self) -> None:
        """Load data from plain text file (one value per line)."""
        with open(self.file_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                try:
                    value = float(line.split()[0])
                    self._data.append(value)
                    self._timestamps.append(datetime.now())
                except (ValueError, IndexError):
                    continue


# =============================================================================
# File reading utilities
# =============================================================================


def read_csv_timeseries(
    file_path: str,
    value_column: Optional[str] = None,
    time_column: Optional[str] = None,
) -> tuple[list[float], list[datetime]]:
    """
    Read time series data from a CSV file.

    Args:
        file_path: Path to CSV file
        value_column: Name of value column (None = auto-detect)
        time_column: Name of timestamp column (None = auto-detect)

    Returns:
        Tuple of (values, timestamps)
    """
    import csv

    values: list[float] = []
    timestamps: list[datetime] = []

    with open(file_path, "r", newline="") as f:
        reader = csv.DictReader(f)

        if not reader.fieldnames:
            return values, timestamps

        # Auto-detect columns
        if value_column is None:
            for name in reader.fieldnames:
                if name.lower() not in ("time", "timestamp", "date", "index", "datetime"):
                    value_column = name
                    break
            if value_column is None:
                value_column = reader.fieldnames[0]

        if time_column is None:
            for name in reader.fieldnames:
                if name.lower() in ("time", "timestamp", "date", "datetime"):
                    time_column = name
                    break

        for row in reader:
            try:
                values.append(float(row[value_column]))

                if time_column and row.get(time_column):
                    try:
                        timestamps.append(datetime.fromisoformat(row[time_column]))
                    except ValueError:
                        timestamps.append(datetime.now())
                else:
                    timestamps.append(datetime.now())
            except (ValueError, KeyError):
                continue

    logger.info(f"Read {len(values)} values from {file_path}")
    return values, timestamps


def read_json_timeseries(
    file_path: str,
    value_key: Optional[str] = None,
    time_key: Optional[str] = None,
) -> tuple[list[float], list[datetime]]:
    """
    Read time series data from a JSON file.

    Supports:
    - Array of numbers: [1.0, 2.0, 3.0]
    - Array of objects: [{"value": 1.0, "time": "..."}, ...]
    - Object with values array: {"values": [1.0, 2.0], "timestamps": [...]}

    Args:
        file_path: Path to JSON file
        value_key: Key for value in objects (None = auto-detect)
        time_key: Key for timestamp in objects (None = auto-detect)

    Returns:
        Tuple of (values, timestamps)
    """
    import json

    values: list[float] = []
    timestamps: list[datetime] = []

    with open(file_path, "r") as f:
        data = json.load(f)

    if isinstance(data, list):
        for item in data:
            if isinstance(item, (int, float)):
                values.append(float(item))
                timestamps.append(datetime.now())
            elif isinstance(item, dict):
                # Extract value
                v = None
                if value_key and value_key in item:
                    v = item[value_key]
                else:
                    for key in ["value", "v", "data", "reading", "y"]:
                        if key in item:
                            v = item[key]
                            break

                if v is not None:
                    try:
                        values.append(float(v))
                    except (TypeError, ValueError):
                        continue

                    # Extract timestamp
                    ts = datetime.now()
                    t_key = time_key
                    if t_key is None:
                        for key in ["time", "timestamp", "t", "datetime", "x"]:
                            if key in item:
                                t_key = key
                                break

                    if t_key and t_key in item:
                        try:
                            ts = datetime.fromisoformat(str(item[t_key]))
                        except ValueError:
                            pass

                    timestamps.append(ts)

    elif isinstance(data, dict):
        # Extract values array
        vals = None
        if value_key and value_key in data:
            vals = data[value_key]
        else:
            for key in ["values", "data", "readings", "y"]:
                if key in data and isinstance(data[key], list):
                    vals = data[key]
                    break

        if vals:
            for v in vals:
                try:
                    values.append(float(v))
                except (TypeError, ValueError):
                    continue

        # Extract timestamps array
        times = None
        if time_key and time_key in data:
            times = data[time_key]
        else:
            for key in ["timestamps", "times", "t", "x"]:
                if key in data and isinstance(data[key], list):
                    times = data[key]
                    break

        if times and len(times) == len(values):
            for t in times:
                try:
                    timestamps.append(datetime.fromisoformat(str(t)))
                except ValueError:
                    timestamps.append(datetime.now())
        else:
            timestamps = [datetime.now() for _ in values]

    logger.info(f"Read {len(values)} values from {file_path}")
    return values, timestamps
