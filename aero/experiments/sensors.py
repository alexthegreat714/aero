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
