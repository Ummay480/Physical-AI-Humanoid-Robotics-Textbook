"""
Base sensor handler for all sensor types.

Provides common functionality for sensor management including lifecycle,
threading, callbacks, and operational state.
"""
import threading
from abc import ABC, abstractmethod
from typing import Optional, Callable
from rclpy.node import Node

from ..common.data_types import SensorConfig, SensorData, SensorType


class BaseSensorHandler(ABC):
    """
    Abstract base class for all sensor handlers.

    Provides common sensor lifecycle management, threading support,
    callback mechanisms, and operational state tracking.
    """

    def __init__(self, node: Node, sensor_config: SensorConfig):
        """
        Initialize the base sensor handler.

        Args:
            node: ROS 2 node to use for sensor operations
            sensor_config: Configuration for this sensor
        """
        self.node = node
        self.sensor_config = sensor_config

        # Extract configuration
        self.sensor_id = sensor_config.sensor_id
        self.sensor_type = sensor_config.sensor_type
        self.operational_params = sensor_config.parameters or {}

        # Operational parameters
        self.timeout = self.operational_params.get('timeout', 1.0)  # seconds
        self.processing_frequency = self.operational_params.get('processing_frequency', 10.0)  # Hz

        # State management
        self.is_active = False
        self.last_data_timestamp: Optional[float] = None

        # Thread safety
        self._lock = threading.Lock()

        # Callbacks
        self.data_callback: Optional[Callable[[SensorData], None]] = None
        self.error_callback: Optional[Callable[[str, Exception], None]] = None

        self.node.get_logger().info(
            f"Initialized BaseSensorHandler for {self.sensor_id} "
            f"(type: {self.sensor_type.value})"
        )

    def start(self):
        """
        Start the sensor and begin data acquisition.
        """
        with self._lock:
            if self.is_active:
                self.node.get_logger().warning(
                    f"Sensor {self.sensor_id} is already active"
                )
                return

            try:
                self._setup_sensor()
                self.is_active = True
                self.node.get_logger().info(f"Started sensor {self.sensor_id}")
            except Exception as e:
                self.node.get_logger().error(
                    f"Failed to start sensor {self.sensor_id}: {e}"
                )
                if self.error_callback:
                    self.error_callback(self.sensor_id, e)
                raise

    def stop(self):
        """
        Stop the sensor and halt data acquisition.
        """
        with self._lock:
            if not self.is_active:
                self.node.get_logger().warning(
                    f"Sensor {self.sensor_id} is not active"
                )
                return

            try:
                self._teardown_sensor()
                self.is_active = False
                self.node.get_logger().info(f"Stopped sensor {self.sensor_id}")
            except Exception as e:
                self.node.get_logger().error(
                    f"Failed to stop sensor {self.sensor_id}: {e}"
                )
                if self.error_callback:
                    self.error_callback(self.sensor_id, e)
                raise

    def is_operational(self) -> bool:
        """
        Check if the sensor is currently operational.

        Returns:
            True if sensor is active and operational, False otherwise
        """
        with self._lock:
            return self.is_active

    def set_data_callback(self, callback: Callable[[SensorData], None]):
        """
        Set the callback function for sensor data.

        Args:
            callback: Function to call when new sensor data is available.
                     Takes SensorData as argument.
        """
        self.data_callback = callback

    def set_error_callback(self, callback: Callable[[str, Exception], None]):
        """
        Set the callback function for sensor errors.

        Args:
            callback: Function to call when sensor error occurs.
                     Takes sensor_id and exception as arguments.
        """
        self.error_callback = callback

    @abstractmethod
    def _setup_sensor(self):
        """
        Set up the sensor (create subscriptions, initialize hardware, etc.).

        This method must be implemented by subclasses to perform sensor-specific
        initialization.
        """
        pass

    @abstractmethod
    def _teardown_sensor(self):
        """
        Tear down the sensor (destroy subscriptions, cleanup hardware, etc.).

        This method must be implemented by subclasses to perform sensor-specific
        cleanup.
        """
        pass

    @abstractmethod
    def get_health_status(self) -> dict:
        """
        Get health status of the sensor.

        This method must be implemented by subclasses to provide sensor-specific
        health information.

        Returns:
            Dictionary containing health status information
        """
        pass
