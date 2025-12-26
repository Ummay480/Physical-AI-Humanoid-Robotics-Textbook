"""
Sensor manager for coordinating multiple sensor handlers.
"""
from typing import Dict, List, Optional, Callable
import threading
from rclpy.node import Node

from .base_sensor import BaseSensorHandler
from .camera_handler import CameraHandler
from .lidar_handler import LidarHandler
from .imu_handler import IMUHandler
from ..common.data_types import SensorType, SensorData, SensorConfig
from ..common.config_handler import SensorConfigHandler
from ..common.utils import get_current_timestamp


class SensorManager:
    """
    Manages multiple sensor handlers and coordinates their operation.
    Provides centralized control for starting, stopping, and monitoring sensors.
    """

    def __init__(self, node: Node, config_handler: SensorConfigHandler):
        """
        Initialize the sensor manager.

        Args:
            node: ROS 2 node to use for sensor operations
            config_handler: Handler for sensor configurations
        """
        self.node = node
        self.config_handler = config_handler

        # Dictionary of sensor handlers indexed by sensor ID
        self.sensor_handlers: Dict[str, BaseSensorHandler] = {}

        # Lock for thread-safe operations
        self._lock = threading.Lock()

        # Callbacks
        self.data_callback = None
        self.error_callback = None

        # Statistics
        self._stats = {
            'total_sensors': 0,
            'active_sensors': 0,
            'total_data_received': 0,
            'errors_count': 0
        }

        self.node.get_logger().info("Initialized SensorManager")

    def add_sensor(self, sensor_config: SensorConfig) -> bool:
        """
        Add a new sensor to the manager.

        Args:
            sensor_config: Configuration for the sensor to add

        Returns:
            True if sensor was added successfully, False otherwise
        """
        try:
            with self._lock:
                # Check if sensor already exists
                if sensor_config.sensor_id in self.sensor_handlers:
                    self.node.get_logger().warning(
                        f"Sensor {sensor_config.sensor_id} already exists"
                    )
                    return False

                # Create appropriate handler based on sensor type
                handler = self._create_sensor_handler(sensor_config)
                if handler is None:
                    return False

                # Set callbacks
                handler.set_data_callback(self._handle_sensor_data)
                handler.set_error_callback(self._handle_sensor_error)

                # Add to handlers
                self.sensor_handlers[sensor_config.sensor_id] = handler
                self._stats['total_sensors'] += 1

                self.node.get_logger().info(
                    f"Added sensor {sensor_config.sensor_id} of type {sensor_config.sensor_type.value}"
                )

                return True

        except Exception as e:
            self.node.get_logger().error(f"Error adding sensor: {e}")
            return False

    def _create_sensor_handler(self, sensor_config: SensorConfig) -> Optional[BaseSensorHandler]:
        """
        Create a sensor handler based on the sensor type.

        Args:
            sensor_config: Configuration for the sensor

        Returns:
            Sensor handler instance or None if type is unsupported
        """
        sensor_type = sensor_config.sensor_type

        if sensor_type == SensorType.CAMERA:
            return CameraHandler(self.node, sensor_config)
        elif sensor_type == SensorType.LIDAR:
            return LidarHandler(self.node, sensor_config)
        elif sensor_type == SensorType.IMU:
            return IMUHandler(self.node, sensor_config)
        else:
            self.node.get_logger().error(
                f"Unsupported sensor type: {sensor_type.value}"
            )
            return None

    def remove_sensor(self, sensor_id: str) -> bool:
        """
        Remove a sensor from the manager.

        Args:
            sensor_id: ID of the sensor to remove

        Returns:
            True if sensor was removed successfully, False otherwise
        """
        try:
            with self._lock:
                if sensor_id not in self.sensor_handlers:
                    self.node.get_logger().warning(
                        f"Sensor {sensor_id} not found"
                    )
                    return False

                # Stop the sensor if active
                handler = self.sensor_handlers[sensor_id]
                if handler.is_operational():
                    handler.stop()

                # Remove from handlers
                del self.sensor_handlers[sensor_id]
                self._stats['total_sensors'] -= 1

                self.node.get_logger().info(f"Removed sensor {sensor_id}")

                return True

        except Exception as e:
            self.node.get_logger().error(f"Error removing sensor: {e}")
            return False

    def start_sensor(self, sensor_id: str) -> bool:
        """
        Start a specific sensor.

        Args:
            sensor_id: ID of the sensor to start

        Returns:
            True if sensor was started successfully, False otherwise
        """
        try:
            with self._lock:
                if sensor_id not in self.sensor_handlers:
                    self.node.get_logger().error(
                        f"Sensor {sensor_id} not found"
                    )
                    return False

                handler = self.sensor_handlers[sensor_id]
                if handler.is_operational():
                    self.node.get_logger().warning(
                        f"Sensor {sensor_id} is already running"
                    )
                    return False

                handler.start()
                self._stats['active_sensors'] += 1

                self.node.get_logger().info(f"Started sensor {sensor_id}")

                return True

        except Exception as e:
            self.node.get_logger().error(f"Error starting sensor: {e}")
            return False

    def stop_sensor(self, sensor_id: str) -> bool:
        """
        Stop a specific sensor.

        Args:
            sensor_id: ID of the sensor to stop

        Returns:
            True if sensor was stopped successfully, False otherwise
        """
        try:
            with self._lock:
                if sensor_id not in self.sensor_handlers:
                    self.node.get_logger().error(
                        f"Sensor {sensor_id} not found"
                    )
                    return False

                handler = self.sensor_handlers[sensor_id]
                if not handler.is_operational():
                    self.node.get_logger().warning(
                        f"Sensor {sensor_id} is not running"
                    )
                    return False

                handler.stop()
                self._stats['active_sensors'] -= 1

                self.node.get_logger().info(f"Stopped sensor {sensor_id}")

                return True

        except Exception as e:
            self.node.get_logger().error(f"Error stopping sensor: {e}")
            return False

    def start_all_sensors(self) -> int:
        """
        Start all sensors in the manager.

        Returns:
            Number of sensors started successfully
        """
        started_count = 0

        # Get snapshot of sensor IDs (start_sensor has its own lock)
        with self._lock:
            sensor_ids = list(self.sensor_handlers.keys())

        # Start each sensor (each call is thread-safe)
        for sensor_id in sensor_ids:
            if self.start_sensor(sensor_id):
                started_count += 1

        self.node.get_logger().info(
            f"Started {started_count}/{len(sensor_ids)} sensors"
        )

        return started_count

    def stop_all_sensors(self) -> int:
        """
        Stop all sensors in the manager.

        Returns:
            Number of sensors stopped successfully
        """
        stopped_count = 0

        # Get snapshot of sensor IDs (stop_sensor has its own lock)
        with self._lock:
            sensor_ids = list(self.sensor_handlers.keys())

        # Stop each sensor (each call is thread-safe)
        for sensor_id in sensor_ids:
            if self.stop_sensor(sensor_id):
                stopped_count += 1

        self.node.get_logger().info(
            f"Stopped {stopped_count}/{len(sensor_ids)} sensors"
        )

        return stopped_count

    def get_sensor_handler(self, sensor_id: str) -> Optional[BaseSensorHandler]:
        """
        Get a sensor handler by ID.

        Args:
            sensor_id: ID of the sensor

        Returns:
            Sensor handler or None if not found
        """
        with self._lock:
            return self.sensor_handlers.get(sensor_id)

    def get_all_sensor_ids(self) -> List[str]:
        """
        Get IDs of all managed sensors.

        Returns:
            List of sensor IDs
        """
        with self._lock:
            return list(self.sensor_handlers.keys())

    def get_active_sensor_ids(self) -> List[str]:
        """
        Get IDs of all active sensors.

        Returns:
            List of active sensor IDs
        """
        with self._lock:
            return [
                sensor_id for sensor_id, handler in self.sensor_handlers.items()
                if handler.is_operational()
            ]

    def get_health_status(self) -> Dict:
        """
        Get health status of all sensors.

        Returns:
            Dictionary containing health status for each sensor
        """
        with self._lock:
            health_status = {
                'timestamp': get_current_timestamp(),
                'total_sensors': self._stats['total_sensors'],
                'active_sensors': self._stats['active_sensors'],
                'sensors': {}
            }

            for sensor_id, handler in self.sensor_handlers.items():
                if hasattr(handler, 'get_health_status'):
                    health_status['sensors'][sensor_id] = handler.get_health_status()

            return health_status

    def set_data_callback(self, callback: Callable[[str, SensorData], None]):
        """
        Set callback for sensor data.

        Args:
            callback: Function to call when sensor data is received.
                     Takes sensor_id and SensorData as arguments.
        """
        self.data_callback = callback

    def set_error_callback(self, callback: Callable[[str, str, Exception], None]):
        """
        Set callback for sensor errors.

        Args:
            callback: Function to call when sensor error occurs.
                     Takes sensor_id, error message, and exception as arguments.
        """
        self.error_callback = callback

    def _handle_sensor_data(self, sensor_data: SensorData):
        """
        Internal handler for sensor data from individual sensors.

        Args:
            sensor_data: Sensor data received
        """
        self._stats['total_data_received'] += 1

        # Forward to external callback if set
        if self.data_callback:
            # Extract sensor_id from the data ID
            sensor_id = sensor_data.id.split('_')[0] if sensor_data.id else 'unknown'
            self.data_callback(sensor_id, sensor_data)

    def _handle_sensor_error(self, sensor_id: str, error: Exception):
        """
        Internal handler for sensor errors.

        Args:
            sensor_id: ID of the sensor with error
            error: Exception that occurred
        """
        self._stats['errors_count'] += 1

        self.node.get_logger().error(
            f"Sensor {sensor_id} error: {error}"
        )

        # Forward to external callback if set
        if self.error_callback:
            self.error_callback(sensor_id, str(error), error)

    def get_statistics(self) -> Dict:
        """
        Get statistics about sensor operations.

        Returns:
            Dictionary containing statistics
        """
        with self._lock:
            return self._stats.copy()

    def load_sensors_from_config(self, sensor_ids: Optional[List[str]] = None) -> int:
        """
        Load sensors from configuration files.

        Args:
            sensor_ids: List of sensor IDs to load. If None, loads all available.

        Returns:
            Number of sensors loaded successfully
        """
        loaded_count = 0

        # Get sensor IDs to load
        if sensor_ids is None:
            sensor_ids = self.config_handler.list_sensor_configs()

        for sensor_id in sensor_ids:
            config = self.config_handler.get_sensor_config(sensor_id)
            if config and self.add_sensor(config):
                loaded_count += 1

        self.node.get_logger().info(
            f"Loaded {loaded_count}/{len(sensor_ids)} sensors from configuration"
        )

        return loaded_count
