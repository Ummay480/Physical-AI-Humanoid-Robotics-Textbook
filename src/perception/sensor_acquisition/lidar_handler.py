"""
LIDAR sensor handler for acquiring and processing LIDAR scan data.
"""
import numpy as np
from typing import Optional, Union
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, PointCloud2
from sensor_msgs_py import point_cloud2

from .base_sensor import BaseSensorHandler
from ..common.data_types import SensorType, SensorData, SensorConfig
from ..common.utils import get_current_timestamp


class LidarHandler(BaseSensorHandler):
    """
    Handler for LIDAR sensors that acquires and processes laser scan or point cloud data.
    Supports both 2D LaserScan and 3D PointCloud2 formats.
    """

    def __init__(self, node: Node, sensor_config: SensorConfig):
        """
        Initialize the LIDAR handler.

        Args:
            node: ROS 2 node to use for LIDAR operations
            sensor_config: Configuration for this LIDAR sensor
        """
        super().__init__(node, sensor_config)

        # LIDAR-specific parameters
        self.range_min = self.operational_params.get('range_min', 0.1)  # meters
        self.range_max = self.operational_params.get('range_max', 30.0)  # meters
        self.angle_min = self.operational_params.get('angle_min', -2.356)  # radians (-135°)
        self.angle_max = self.operational_params.get('angle_max', 2.356)  # radians (135°)
        self.angle_increment = self.operational_params.get('angle_increment', 0.0043)  # radians
        self.frame_id = self.operational_params.get('frame_id', f'{self.sensor_id}_frame')

        # Data format: 'laser_scan' or 'point_cloud'
        self.data_format = self.operational_params.get('data_format', 'laser_scan')

        # Subscription to LIDAR topic
        self.subscription = None

        # Latest scan data
        self._latest_scan = None
        self._scan_timestamp = None

        self.node.get_logger().info(
            f"Initialized LidarHandler for {self.sensor_id} "
            f"with range [{self.range_min}, {self.range_max}] meters "
            f"and format {self.data_format}"
        )

    def _setup_sensor(self):
        """
        Set up the LIDAR sensor subscription.
        """
        lidar_topic = self.operational_params.get('topic', f'/{self.sensor_id}/scan')

        from ..nodes.base_node import REAL_TIME_SENSOR_DATA_QOS

        # Subscribe based on data format
        if self.data_format == 'point_cloud':
            self.subscription = self.node.create_subscription(
                PointCloud2,
                lidar_topic,
                self._point_cloud_callback,
                REAL_TIME_SENSOR_DATA_QOS
            )
        else:
            self.subscription = self.node.create_subscription(
                LaserScan,
                lidar_topic,
                self._laser_scan_callback,
                REAL_TIME_SENSOR_DATA_QOS
            )

        self.node.get_logger().info(f"LIDAR handler subscribed to {lidar_topic}")

    def _teardown_sensor(self):
        """
        Tear down the LIDAR sensor subscription.
        """
        if self.subscription:
            self.node.destroy_subscription(self.subscription)
            self.subscription = None

        self.node.get_logger().info(f"LIDAR handler {self.sensor_id} torn down")

    def _laser_scan_callback(self, msg: LaserScan):
        """
        Callback for receiving laser scan data.

        Args:
            msg: ROS LaserScan message
        """
        if not self.is_active:
            return

        try:
            # Convert ranges to numpy array
            ranges = np.array(msg.ranges)
            intensities = np.array(msg.intensities) if msg.intensities else None

            # Store latest scan
            with self._lock:
                self._latest_scan = ranges
                self._scan_timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
                self.last_data_timestamp = self._scan_timestamp

            # Create SensorData object
            sensor_data = SensorData(
                sensor_type=SensorType.LIDAR,
                timestamp=self._scan_timestamp,
                data=ranges,
                frame_id=msg.header.frame_id,
                metadata={
                    'range_min': msg.range_min,
                    'range_max': msg.range_max,
                    'angle_min': msg.angle_min,
                    'angle_max': msg.angle_max,
                    'angle_increment': msg.angle_increment,
                    'time_increment': msg.time_increment,
                    'scan_time': msg.scan_time,
                    'intensities': intensities.tolist() if intensities is not None else None,
                    'data_format': 'laser_scan'
                },
                id=f"{self.sensor_id}_{int(self._scan_timestamp * 1e9)}"
            )

            # Call data callback if set
            if self.data_callback:
                self.data_callback(sensor_data)

        except Exception as e:
            self.node.get_logger().error(f"Error processing laser scan: {e}")
            if self.error_callback:
                self.error_callback(self.sensor_id, e)

    def _point_cloud_callback(self, msg: PointCloud2):
        """
        Callback for receiving point cloud data.

        Args:
            msg: ROS PointCloud2 message
        """
        if not self.is_active:
            return

        try:
            # Convert PointCloud2 to numpy array
            points = point_cloud2.read_points_numpy(msg, field_names=("x", "y", "z"))

            # Store latest point cloud
            with self._lock:
                self._latest_scan = points
                self._scan_timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
                self.last_data_timestamp = self._scan_timestamp

            # Create SensorData object
            sensor_data = SensorData(
                sensor_type=SensorType.LIDAR,
                timestamp=self._scan_timestamp,
                data=points,
                frame_id=msg.header.frame_id,
                metadata={
                    'height': msg.height,
                    'width': msg.width,
                    'fields': [field.name for field in msg.fields],
                    'is_bigendian': msg.is_bigendian,
                    'point_step': msg.point_step,
                    'row_step': msg.row_step,
                    'is_dense': msg.is_dense,
                    'data_format': 'point_cloud'
                },
                id=f"{self.sensor_id}_{int(self._scan_timestamp * 1e9)}"
            )

            # Call data callback if set
            if self.data_callback:
                self.data_callback(sensor_data)

        except Exception as e:
            self.node.get_logger().error(f"Error processing point cloud: {e}")
            if self.error_callback:
                self.error_callback(self.sensor_id, e)

    def get_latest_scan(self) -> Optional[np.ndarray]:
        """
        Get the latest LIDAR scan data.

        Returns:
            Latest scan as numpy array or None if no scan available
        """
        with self._lock:
            return self._latest_scan.copy() if self._latest_scan is not None else None

    def get_latest_timestamp(self) -> Optional[float]:
        """
        Get the timestamp of the latest scan.

        Returns:
            Timestamp of latest scan or None if no scan available
        """
        with self._lock:
            return self._scan_timestamp

    def get_scan_as_cartesian(self) -> Optional[np.ndarray]:
        """
        Convert the latest laser scan to Cartesian coordinates (x, y).

        Returns:
            Numpy array of shape (N, 2) with x, y coordinates or None if unavailable
        """
        with self._lock:
            if self._latest_scan is None or self.data_format != 'laser_scan':
                return None

            ranges = self._latest_scan

            # Generate angles for each range measurement
            num_readings = len(ranges)
            angles = self.angle_min + np.arange(num_readings) * self.angle_increment

            # Filter out invalid ranges
            valid_mask = (ranges >= self.range_min) & (ranges <= self.range_max)

            # Convert to Cartesian coordinates
            x = ranges[valid_mask] * np.cos(angles[valid_mask])
            y = ranges[valid_mask] * np.sin(angles[valid_mask])

            return np.column_stack((x, y))

    def filter_scan_by_range(self, scan: np.ndarray, min_range: float = None,
                            max_range: float = None) -> np.ndarray:
        """
        Filter scan data by range values.

        Args:
            scan: Scan data to filter
            min_range: Minimum range (default: sensor's range_min)
            max_range: Maximum range (default: sensor's range_max)

        Returns:
            Filtered scan data
        """
        if min_range is None:
            min_range = self.range_min
        if max_range is None:
            max_range = self.range_max

        valid_mask = (scan >= min_range) & (scan <= max_range)
        return scan[valid_mask]

    def validate_scan(self, scan: np.ndarray) -> bool:
        """
        Validate that a scan meets expected criteria.

        Args:
            scan: Scan data to validate

        Returns:
            True if scan is valid, False otherwise
        """
        if scan is None:
            return False

        # Check scan is not empty
        if scan.size == 0:
            return False

        # For laser scans, check that at least some readings are in valid range
        if self.data_format == 'laser_scan':
            valid_readings = np.sum((scan >= self.range_min) & (scan <= self.range_max))
            if valid_readings < len(scan) * 0.1:  # At least 10% valid readings
                self.node.get_logger().warning(
                    f"Less than 10% of scan readings are valid ({valid_readings}/{len(scan)})"
                )
                return False

        return True

    def get_health_status(self) -> dict:
        """
        Get health status of the LIDAR sensor.

        Returns:
            Dictionary containing health status information
        """
        with self._lock:
            current_time = get_current_timestamp()
            time_since_last_data = None

            if self.last_data_timestamp:
                time_since_last_data = current_time - self.last_data_timestamp

            is_healthy = (
                self.is_active and
                self._latest_scan is not None and
                (time_since_last_data is None or time_since_last_data < self.timeout)
            )

            return {
                'sensor_id': self.sensor_id,
                'sensor_type': 'lidar',
                'is_active': self.is_active,
                'is_healthy': is_healthy,
                'last_data_timestamp': self.last_data_timestamp,
                'time_since_last_data': time_since_last_data,
                'has_data': self._latest_scan is not None,
                'data_format': self.data_format,
                'range_min': self.range_min,
                'range_max': self.range_max,
                'frame_id': self.frame_id
            }
