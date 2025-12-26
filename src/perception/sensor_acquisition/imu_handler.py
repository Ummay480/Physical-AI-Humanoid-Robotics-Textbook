"""
IMU (Inertial Measurement Unit) sensor handler for acquiring orientation and motion data.
"""
import numpy as np
from typing import Optional, Tuple
from rclpy.node import Node
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Quaternion, Vector3

from .base_sensor import BaseSensorHandler
from ..common.data_types import SensorType, SensorData, SensorConfig
from ..common.utils import get_current_timestamp


class IMUHandler(BaseSensorHandler):
    """
    Handler for IMU sensors that acquires orientation, angular velocity, and linear acceleration data.
    """

    def __init__(self, node: Node, sensor_config: SensorConfig):
        """
        Initialize the IMU handler.

        Args:
            node: ROS 2 node to use for IMU operations
            sensor_config: Configuration for this IMU sensor
        """
        super().__init__(node, sensor_config)

        # IMU-specific parameters
        self.linear_acceleration_stddev = self.operational_params.get(
            'linear_acceleration_stddev', 0.017
        )
        self.angular_velocity_stddev = self.operational_params.get(
            'angular_velocity_stddev', 0.001
        )
        self.orientation_stddev = self.operational_params.get(
            'orientation_stddev', 6.66e-05
        )
        self.frame_id = self.operational_params.get('frame_id', f'{self.sensor_id}_frame')

        # Subscription to IMU topic
        self.subscription = None

        # Latest IMU data
        self._latest_orientation = None
        self._latest_angular_velocity = None
        self._latest_linear_acceleration = None
        self._imu_timestamp = None

        # Covariance matrices (for uncertainty)
        self._orientation_covariance = None
        self._angular_velocity_covariance = None
        self._linear_acceleration_covariance = None

        self.node.get_logger().info(
            f"Initialized IMUHandler for {self.sensor_id} with frame {self.frame_id}"
        )

    def _setup_sensor(self):
        """
        Set up the IMU sensor subscription.
        """
        imu_topic = self.operational_params.get('topic', f'/{self.sensor_id}/data')

        from ..nodes.base_node import IMU_SENSOR_DATA_QOS

        self.subscription = self.node.create_subscription(
            Imu,
            imu_topic,
            self._imu_callback,
            IMU_SENSOR_DATA_QOS
        )

        self.node.get_logger().info(f"IMU handler subscribed to {imu_topic}")

    def _teardown_sensor(self):
        """
        Tear down the IMU sensor subscription.
        """
        if self.subscription:
            self.node.destroy_subscription(self.subscription)
            self.subscription = None

        self.node.get_logger().info(f"IMU handler {self.sensor_id} torn down")

    def _imu_callback(self, msg: Imu):
        """
        Callback for receiving IMU data.

        Args:
            msg: ROS Imu message
        """
        if not self.is_active:
            return

        try:
            # Extract orientation (quaternion)
            orientation = np.array([
                msg.orientation.x,
                msg.orientation.y,
                msg.orientation.z,
                msg.orientation.w
            ])

            # Extract angular velocity
            angular_velocity = np.array([
                msg.angular_velocity.x,
                msg.angular_velocity.y,
                msg.angular_velocity.z
            ])

            # Extract linear acceleration
            linear_acceleration = np.array([
                msg.linear_acceleration.x,
                msg.linear_acceleration.y,
                msg.linear_acceleration.z
            ])

            # Extract covariances
            orientation_cov = np.array(msg.orientation_covariance).reshape((3, 3))
            angular_velocity_cov = np.array(msg.angular_velocity_covariance).reshape((3, 3))
            linear_acceleration_cov = np.array(msg.linear_acceleration_covariance).reshape((3, 3))

            # Store latest IMU data
            with self._lock:
                self._latest_orientation = orientation
                self._latest_angular_velocity = angular_velocity
                self._latest_linear_acceleration = linear_acceleration
                self._orientation_covariance = orientation_cov
                self._angular_velocity_covariance = angular_velocity_cov
                self._linear_acceleration_covariance = linear_acceleration_cov
                self._imu_timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
                self.last_data_timestamp = self._imu_timestamp

            # Combine all IMU data into a single array
            imu_data = np.concatenate([
                orientation,
                angular_velocity,
                linear_acceleration
            ])

            # Create SensorData object
            sensor_data = SensorData(
                sensor_type=SensorType.IMU,
                timestamp=self._imu_timestamp,
                data=imu_data,
                frame_id=msg.header.frame_id,
                metadata={
                    'orientation': orientation.tolist(),
                    'angular_velocity': angular_velocity.tolist(),
                    'linear_acceleration': linear_acceleration.tolist(),
                    'orientation_covariance': orientation_cov.tolist(),
                    'angular_velocity_covariance': angular_velocity_cov.tolist(),
                    'linear_acceleration_covariance': linear_acceleration_cov.tolist()
                },
                id=f"{self.sensor_id}_{int(self._imu_timestamp * 1e9)}"
            )

            # Call data callback if set
            if self.data_callback:
                self.data_callback(sensor_data)

        except Exception as e:
            self.node.get_logger().error(f"Error processing IMU data: {e}")
            if self.error_callback:
                self.error_callback(self.sensor_id, e)

    def get_latest_orientation(self) -> Optional[np.ndarray]:
        """
        Get the latest orientation as a quaternion.

        Returns:
            Quaternion [x, y, z, w] or None if unavailable
        """
        with self._lock:
            return self._latest_orientation.copy() if self._latest_orientation is not None else None

    def get_latest_angular_velocity(self) -> Optional[np.ndarray]:
        """
        Get the latest angular velocity.

        Returns:
            Angular velocity [x, y, z] or None if unavailable
        """
        with self._lock:
            return self._latest_angular_velocity.copy() if self._latest_angular_velocity is not None else None

    def get_latest_linear_acceleration(self) -> Optional[np.ndarray]:
        """
        Get the latest linear acceleration.

        Returns:
            Linear acceleration [x, y, z] or None if unavailable
        """
        with self._lock:
            return self._latest_linear_acceleration.copy() if self._latest_linear_acceleration is not None else None

    def get_latest_timestamp(self) -> Optional[float]:
        """
        Get the timestamp of the latest IMU data.

        Returns:
            Timestamp of latest data or None if unavailable
        """
        with self._lock:
            return self._imu_timestamp

    def get_euler_angles(self) -> Optional[Tuple[float, float, float]]:
        """
        Convert the latest orientation quaternion to Euler angles (roll, pitch, yaw).

        Returns:
            Tuple of (roll, pitch, yaw) in radians or None if unavailable
        """
        with self._lock:
            if self._latest_orientation is None:
                return None

            x, y, z, w = self._latest_orientation

            # Roll (x-axis rotation)
            sinr_cosp = 2 * (w * x + y * z)
            cosr_cosp = 1 - 2 * (x * x + y * y)
            roll = np.arctan2(sinr_cosp, cosr_cosp)

            # Pitch (y-axis rotation)
            sinp = 2 * (w * y - z * x)
            if abs(sinp) >= 1:
                pitch = np.copysign(np.pi / 2, sinp)  # Use 90 degrees if out of range
            else:
                pitch = np.arcsin(sinp)

            # Yaw (z-axis rotation)
            siny_cosp = 2 * (w * z + x * y)
            cosy_cosp = 1 - 2 * (y * y + z * z)
            yaw = np.arctan2(siny_cosp, cosy_cosp)

            return (roll, pitch, yaw)

    def validate_imu_data(self) -> bool:
        """
        Validate that IMU data meets expected criteria.

        Returns:
            True if IMU data is valid, False otherwise
        """
        with self._lock:
            if self._latest_orientation is None:
                return False

            # Validate quaternion is normalized
            quat_norm = np.linalg.norm(self._latest_orientation)
            if not np.isclose(quat_norm, 1.0, atol=0.01):
                self.node.get_logger().warning(
                    f"Quaternion not normalized: norm = {quat_norm}"
                )
                return False

            # Check for NaN or inf values
            if not np.all(np.isfinite(self._latest_orientation)):
                self.node.get_logger().warning("Orientation contains NaN or inf")
                return False

            if self._latest_angular_velocity is not None:
                if not np.all(np.isfinite(self._latest_angular_velocity)):
                    self.node.get_logger().warning("Angular velocity contains NaN or inf")
                    return False

            if self._latest_linear_acceleration is not None:
                if not np.all(np.isfinite(self._latest_linear_acceleration)):
                    self.node.get_logger().warning("Linear acceleration contains NaN or inf")
                    return False

            return True

    def get_health_status(self) -> dict:
        """
        Get health status of the IMU sensor.

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
                self._latest_orientation is not None and
                (time_since_last_data is None or time_since_last_data < self.timeout) and
                self.validate_imu_data()
            )

            euler_angles = self.get_euler_angles()

            return {
                'sensor_id': self.sensor_id,
                'sensor_type': 'imu',
                'is_active': self.is_active,
                'is_healthy': is_healthy,
                'last_data_timestamp': self.last_data_timestamp,
                'time_since_last_data': time_since_last_data,
                'has_orientation': self._latest_orientation is not None,
                'has_angular_velocity': self._latest_angular_velocity is not None,
                'has_linear_acceleration': self._latest_linear_acceleration is not None,
                'euler_angles': euler_angles,
                'frame_id': self.frame_id
            }
