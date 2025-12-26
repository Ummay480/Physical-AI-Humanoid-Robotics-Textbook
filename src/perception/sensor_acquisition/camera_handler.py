"""
Camera sensor handler for acquiring and processing camera data.
"""
import numpy as np
from typing import Optional
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

from .base_sensor import BaseSensorHandler
from ..common.data_types import SensorType, SensorData, SensorConfig
from ..common.utils import get_current_timestamp, timestamp_to_ros_time


class CameraHandler(BaseSensorHandler):
    """
    Handler for camera sensors that acquires and processes camera images.
    Supports standard RGB/BGR cameras and provides image data at configured frequencies.
    """

    def __init__(self, node: Node, sensor_config: SensorConfig):
        """
        Initialize the camera handler.

        Args:
            node: ROS 2 node to use for camera operations
            sensor_config: Configuration for this camera sensor
        """
        super().__init__(node, sensor_config)

        # Camera-specific parameters
        self.resolution = self.operational_params.get('resolution', [640, 480])
        self.format = self.operational_params.get('format', 'bgr8')
        self.frame_id = self.operational_params.get('frame_id', f'{self.sensor_id}_optical_frame')

        # CV Bridge for ROS-OpenCV conversion
        self.cv_bridge = CvBridge()

        # Subscription to camera topic
        self.subscription = None

        # Latest image data
        self._latest_image = None
        self._image_timestamp = None

        self.node.get_logger().info(
            f"Initialized CameraHandler for {self.sensor_id} "
            f"with resolution {self.resolution} and format {self.format}"
        )

    def _setup_sensor(self):
        """
        Set up the camera sensor subscription.
        """
        # Subscribe to camera topic
        camera_topic = self.operational_params.get('topic', f'/{self.sensor_id}/image_raw')

        from ..nodes.base_node import REAL_TIME_SENSOR_DATA_QOS

        self.subscription = self.node.create_subscription(
            Image,
            camera_topic,
            self._image_callback,
            REAL_TIME_SENSOR_DATA_QOS
        )

        self.node.get_logger().info(f"Camera handler subscribed to {camera_topic}")

    def _teardown_sensor(self):
        """
        Tear down the camera sensor subscription.
        """
        if self.subscription:
            self.node.destroy_subscription(self.subscription)
            self.subscription = None

        self.node.get_logger().info(f"Camera handler {self.sensor_id} torn down")

    def _image_callback(self, msg: Image):
        """
        Callback for receiving camera images.

        Args:
            msg: ROS Image message
        """
        if not self.is_active:
            return

        try:
            # Convert ROS Image to OpenCV format
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding=self.format)

            # Store latest image
            with self._lock:
                self._latest_image = cv_image
                self._image_timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
                self.last_data_timestamp = self._image_timestamp

            # Create SensorData object
            sensor_data = SensorData(
                sensor_type=SensorType.CAMERA,
                timestamp=self._image_timestamp,
                data=cv_image,
                frame_id=msg.header.frame_id,
                metadata={
                    'resolution': list(cv_image.shape[:2]),
                    'format': self.format,
                    'encoding': msg.encoding,
                    'step': msg.step,
                    'is_bigendian': msg.is_bigendian
                },
                id=f"{self.sensor_id}_{int(self._image_timestamp * 1e9)}"
            )

            # Call data callback if set
            if self.data_callback:
                self.data_callback(sensor_data)

        except Exception as e:
            self.node.get_logger().error(f"Error processing camera image: {e}")
            if self.error_callback:
                self.error_callback(self.sensor_id, e)

    def get_latest_image(self) -> Optional[np.ndarray]:
        """
        Get the latest camera image.

        Returns:
            Latest image as numpy array or None if no image available
        """
        with self._lock:
            return self._latest_image.copy() if self._latest_image is not None else None

    def get_latest_timestamp(self) -> Optional[float]:
        """
        Get the timestamp of the latest image.

        Returns:
            Timestamp of latest image or None if no image available
        """
        with self._lock:
            return self._image_timestamp

    def capture_image(self) -> Optional[SensorData]:
        """
        Capture a single image from the camera.

        Returns:
            SensorData containing the captured image or None if unavailable
        """
        with self._lock:
            if self._latest_image is None:
                return None

            sensor_data = SensorData(
                sensor_type=SensorType.CAMERA,
                timestamp=self._image_timestamp,
                data=self._latest_image.copy(),
                frame_id=self.frame_id,
                metadata={
                    'resolution': list(self._latest_image.shape[:2]),
                    'format': self.format,
                    'capture_mode': 'single'
                },
                id=f"{self.sensor_id}_{int(self._image_timestamp * 1e9)}"
            )

            return sensor_data

    def validate_image(self, image: np.ndarray) -> bool:
        """
        Validate that an image meets expected criteria.

        Args:
            image: Image to validate

        Returns:
            True if image is valid, False otherwise
        """
        if image is None:
            return False

        # Check image is not empty
        if image.size == 0:
            return False

        # Check image has expected dimensions (height, width, channels)
        if len(image.shape) < 2:
            return False

        # Check resolution matches expected values (with some tolerance)
        expected_height, expected_width = self.resolution
        actual_height, actual_width = image.shape[:2]

        # Allow 10% tolerance in resolution
        height_diff = abs(actual_height - expected_height) / expected_height
        width_diff = abs(actual_width - expected_width) / expected_width

        if height_diff > 0.1 or width_diff > 0.1:
            self.node.get_logger().warning(
                f"Image resolution {actual_width}x{actual_height} differs from "
                f"expected {expected_width}x{expected_height}"
            )

        return True

    def get_health_status(self) -> dict:
        """
        Get health status of the camera sensor.

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
                self._latest_image is not None and
                (time_since_last_data is None or time_since_last_data < self.timeout)
            )

            return {
                'sensor_id': self.sensor_id,
                'sensor_type': 'camera',
                'is_active': self.is_active,
                'is_healthy': is_healthy,
                'last_data_timestamp': self.last_data_timestamp,
                'time_since_last_data': time_since_last_data,
                'has_data': self._latest_image is not None,
                'resolution': self.resolution,
                'format': self.format,
                'frame_id': self.frame_id
            }
