#!/usr/bin/env python3
"""
Sensor Data Acquisition Node for the perception system.

This node manages multiple sensor handlers and publishes their data to ROS topics.
"""
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import json

from .base_node import BasePerceptionNode, REAL_TIME_SENSOR_DATA_QOS
from ..sensor_acquisition.sensor_manager import SensorManager
from ..common.config_handler import SensorConfigHandler
from ..common.data_types import SensorData, SensorType
from ..common.utils import get_current_timestamp, timestamp_to_ros_time

# Import ROS message types
from sensor_msgs.msg import Image, LaserScan, PointCloud2, Imu
from cv_bridge import CvBridge


class SensorAcquisitionNode(BasePerceptionNode):
    """
    ROS 2 node for acquiring data from multiple sensors.

    Manages camera, lidar, and IMU sensors, and publishes their data to ROS topics.
    """

    def __init__(self):
        """Initialize the sensor acquisition node."""
        super().__init__('sensor_acquisition_node')

        # Configuration handler
        config_dir = self.get_parameter('config_dir').get_parameter_value().string_value \
            if self.has_parameter('config_dir') else 'config'
        self.declare_parameter('config_dir', config_dir)

        self.config_handler = SensorConfigHandler(config_dir)

        # Sensor manager
        self.sensor_manager = SensorManager(self, self.config_handler)

        # Set callbacks for sensor data and errors
        self.sensor_manager.set_data_callback(self._on_sensor_data)
        self.sensor_manager.set_error_callback(self._on_sensor_error)

        # Publishers for different sensor types
        self.publishers = {}

        # CV Bridge for image conversion
        self.cv_bridge = CvBridge()

        # Status publisher
        self.status_publisher = self.create_publisher(
            String,
            '/perception/system_status',
            self.get_qos_profile('status')
        )

        # Status timer (1 Hz)
        self.status_timer = self.create_timer(1.0, self._publish_status)

        # Load and start sensors
        self._initialize_sensors()

        self.get_logger().info('Sensor Acquisition Node initialized')

    def _initialize_sensors(self):
        """Initialize and start all configured sensors."""
        # Declare parameters for sensors to load
        self.declare_parameter('sensors_to_load', ['camera_front', 'lidar_3d', 'imu_main'])

        sensor_ids = self.get_parameter('sensors_to_load').value

        # Load sensors from configuration
        loaded_count = self.sensor_manager.load_sensors_from_config(sensor_ids)

        self.get_logger().info(f'Loaded {loaded_count} sensors')

        # Start all sensors
        started_count = self.sensor_manager.start_all_sensors()

        self.get_logger().info(f'Started {started_count} sensors')

        # Create publishers for each sensor type
        self._create_publishers()

    def _create_publishers(self):
        """Create publishers for each sensor in the manager."""
        for sensor_id in self.sensor_manager.get_all_sensor_ids():
            handler = self.sensor_manager.get_sensor_handler(sensor_id)

            if handler is None:
                continue

            sensor_type = handler.sensor_type

            # Create publisher based on sensor type
            if sensor_type == SensorType.CAMERA:
                topic = f'/perception/{sensor_id}/image_processed'
                self.publishers[sensor_id] = self.create_publisher_with_profile(
                    Image, topic, 'real_time_sensor_data'
                )
                self.get_logger().info(f'Created camera publisher: {topic}')

            elif sensor_type == SensorType.LIDAR:
                topic = f'/perception/{sensor_id}/scan_processed'
                # Use LaserScan for now (could also support PointCloud2)
                self.publishers[sensor_id] = self.create_publisher_with_profile(
                    LaserScan, topic, 'real_time_sensor_data'
                )
                self.get_logger().info(f'Created lidar publisher: {topic}')

            elif sensor_type == SensorType.IMU:
                topic = f'/perception/{sensor_id}/data_processed'
                self.publishers[sensor_id] = self.create_publisher_with_profile(
                    Imu, topic, 'imu_sensor_data'
                )
                self.get_logger().info(f'Created IMU publisher: {topic}')

    def _on_sensor_data(self, sensor_id: str, sensor_data: SensorData):
        """
        Callback for sensor data from the sensor manager.

        Args:
            sensor_id: ID of the sensor that produced the data
            sensor_data: The sensor data
        """
        # Publish data based on sensor type
        if sensor_id not in self.publishers:
            self.get_logger().warning(f'No publisher for sensor {sensor_id}')
            return

        publisher = self.publishers[sensor_id]

        try:
            # Convert SensorData to appropriate ROS message and publish
            if sensor_data.sensor_type == SensorType.CAMERA:
                self._publish_camera_data(publisher, sensor_data)
            elif sensor_data.sensor_type == SensorType.LIDAR:
                self._publish_lidar_data(publisher, sensor_data)
            elif sensor_data.sensor_type == SensorType.IMU:
                self._publish_imu_data(publisher, sensor_data)

        except Exception as e:
            self.get_logger().error(f'Error publishing data from {sensor_id}: {e}')

    def _publish_camera_data(self, publisher, sensor_data: SensorData):
        """
        Publish camera data as ROS Image message.

        Args:
            publisher: ROS publisher
            sensor_data: Camera sensor data
        """
        # Convert numpy array to ROS Image message
        image_format = sensor_data.metadata.get('format', 'bgr8')

        image_msg = self.cv_bridge.cv2_to_imgmsg(
            sensor_data.data,
            encoding=image_format
        )

        image_msg.header.stamp = timestamp_to_ros_time(sensor_data.timestamp)
        image_msg.header.frame_id = sensor_data.frame_id

        publisher.publish(image_msg)

    def _publish_lidar_data(self, publisher, sensor_data: SensorData):
        """
        Publish LIDAR data as ROS LaserScan message.

        Args:
            publisher: ROS publisher
            sensor_data: LIDAR sensor data
        """
        scan_msg = LaserScan()

        scan_msg.header.stamp = timestamp_to_ros_time(sensor_data.timestamp)
        scan_msg.header.frame_id = sensor_data.frame_id

        # Set scan parameters from metadata
        metadata = sensor_data.metadata
        scan_msg.range_min = float(metadata.get('range_min', 0.1))
        scan_msg.range_max = float(metadata.get('range_max', 30.0))
        scan_msg.angle_min = float(metadata.get('angle_min', -2.356))
        scan_msg.angle_max = float(metadata.get('angle_max', 2.356))
        scan_msg.angle_increment = float(metadata.get('angle_increment', 0.0043))
        scan_msg.time_increment = float(metadata.get('time_increment', 0.0))
        scan_msg.scan_time = float(metadata.get('scan_time', 0.1))

        # Set ranges
        scan_msg.ranges = sensor_data.data.tolist()

        # Set intensities if available
        if 'intensities' in metadata and metadata['intensities']:
            scan_msg.intensities = metadata['intensities']

        publisher.publish(scan_msg)

    def _publish_imu_data(self, publisher, sensor_data: SensorData):
        """
        Publish IMU data as ROS Imu message.

        Args:
            publisher: ROS publisher
            sensor_data: IMU sensor data
        """
        imu_msg = Imu()

        imu_msg.header.stamp = timestamp_to_ros_time(sensor_data.timestamp)
        imu_msg.header.frame_id = sensor_data.frame_id

        # Extract data from metadata
        metadata = sensor_data.metadata

        # Orientation (quaternion)
        orientation = metadata.get('orientation', [0, 0, 0, 1])
        imu_msg.orientation.x = float(orientation[0])
        imu_msg.orientation.y = float(orientation[1])
        imu_msg.orientation.z = float(orientation[2])
        imu_msg.orientation.w = float(orientation[3])

        # Angular velocity
        angular_velocity = metadata.get('angular_velocity', [0, 0, 0])
        imu_msg.angular_velocity.x = float(angular_velocity[0])
        imu_msg.angular_velocity.y = float(angular_velocity[1])
        imu_msg.angular_velocity.z = float(angular_velocity[2])

        # Linear acceleration
        linear_acceleration = metadata.get('linear_acceleration', [0, 0, 0])
        imu_msg.linear_acceleration.x = float(linear_acceleration[0])
        imu_msg.linear_acceleration.y = float(linear_acceleration[1])
        imu_msg.linear_acceleration.z = float(linear_acceleration[2])

        # Covariances (flattened 3x3 matrices)
        if 'orientation_covariance' in metadata:
            imu_msg.orientation_covariance = [
                float(x) for row in metadata['orientation_covariance'] for x in row
            ]

        if 'angular_velocity_covariance' in metadata:
            imu_msg.angular_velocity_covariance = [
                float(x) for row in metadata['angular_velocity_covariance'] for x in row
            ]

        if 'linear_acceleration_covariance' in metadata:
            imu_msg.linear_acceleration_covariance = [
                float(x) for row in metadata['linear_acceleration_covariance'] for x in row
            ]

        publisher.publish(imu_msg)

    def _on_sensor_error(self, sensor_id: str, error_msg: str, error: Exception):
        """
        Callback for sensor errors.

        Args:
            sensor_id: ID of the sensor with error
            error_msg: Error message
            error: Exception that occurred
        """
        self.get_logger().error(
            f'Sensor error from {sensor_id}: {error_msg}'
        )

    def _publish_status(self):
        """Publish system status periodically."""
        health_status = self.sensor_manager.get_health_status()
        stats = self.sensor_manager.get_statistics()

        status = {
            'node': 'sensor_acquisition_node',
            'timestamp': health_status['timestamp'],
            'total_sensors': health_status['total_sensors'],
            'active_sensors': health_status['active_sensors'],
            'total_data_received': stats['total_data_received'],
            'errors_count': stats['errors_count'],
            'sensors': health_status['sensors']
        }

        # Determine overall status
        if health_status['active_sensors'] == 0:
            overall_status = 'ERROR'
        elif any(not s['is_healthy'] for s in health_status['sensors'].values()):
            overall_status = 'WARNING'
        else:
            overall_status = 'OK'

        status['overall_status'] = overall_status

        # Publish as JSON string
        status_msg = String()
        status_msg.data = json.dumps(status)
        self.status_publisher.publish(status_msg)

    def shutdown(self):
        """Shutdown the node and stop all sensors."""
        self.get_logger().info('Shutting down sensor acquisition node')

        # Stop all sensors
        self.sensor_manager.stop_all_sensors()

        # Destroy timers
        if self.status_timer:
            self.status_timer.cancel()


def main(args=None):
    """Main entry point for the sensor acquisition node."""
    rclpy.init(args=args)

    node = SensorAcquisitionNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
