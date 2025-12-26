#!/usr/bin/env python3
"""
Sensor Fusion Node for the perception system.

This node subscribes to sensor acquisition outputs and performs basic fusion
using timestamp-based synchronization to validate the fusion pipeline.
"""
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import LaserScan, Imu
import json
import threading
from typing import Dict, Optional

from .base_node import BasePerceptionNode
from ..sensor_fusion.base_fusion_engine import SimpleTimestampFusionEngine
from ..common.data_types import SensorData, SensorType, FusedData
from ..common.utils import get_current_timestamp, timestamp_to_ros_time


class SensorFusionNode(BasePerceptionNode):
    """
    ROS 2 node for sensor fusion operations.

    Subscribes to sensor acquisition outputs and performs basic timestamp-based
    fusion to validate the fusion pipeline without complex algorithms.
    """

    def __init__(self):
        """Initialize the sensor fusion node."""
        super().__init__('sensor_fusion_node')

        # Initialize fusion engine
        self.fusion_engine = SimpleTimestampFusionEngine()

        # Thread lock for data access
        self._lock = threading.Lock()

        # Storage for latest sensor data
        self.latest_sensor_data: Dict[str, SensorData] = {}

        # Publishers
        self.fused_state_publisher = self.create_publisher_with_profile(
            String,  # Using String for JSON serialization of fused data
            '/perception/fused_state',
            'processed_data'
        )

        # Status publisher
        self.status_publisher = self.create_publisher(
            String,
            '/perception/fusion_status',
            self.get_qos_profile('status')
        )

        # Subscribers for sensor acquisition outputs
        self._create_subscribers()

        # Fusion timer (10 Hz)
        self.fusion_timer = self.create_timer(0.1, self._perform_fusion)

        # Status timer (1 Hz)
        self.status_timer = self.create_timer(1.0, self._publish_status)

        self.get_logger().info('Sensor Fusion Node initialized')

    def _create_subscribers(self):
        """Create subscribers for sensor acquisition outputs."""
        # Subscribe to processed sensor data from sensor acquisition node
        # These would be the outputs from sensor_acquisition_node.py

        # LIDAR processed data subscription
        self.lidar_sub = self.create_subscription_with_profile(
            String,  # Using String for JSON serialized sensor data
            '/perception/lidar_3d/scan_processed',
            self._lidar_callback,
            'real_time_sensor_data'
        )

        # IMU processed data subscription
        self.imu_sub = self.create_subscription_with_profile(
            String,  # Using String for JSON serialized sensor data
            '/perception/imu_main/data_processed',
            self._imu_callback,
            'imu_sensor_data'
        )

        self.get_logger().info('Created subscribers for sensor acquisition outputs')

    def _lidar_callback(self, msg: String):
        """
        Callback for LIDAR processed data.

        Args:
            msg: JSON string containing processed LIDAR data
        """
        try:
            # Parse the JSON message to get sensor data
            data_dict = json.loads(msg.data)

            # Convert to SensorData object
            sensor_data = SensorData(
                id=data_dict.get('id', f"lidar_{int(get_current_timestamp() * 1e9)}"),
                sensor_type=SensorType.LIDAR,
                timestamp=data_dict.get('timestamp', get_current_timestamp()),
                data=data_dict.get('data', []),
                frame_id=data_dict.get('frame_id', 'lidar_frame'),
                metadata=data_dict.get('metadata', {}),
                state=data_dict.get('state', 'ACQUIRED')
            )

            # Store the latest data
            with self._lock:
                self.latest_sensor_data['lidar_3d'] = sensor_data
                # Add to fusion engine
                self.fusion_engine.add_sensor_data('lidar_3d', sensor_data)

            self.get_logger().debug(f'Received LIDAR data with timestamp: {sensor_data.timestamp}')

        except Exception as e:
            self.get_logger().error(f'Error processing LIDAR callback: {e}')

    def _imu_callback(self, msg: String):
        """
        Callback for IMU processed data.

        Args:
            msg: JSON string containing processed IMU data
        """
        try:
            # Parse the JSON message to get sensor data
            data_dict = json.loads(msg.data)

            # Convert to SensorData object
            sensor_data = SensorData(
                id=data_dict.get('id', f"imu_{int(get_current_timestamp() * 1e9)}"),
                sensor_type=SensorType.IMU,
                timestamp=data_dict.get('timestamp', get_current_timestamp()),
                data=data_dict.get('data', []),
                frame_id=data_dict.get('frame_id', 'imu_frame'),
                metadata=data_dict.get('metadata', {}),
                state=data_dict.get('state', 'ACQUIRED')
            )

            # Store the latest data
            with self._lock:
                self.latest_sensor_data['imu_main'] = sensor_data
                # Add to fusion engine
                self.fusion_engine.add_sensor_data('imu_main', sensor_data)

            self.get_logger().debug(f'Received IMU data with timestamp: {sensor_data.timestamp}')

        except Exception as e:
            self.get_logger().error(f'Error processing IMU callback: {e}')

    def _perform_fusion(self):
        """Perform sensor fusion on synchronized data."""
        try:
            # Get current time for synchronization
            current_time = get_current_timestamp()

            # Get synchronized data from fusion engine
            synchronized_data = self.fusion_engine.get_synchronized_data(
                timestamp=current_time,
                tolerance=0.1  # 100ms tolerance
            )

            # Check if we have both LIDAR and IMU data
            if 'lidar_3d' in synchronized_data and 'imu_main' in synchronized_data:
                # Perform fusion
                fused_data = self.fusion_engine.fuse_data(synchronized_data)

                if fused_data is not None:
                    # Publish fused data as JSON string
                    fused_msg = String()
                    fused_msg.data = json.dumps({
                        'id': fused_data.id,
                        'timestamp': fused_data.timestamp,
                        'fused_payload': fused_data.fused_payload,
                        'source_sensor_ids': fused_data.source_sensor_ids,
                        'confidence_score': fused_data.confidence_score,
                        'coordinate_frame': fused_data.coordinate_frame
                    })

                    self.fused_state_publisher.publish(fused_msg)
                    self.get_logger().info(
                        f'Published fused data with confidence: {fused_data.confidence_score:.2f}'
                    )
                else:
                    self.get_logger().warning('Fusion failed - no valid fused data produced')
            else:
                # Log if we don't have both sensors available
                available_sensors = list(synchronized_data.keys())
                if available_sensors:
                    self.get_logger().debug(f'Insufficient sensors for fusion. Available: {available_sensors}')
                else:
                    self.get_logger().debug('No synchronized sensor data available')

        except Exception as e:
            self.get_logger().error(f'Error in fusion process: {e}')

    def _publish_status(self):
        """Publish fusion status periodically."""
        try:
            # Get fusion statistics
            stats = self.fusion_engine.get_fusion_statistics()

            status = {
                'node': 'sensor_fusion_node',
                'timestamp': get_current_timestamp(),
                'fusion_stats': stats,
                'latest_sensor_data': {
                    sensor_id: {
                        'timestamp': data.timestamp,
                        'sensor_type': data.sensor_type.value
                    }
                    for sensor_id, data in self.latest_sensor_data.items()
                },
                'overall_status': 'OK'  # This would be computed based on stats
            }

            # Determine overall status
            success_rate = stats.get('success_rate', 0)
            if success_rate < 0.5:
                status['overall_status'] = 'WARNING'
            elif stats['failed_fusions'] > 0:
                status['overall_status'] = 'DEGRADED'
            else:
                status['overall_status'] = 'OK'

            # Publish status as JSON string
            status_msg = String()
            status_msg.data = json.dumps(status)
            self.status_publisher.publish(status_msg)

        except Exception as e:
            self.get_logger().error(f'Error publishing fusion status: {e}')

    def shutdown(self):
        """Shutdown the node."""
        self.get_logger().info('Shutting down sensor fusion node')

        # Cancel timers
        if self.fusion_timer:
            self.fusion_timer.cancel()
        if self.status_timer:
            self.status_timer.cancel()


def main(args=None):
    """Main entry point for the sensor fusion node."""
    rclpy.init(args=args)

    node = SensorFusionNode()

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
