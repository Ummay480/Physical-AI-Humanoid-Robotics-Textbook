"""
Perception node for Isaac Sim
Implements T015: Integrate Perception Algorithms with Isaac Sim
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import numpy as np
import cv2
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Header
from cv_bridge import CvBridge
from ..common.base_node import BaseAIRobotNode
from ..common.models.perception_models import SensorData, SensorType
from ..isaac_ros.perception import ObjectDetectionSystem
import torch


class IsaacSimObjectDetectorNode(BaseAIRobotNode):
    """
    ROS 2 node that integrates object detection algorithms with Isaac Sim camera feed.
    Implements perception algorithms that work with Isaac Sim simulation environment.
    """

    def __init__(self):
        super().__init__('isaac_sim_object_detector')

        # Initialize CV bridge for image conversion
        self.bridge = CvBridge()

        # Initialize object detection system
        self.detection_system = ObjectDetectionSystem()

        # Create subscription to Isaac Sim camera feed
        self.subscription = self.create_subscription(
            Image,
            '/isaac_sim/camera/rgb/image_raw',
            self.camera_callback,
            qos_profile=QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                history=HistoryPolicy.KEEP_LAST,
                depth=1
            )
        )

        # Create publisher for detection results
        self.detection_publisher = self.create_publisher(
            # Using a custom message type that follows Module-2 format
            # In a real implementation, this would be a proper ROS message
            'DetectedObjects',  # Placeholder - would be actual message type
            '/isaac_perception/detections',
            qos_profile=QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                history=HistoryPolicy.KEEP_LAST,
                depth=10
            )
        )

        # Create publisher for system status
        self.status_publisher = self.create_publisher(
            'SystemStatus',  # Placeholder - would be actual message type
            '/isaac_perception/status',
            qos_profile=QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                history=HistoryPolicy.KEEP_LAST,
                depth=10
            )
        )

        # Timer for performance monitoring
        self.timer = self.create_timer(1.0, self.publish_status)

        # Performance metrics
        self.frame_count = 0
        self.detection_times = []

        self.get_logger().info('Isaac Sim Object Detector Node initialized')

    def camera_callback(self, msg: Image):
        """
        Callback function for processing camera images from Isaac Sim.

        Args:
            msg: Image message from Isaac Sim camera
        """
        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Start timing for performance measurement
            start_time = self.get_clock().now().nanoseconds / 1e9

            # Run object detection on the image
            detections = self.detection_system.detect_objects(cv_image)

            # Calculate detection time
            end_time = self.get_clock().now().nanoseconds / 1e9
            detection_time = end_time - start_time
            self.detection_times.append(detection_time)

            # Keep only the last 100 timing measurements
            if len(self.detection_times) > 100:
                self.detection_times = self.detection_times[-100:]

            # Publish detection results
            self.publish_detections(detections, msg.header)

            # Update frame count
            self.frame_count += 1

            # Log performance if needed
            if self.frame_count % 30 == 0:  # Log every 30 frames
                avg_time = sum(self.detection_times) / len(self.detection_times) if self.detection_times else 0
                fps = 1.0 / avg_time if avg_time > 0 else 0
                self.get_logger().info(f'Detection performance: {fps:.2f} FPS, avg time: {avg_time*1000:.2f} ms')

        except Exception as e:
            self.get_logger().error(f'Error processing camera image: {str(e)}')

    def publish_detections(self, detections, header: Header):
        """
        Publish detection results to ROS topic.

        Args:
            detections: List of detected objects with bounding boxes
            header: Header from original image message
        """
        # In a real implementation, this would create and publish a proper ROS message
        # For now, we'll just log the detections
        detection_count = len(detections) if detections else 0
        self.get_logger().debug(f'Published {detection_count} detections')

        # Publish to the detection topic (placeholder implementation)
        # detection_msg = self.create_detection_message(detections, header)
        # self.detection_publisher.publish(detection_msg)

    def publish_status(self):
        """
        Publish system status information.
        """
        # Calculate performance metrics
        avg_detection_time = sum(self.detection_times) / len(self.detection_times) if self.detection_times else 0
        fps = 1.0 / avg_detection_time if avg_detection_time > 0 else 0

        # Create and publish status message (placeholder)
        status_msg = {
            'node_name': self.get_name(),
            'detection_fps': fps,
            'total_frames_processed': self.frame_count,
            'average_detection_time_ms': avg_detection_time * 1000,
            'model_loaded': True,
            'gpu_acceleration': torch.cuda.is_available() if 'torch' in globals() else False
        }

        self.get_logger().debug(f'System status: {status_msg}')

    def process(self):
        """
        Main processing loop (override from base class).
        """
        # This node is event-driven (callback-based), so the main processing
        # happens in the camera_callback method
        pass


def main(args=None):
    """Main function to run the Isaac Sim Object Detector Node."""
    rclpy.init(args=args)

    node = IsaacSimObjectDetectorNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()