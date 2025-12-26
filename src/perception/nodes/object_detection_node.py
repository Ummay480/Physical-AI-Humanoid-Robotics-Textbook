#!/usr/bin/env python3
"""
Object Detection Node for the perception system.

This node subscribes to camera images, performs object detection using YOLO,
and publishes detected objects.
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import numpy as np
import cv2
import json
from typing import List, Optional

from .base_node import BasePerceptionNode
from ..computer_vision.object_detector import YOLODetector
from ..computer_vision.cv_utils import (
    enhance_image, denoise_image, estimate_object_position_3d,
    convert_to_ros_point
)
from ..common.data_types import DetectedObject, ObjectType
from ..common.utils import get_current_timestamp, timestamp_to_ros_time

# Import perception messages
try:
    from perception_msgs.msg import DetectedObject as DetectedObjectMsg
except ImportError:
    print("Warning: perception_msgs not found. Using std_msgs instead.")
    DetectedObjectMsg = None


class ObjectDetectionNode(BasePerceptionNode):
    """
    ROS 2 node for detecting objects in camera images.

    Subscribes to camera topics and publishes detected objects with position estimates.
    """

    def __init__(self):
        """Initialize the object detection node."""
        super().__init__('object_detection_node')

        # Get detection configuration
        detection_config = self.get_detection_config()

        # Initialize YOLO detector
        model_path = detection_config['model_path']
        confidence_threshold = detection_config['confidence_threshold']
        target_classes = detection_config['target_classes']

        self.detector = YOLODetector(
            model_path=model_path,
            confidence_threshold=confidence_threshold,
            target_classes=target_classes
        )

        # Load model
        if not self.detector.load_model():
            self.get_logger().error("Failed to load object detection model")

        # CV Bridge for image conversion
        self.cv_bridge = CvBridge()

        # Camera parameters (should be loaded from calibration)
        self.declare_parameter('camera.fx', 525.0)
        self.declare_parameter('camera.fy', 525.0)
        self.declare_parameter('camera.cx', 319.5)
        self.declare_parameter('camera.cy', 239.5)

        self.camera_params = {
            'fx': self.get_parameter('camera.fx').value,
            'fy': self.get_parameter('camera.fy').value,
            'cx': self.get_parameter('camera.cx').value,
            'cy': self.get_parameter('camera.cy').value
        }

        # Image preprocessing options
        self.declare_parameter('preprocessing.enhance', True)
        self.declare_parameter('preprocessing.denoise', False)

        self.enhance_enabled = self.get_parameter('preprocessing.enhance').value
        self.denoise_enabled = self.get_parameter('preprocessing.denoise').value

        # Visualization
        self.declare_parameter('visualization.enabled', False)
        self.visualization_enabled = self.get_parameter('visualization.enabled').value

        # Subscribe to camera topics
        self.declare_parameter('camera_topics', ['/perception/camera_front/image_processed'])
        camera_topics = self.get_parameter('camera_topics').value

        self.subscriptions = []
        for topic in camera_topics:
            sub = self.create_subscription_with_profile(
                Image,
                topic,
                self._image_callback,
                'real_time_sensor_data'
            )
            self.subscriptions.append(sub)
            self.get_logger().info(f'Subscribed to camera topic: {topic}')

        # Publishers
        if DetectedObjectMsg:
            from perception_msgs.msg import DetectedObject as DOMsg
            # Create array message type
            from std_msgs.msg import Header
            # For now, publish as individual messages
            # TODO: Create DetectedObjectArray message
            self.detection_publisher = self.create_publisher_with_profile(
                String,  # Temporary: use String with JSON
                '/perception/objects_detected',
                'processed_data'
            )
        else:
            self.detection_publisher = self.create_publisher_with_profile(
                String,
                '/perception/objects_detected',
                'processed_data'
            )

        # Visualization publisher (if enabled)
        if self.visualization_enabled:
            self.vis_publisher = self.create_publisher_with_profile(
                Image,
                '/perception/detection_visualization',
                'processed_data'
            )

        # Statistics
        self.stats = {
            'frames_processed': 0,
            'total_detections': 0,
            'avg_processing_time': 0.0
        }

        # Status publisher timer (1 Hz)
        self.status_timer = self.create_timer(1.0, self._publish_status)

        self.get_logger().info('Object Detection Node initialized')

    def _image_callback(self, msg: Image):
        """
        Callback for incoming camera images.

        Args:
            msg: ROS Image message
        """
        try:
            # Convert ROS Image to OpenCV format
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Preprocess image
            processed_image = self._preprocess_image(cv_image)

            # Detect objects
            start_time = get_current_timestamp()
            detections = self.detector.detect(processed_image)
            processing_time = get_current_timestamp() - start_time

            # Update statistics
            self.stats['frames_processed'] += 1
            self.stats['total_detections'] += len(detections)
            self.stats['avg_processing_time'] = (
                (self.stats['avg_processing_time'] * (self.stats['frames_processed'] - 1) +
                 processing_time) / self.stats['frames_processed']
            )

            # Estimate 3D positions for detections
            detections_with_positions = self._estimate_positions(
                detections, cv_image.shape, None
            )

            # Publish detections
            self._publish_detections(detections_with_positions, msg.header)

            # Publish visualization if enabled
            if self.visualization_enabled:
                self._publish_visualization(processed_image, detections_with_positions, msg.header)

            # Log detection info
            if len(detections) > 0:
                self.get_logger().info(
                    f'Detected {len(detections)} objects in {processing_time*1000:.1f}ms'
                )

        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image before detection.

        Args:
            image: Input image

        Returns:
            Preprocessed image
        """
        processed = image.copy()

        # Enhance image if enabled
        if self.enhance_enabled:
            processed = enhance_image(processed, method='clahe')

        # Denoise if enabled
        if self.denoise_enabled:
            processed = denoise_image(processed, method='bilateral')

        return processed

    def _estimate_positions(self, detections: List[DetectedObject],
                          image_shape: tuple,
                          depth_image: Optional[np.ndarray]) -> List[DetectedObject]:
        """
        Estimate 3D positions for detected objects.

        Args:
            detections: List of detected objects
            image_shape: Shape of input image
            depth_image: Depth image (if available)

        Returns:
            Detections with updated 3D positions
        """
        for detection in detections:
            # Estimate 3D position
            position_3d = estimate_object_position_3d(
                detection.bounding_box,
                depth_image,
                image_shape,
                self.camera_params
            )

            # Update position
            detection.position = position_3d

        return detections

    def _publish_detections(self, detections: List[DetectedObject], header):
        """
        Publish detected objects.

        Args:
            detections: List of detected objects
            header: Original image header
        """
        # Convert detections to JSON (temporary solution)
        detections_dict = {
            'header': {
                'timestamp': header.stamp.sec + header.stamp.nanosec / 1e9,
                'frame_id': header.frame_id
            },
            'detections': []
        }

        for det in detections:
            detection_dict = {
                'id': det.id,
                'object_type': det.object_type.value,
                'position': {
                    'x': float(det.position[0]),
                    'y': float(det.position[1]),
                    'z': float(det.position[2])
                },
                'bounding_box': {
                    'x': float(det.bounding_box[0]),
                    'y': float(det.bounding_box[1]),
                    'width': float(det.bounding_box[2]),
                    'height': float(det.bounding_box[3])
                },
                'confidence': float(det.confidence_score),
                'timestamp': det.timestamp
            }
            detections_dict['detections'].append(detection_dict)

        # Publish as JSON string
        msg = String()
        msg.data = json.dumps(detections_dict)
        self.detection_publisher.publish(msg)

    def _publish_visualization(self, image: np.ndarray,
                              detections: List[DetectedObject],
                              header):
        """
        Publish visualization of detections.

        Args:
            image: Input image
            detections: Detected objects
            header: Original image header
        """
        # Visualize detections on image
        vis_image = self.detector.visualize_detections(image, detections)

        # Add text overlay with detection count
        text = f"Detections: {len(detections)}"
        cv2.putText(vis_image, text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Convert to ROS Image message
        vis_msg = self.cv_bridge.cv2_to_imgmsg(vis_image, encoding='bgr8')
        vis_msg.header = header

        self.vis_publisher.publish(vis_msg)

    def _publish_status(self):
        """Publish node status periodically."""
        status = {
            'node': 'object_detection_node',
            'timestamp': get_current_timestamp(),
            'model_loaded': self.detector.is_loaded,
            'confidence_threshold': self.detector.confidence_threshold,
            'target_classes': self.detector.target_classes,
            'stats': self.stats,
            'camera_params': self.camera_params
        }

        # Log statistics periodically
        if self.stats['frames_processed'] % 100 == 0 and self.stats['frames_processed'] > 0:
            self.get_logger().info(
                f"Processed {self.stats['frames_processed']} frames, "
                f"avg time: {self.stats['avg_processing_time']*1000:.1f}ms"
            )

    def shutdown(self):
        """Shutdown the node."""
        self.get_logger().info('Shutting down object detection node')

        # Cancel timers
        if self.status_timer:
            self.status_timer.cancel()


def main(args=None):
    """Main entry point for the object detection node."""
    rclpy.init(args=args)

    node = ObjectDetectionNode()

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
