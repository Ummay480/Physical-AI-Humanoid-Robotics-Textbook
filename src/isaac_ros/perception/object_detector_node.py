"""
Object Detection and Classification System for Isaac ROS
Implements T016: Create Object Detection and Classification System
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import numpy as np
import cv2
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Header
from cv_bridge import CvBridge
from ...common.base_node import BaseAIRobotNode
from ...common.models.perception_models import DetectionResult, SensorType
from .detection_system import ObjectDetectionSystem
import torch
from typing import List, Tuple
import time


class IsaacObjectDetectionNode(BaseAIRobotNode):
    """
    ROS 2 node for Isaac ROS object detection and classification system.
    Implements complete object detection pipeline with classification.
    """

    def __init__(self):
        super().__init__('isaac_object_detection')

        # Initialize CV bridge for image conversion
        self.bridge = CvBridge()

        # Initialize object detection system
        self.detection_system = ObjectDetectionSystem(confidence_threshold=0.5)

        # Create subscription to camera topics
        self.subscription = self.create_subscription(
            Image,
            '/camera/rgb/image_raw',  # Generic camera topic
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
        self.detection_fps = 0

        # Object tracking (for optional tracking across frames)
        self.tracked_objects = {}

        self.get_logger().info('Isaac Object Detection Node initialized')

    def camera_callback(self, msg: Image):
        """
        Callback function for processing camera images.

        Args:
            msg: Image message from camera
        """
        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Start timing for performance measurement
            start_time = time.time()

            # Run object detection on the image
            detections = self.detection_system.detect_objects(cv_image)

            # Calculate detection time
            detection_time = time.time() - start_time
            self.detection_times.append(detection_time)

            # Keep only the last 100 timing measurements
            if len(self.detection_times) > 100:
                self.detection_times = self.detection_times[-100:]

            # Calculate FPS
            if len(self.detection_times) > 0:
                avg_time = sum(self.detection_times) / len(self.detection_times)
                self.detection_fps = 1.0 / avg_time if avg_time > 0 else 0

            # Filter detections by confidence threshold
            filtered_detections = [d for d in detections if d.confidence >= self.detection_system.confidence_threshold]

            # Apply non-maximum suppression to reduce overlapping detections
            nms_detections = self._apply_non_maximum_suppression(filtered_detections)

            # Update frame count
            self.frame_count += 1

            # Log performance if needed
            if self.frame_count % 30 == 0:  # Log every 30 frames
                self.get_logger().info(
                    f'Detection performance: {self.detection_fps:.2f} FPS, '
                    f'avg time: {np.mean(self.detection_times)*1000:.2f} ms, '
                    f'detected: {len(nms_detections)} objects'
                )

            # Publish detection results
            self.publish_detections(nms_detections, msg.header)

        except Exception as e:
            self.get_logger().error(f'Error processing camera image: {str(e)}')

    def _apply_non_maximum_suppression(self, detections, iou_threshold: float = 0.5) -> List:
        """
        Apply Non-Maximum Suppression to reduce overlapping detections.

        Args:
            detections: List of detections
            iou_threshold: Intersection over Union threshold

        Returns:
            List of detections after NMS
        """
        if len(detections) == 0:
            return []

        # Sort detections by confidence in descending order
        sorted_detections = sorted(detections, key=lambda x: x.confidence, reverse=True)

        # Apply NMS
        selected_detections = []
        for detection in sorted_detections:
            overlap = False
            for selected in selected_detections:
                # Calculate IoU between bounding boxes
                iou = self._calculate_iou(detection.bbox, selected.bbox)
                if iou > iou_threshold:
                    overlap = True
                    break

            if not overlap:
                selected_detections.append(detection)

        return selected_detections

    def _calculate_iou(self, bbox1: Tuple, bbox2: Tuple) -> float:
        """
        Calculate Intersection over Union between two bounding boxes.

        Args:
            bbox1: First bounding box (x_min, y_min, x_max, y_max)
            bbox2: Second bounding box (x_min, y_min, x_max, y_max)

        Returns:
            IoU value
        """
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2

        # Calculate intersection area
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)

        if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
            return 0.0

        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)

        # Calculate union area
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = area1 + area2 - inter_area

        return inter_area / union_area if union_area > 0 else 0.0

    def publish_detections(self, detections, header: Header):
        """
        Publish detection results to ROS topic.

        Args:
            detections: List of detected objects with bounding boxes
            header: Header from original image message
        """
        # Convert detections to ROS message format
        # In a real implementation, this would create and publish a proper ROS message
        detection_count = len(detections)
        if detection_count > 0:
            self.get_logger().info(f'Publishing {detection_count} detections')

            # Print detection details for debugging
            for i, detection in enumerate(detections):
                self.get_logger().debug(
                    f'Detection {i+1}: {detection.class_name} '
                    f'(conf: {detection.confidence:.2f}) at {detection.bbox}'
                )

        # Publish to the detection topic (placeholder implementation)
        # detection_msg = self.create_detection_message(detections, header)
        # self.detection_publisher.publish(detection_msg)

    def publish_status(self):
        """
        Publish system status information.
        """
        # Calculate performance metrics
        avg_detection_time = sum(self.detection_times) / len(self.detection_times) if self.detection_times else 0

        # Create and publish status message (placeholder)
        status_msg = {
            'node_name': self.get_name(),
            'detection_fps': self.detection_fps,
            'total_frames_processed': self.frame_count,
            'average_detection_time_ms': avg_detection_time * 1000,
            'active_detections': len(self.tracked_objects) if self.tracked_objects else 0,
            'model_loaded': self.detection_system.model_loaded,
            'gpu_acceleration': torch.cuda.is_available() if 'torch' in globals() else False,
            'confidence_threshold': self.detection_system.confidence_threshold
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
    """Main function to run the Isaac Object Detection Node."""
    rclpy.init(args=args)

    node = IsaacObjectDetectionNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()