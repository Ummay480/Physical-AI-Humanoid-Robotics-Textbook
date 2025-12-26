"""
Main Perception ROS 2 Node
Implements T018: Create Perception ROS 2 Node
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.parameter import Parameter
import numpy as np
import cv2
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Header
from cv_bridge import CvBridge
from ...common.base_node import BaseAIRobotNode
from ...common.models.perception_models import DetectionResult, SensorType
from ..perception import ObjectDetectionSystem
import torch
from typing import List, Dict, Any
import time


class IsaacPerceptionNode(BaseAIRobotNode):
    """
    Main ROS 2 node for Isaac Sim perception system.
    Extends BaseAIRobotNode and integrates all perception components.
    """

    def __init__(self):
        super().__init__('isaac_perception')

        # Initialize CV bridge for image conversion
        self.bridge = CvBridge()

        # Initialize object detection system
        self.detection_system = ObjectDetectionSystem(confidence_threshold=0.5)

        # Get parameters from configuration
        self.declare_parameter('confidence_threshold', 0.5)
        self.declare_parameter('enable_tracking', False)
        self.declare_parameter('enable_depth_processing', False)
        self.declare_parameter('publish_visualization', True)

        self.confidence_threshold = self.get_parameter('confidence_threshold').value
        self.enable_tracking = self.get_parameter('enable_tracking').value
        self.enable_depth_processing = self.get_parameter('enable_depth_processing').value
        self.publish_visualization = self.get_parameter('publish_visualization').value

        # Update detection system with parameter
        self.detection_system.confidence_threshold = self.confidence_threshold

        # Create subscription to Isaac Sim camera topics
        self.camera_subscription = self.create_subscription(
            Image,
            '/isaac_sim/camera/rgb/image_raw',
            self.camera_callback,
            qos_profile=QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                history=HistoryPolicy.KEEP_LAST,
                depth=1
            )
        )

        # Optional: Create subscription to depth camera if enabled
        if self.enable_depth_processing:
            self.depth_subscription = self.create_subscription(
                Image,
                '/isaac_sim/camera/depth/image_raw',
                self.depth_callback,
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

        # Optional: Create publisher for visualization
        if self.publish_visualization:
            self.visualization_publisher = self.create_publisher(
                Image,
                '/isaac_perception/visualization',
                qos_profile=QoSProfile(
                    reliability=ReliabilityPolicy.RELIABLE,
                    history=HistoryPolicy.KEEP_LAST,
                    depth=1
                )
            )

        # Timer for performance monitoring and status publishing
        self.status_timer = self.create_timer(1.0, self.publish_status)

        # Performance metrics
        self.frame_count = 0
        self.detection_times = []
        self.detection_fps = 0

        # Object tracking (for optional tracking across frames)
        if self.enable_tracking:
            self.tracked_objects = {}
            self.next_object_id = 0

        # Store depth image for 3D position estimation
        self.latest_depth_image = None
        self.depth_timestamp = None

        self.get_logger().info('Isaac Perception Node initialized')
        self.get_logger().info(f'Confidence threshold: {self.confidence_threshold}')
        self.get_logger().info(f'Object tracking: {self.enable_tracking}')
        self.get_logger().info(f'Depth processing: {self.enable_depth_processing}')
        self.get_logger().info(f'Visualization: {self.publish_visualization}')

    def camera_callback(self, msg: Image):
        """
        Callback function for processing RGB camera images from Isaac Sim.

        Args:
            msg: Image message from Isaac Sim camera
        """
        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Start timing for performance measurement
            start_time = time.time()

            # Run object detection on the image
            if self.enable_depth_processing and self.latest_depth_image is not None:
                # Use depth information for 3D position estimation
                detections = self.detection_system.detect_objects_with_depth(
                    cv_image, self.latest_depth_image
                )
            else:
                # Standard 2D detection
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
            filtered_detections = [d for d in detections if d.confidence >= self.confidence_threshold]

            # Apply non-maximum suppression to reduce overlapping detections
            nms_detections = self._apply_non_maximum_suppression(filtered_detections)

            # Apply object tracking if enabled
            if self.enable_tracking:
                tracked_detections = self._apply_object_tracking(nms_detections)
            else:
                tracked_detections = nms_detections

            # Update frame count
            self.frame_count += 1

            # Log performance if needed
            if self.frame_count % 30 == 0:  # Log every 30 frames
                self.get_logger().info(
                    f'Detection performance: {self.detection_fps:.2f} FPS, '
                    f'avg time: {np.mean(self.detection_times)*1000:.2f} ms, '
                    f'detected: {len(tracked_detections)} objects'
                )

            # Publish detection results
            self.publish_detections(tracked_detections, msg.header)

            # Publish visualization if enabled
            if self.publish_visualization:
                self.publish_visualization_image(cv_image, tracked_detections, msg.header)

        except Exception as e:
            self.get_logger().error(f'Error processing camera image: {str(e)}')

    def depth_callback(self, msg: Image):
        """
        Callback function for processing depth camera images.

        Args:
            msg: Image message from Isaac Sim depth camera
        """
        try:
            # Convert ROS Image message to OpenCV image (depth format)
            depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
            self.latest_depth_image = depth_image
            self.depth_timestamp = msg.header.stamp

        except Exception as e:
            self.get_logger().error(f'Error processing depth image: {str(e)}')

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

    def _calculate_iou(self, bbox1: tuple, bbox2: tuple) -> float:
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

    def _apply_object_tracking(self, detections) -> List:
        """
        Apply object tracking to associate detections across frames.

        Args:
            detections: List of current detections

        Returns:
            List of detections with tracking IDs
        """
        # For simplicity, we'll implement a basic tracking algorithm
        # In a real implementation, this would use more sophisticated tracking
        tracked_detections = []

        for detection in detections:
            # Find the closest existing tracked object
            best_match_id = None
            best_distance = float('inf')

            for obj_id, tracked_obj in self.tracked_objects.items():
                # Calculate distance between detection and tracked object
                det_center_x = (detection.bbox[0] + detection.bbox[2]) / 2
                det_center_y = (detection.bbox[1] + detection.bbox[3]) / 2
                track_center_x = (tracked_obj['bbox'][0] + tracked_obj['bbox'][2]) / 2
                track_center_y = (tracked_obj['bbox'][1] + tracked_obj['bbox'][3]) / 2

                distance = np.sqrt((det_center_x - track_center_x)**2 + (det_center_y - track_center_y)**2)

                if distance < best_distance and distance < 50:  # Threshold for matching
                    best_distance = distance
                    best_match_id = obj_id

            if best_match_id is not None:
                # Update existing tracked object
                self.tracked_objects[best_match_id]['bbox'] = detection.bbox
                self.tracked_objects[best_match_id]['last_seen'] = self.get_clock().now()
                tracked_detections.append({
                    'detection': detection,
                    'track_id': best_match_id
                })
            else:
                # Create new tracked object
                new_id = self.next_object_id
                self.tracked_objects[new_id] = {
                    'bbox': detection.bbox,
                    'class_name': detection.class_name,
                    'last_seen': self.get_clock().now()
                }
                self.next_object_id += 1
                tracked_detections.append({
                    'detection': detection,
                    'track_id': new_id
                })

        # Remove old tracked objects (simple cleanup)
        current_time = self.get_clock().now()
        for obj_id in list(self.tracked_objects.keys()):
            if (current_time - self.tracked_objects[obj_id]['last_seen']).nanoseconds > 1e9:  # 1 second
                del self.tracked_objects[obj_id]

        return tracked_detections

    def publish_detections(self, detections, header: Header):
        """
        Publish detection results to ROS topic.

        Args:
            detections: List of detected objects with bounding boxes
            header: Header from original image message
        """
        # Convert detections to ROS message format
        detection_count = len(detections)
        if detection_count > 0:
            self.get_logger().debug(f'Publishing {detection_count} detections')

            # Print detection details for debugging
            for i, det in enumerate(detections):
                if isinstance(det, dict):  # If it's a tracked detection
                    detection = det['detection']
                    track_id = det['track_id']
                    self.get_logger().debug(
                        f'Detection {i+1}: {detection.class_name} '
                        f'(conf: {detection.confidence:.2f}, track_id: {track_id}) at {detection.bbox}'
                    )
                else:  # If it's a regular detection
                    self.get_logger().debug(
                        f'Detection {i+1}: {det.class_name} '
                        f'(conf: {det.confidence:.2f}) at {det.bbox}'
                    )

        # Publish to the detection topic (placeholder implementation)
        # detection_msg = self.create_detection_message(detections, header)
        # self.detection_publisher.publish(detection_msg)

    def publish_visualization_image(self, image, detections, header: Header):
        """
        Publish visualization image with bounding boxes drawn.

        Args:
            image: Original image
            detections: List of detections
            header: Header from original image message
        """
        try:
            # Create a copy of the image for visualization
            vis_image = image.copy()

            # Draw bounding boxes and labels
            for i, det in enumerate(detections):
                if isinstance(det, dict):  # If it's a tracked detection
                    detection = det['detection']
                    track_id = det['track_id']
                    label = f"{detection.class_name} ({detection.confidence:.2f}) ID:{track_id}"
                else:  # If it's a regular detection
                    detection = det
                    label = f"{detection.class_name} ({detection.confidence:.2f})"

                # Get bounding box coordinates
                x_min, y_min, x_max, y_max = detection.bbox

                # Draw bounding box
                cv2.rectangle(vis_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                # Draw label background
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                cv2.rectangle(vis_image, (x_min, y_min - label_size[1] - 10),
                             (x_min + label_size[0], y_min), (0, 255, 0), -1)

                # Draw label text
                cv2.putText(vis_image, label, (x_min, y_min - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

            # Convert back to ROS Image message
            ros_image = self.bridge.cv2_to_imgmsg(vis_image, encoding='bgr8')
            ros_image.header = header

            # Publish visualization
            self.visualization_publisher.publish(ros_image)

        except Exception as e:
            self.get_logger().error(f'Error publishing visualization: {str(e)}')

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
            'active_detections': len(self.tracked_objects) if self.enable_tracking else 0,
            'model_loaded': self.detection_system.model_loaded,
            'gpu_acceleration': torch.cuda.is_available() if 'torch' in globals() else False,
            'confidence_threshold': self.confidence_threshold,
            'parameters': {
                'enable_tracking': self.enable_tracking,
                'enable_depth_processing': self.enable_depth_processing,
                'publish_visualization': self.publish_visualization
            }
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
    """Main function to run the Isaac Perception Node."""
    rclpy.init(args=args)

    node = IsaacPerceptionNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()