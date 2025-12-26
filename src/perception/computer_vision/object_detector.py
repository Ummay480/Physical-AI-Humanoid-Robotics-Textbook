"""
Object detection model interface for the perception system.

Provides an abstract interface for object detection models and a concrete
implementation using YOLO for humanoid robotics objects.
"""
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import cv2
from pathlib import Path

from ..common.data_types import DetectedObject, ObjectType
from ..common.utils import get_current_timestamp


class ObjectDetectionModel(ABC):
    """
    Abstract base class for object detection models.
    """

    def __init__(self, model_path: str, confidence_threshold: float = 0.5):
        """
        Initialize the object detection model.

        Args:
            model_path: Path to the model file
            confidence_threshold: Minimum confidence threshold for detections
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.is_loaded = False

    @abstractmethod
    def load_model(self) -> bool:
        """
        Load the object detection model.

        Returns:
            True if model loaded successfully, False otherwise
        """
        pass

    @abstractmethod
    def detect(self, image: np.ndarray) -> List[DetectedObject]:
        """
        Detect objects in an image.

        Args:
            image: Input image as numpy array (BGR format)

        Returns:
            List of detected objects
        """
        pass

    @abstractmethod
    def get_supported_classes(self) -> List[str]:
        """
        Get list of object classes supported by this model.

        Returns:
            List of class names
        """
        pass

    def set_confidence_threshold(self, threshold: float):
        """
        Set the confidence threshold for detections.

        Args:
            threshold: New confidence threshold (0.0 - 1.0)
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"Threshold must be between 0.0 and 1.0, got {threshold}")
        self.confidence_threshold = threshold

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model.

        Returns:
            Dictionary containing model information
        """
        return {
            'model_path': self.model_path,
            'confidence_threshold': self.confidence_threshold,
            'is_loaded': self.is_loaded,
            'supported_classes': self.get_supported_classes()
        }


class YOLODetector(ObjectDetectionModel):
    """
    YOLO-based object detector for humanoid robotics objects.

    Supports detection of:
    - Humans
    - Furniture (chairs, tables)
    - Navigation elements (doors, stairs)
    - Obstacles
    """

    # Mapping from YOLO class names to ObjectType enum
    CLASS_MAPPING = {
        'person': ObjectType.HUMAN,
        'human': ObjectType.HUMAN,
        'chair': ObjectType.CHAIR,
        'couch': ObjectType.FURNITURE,
        'sofa': ObjectType.FURNITURE,
        'table': ObjectType.TABLE,
        'dining table': ObjectType.TABLE,
        'door': ObjectType.DOOR,
        'stairs': ObjectType.STAIR,
        'staircase': ObjectType.STAIR,
        'obstacle': ObjectType.OBSTACLE
    }

    def __init__(self, model_path: str, confidence_threshold: float = 0.5,
                 target_classes: Optional[List[str]] = None):
        """
        Initialize the YOLO detector.

        Args:
            model_path: Path to the YOLO model file
            confidence_threshold: Minimum confidence threshold for detections
            target_classes: Specific classes to detect (None for all)
        """
        super().__init__(model_path, confidence_threshold)

        self.target_classes = target_classes or [
            'human', 'chair', 'table', 'door', 'stair', 'obstacle'
        ]

        # Try to import ultralytics YOLO
        try:
            from ultralytics import YOLO
            self.YOLO = YOLO
            self.ultralytics_available = True
        except ImportError:
            self.ultralytics_available = False
            print("Warning: ultralytics not available. Install with: pip install ultralytics")

        # Model input size
        self.input_size = (640, 640)

        # NMS threshold
        self.nms_threshold = 0.45

    def load_model(self) -> bool:
        """
        Load the YOLO model.

        Returns:
            True if model loaded successfully, False otherwise
        """
        if not self.ultralytics_available:
            print("Error: ultralytics package not available")
            return False

        try:
            # Check if model file exists
            if not Path(self.model_path).exists():
                print(f"Warning: Model file not found at {self.model_path}")
                # Use a default YOLO model
                print("Using default YOLOv8n model")
                self.model = self.YOLO('yolov8n.pt')
            else:
                self.model = self.YOLO(self.model_path)

            self.is_loaded = True
            print(f"YOLO model loaded successfully from {self.model_path}")
            return True

        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            self.is_loaded = False
            return False

    def detect(self, image: np.ndarray) -> List[DetectedObject]:
        """
        Detect objects in an image using YOLO.

        Args:
            image: Input image as numpy array (BGR format)

        Returns:
            List of detected objects
        """
        if not self.is_loaded:
            print("Error: Model not loaded")
            return []

        try:
            # Run inference
            results = self.model(image, conf=self.confidence_threshold, verbose=False)

            detected_objects = []

            # Process results
            for result in results:
                boxes = result.boxes

                for i in range(len(boxes)):
                    # Get box data
                    box = boxes[i]
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    xyxy = box.xyxy[0].cpu().numpy()  # Bounding box coordinates

                    # Get class name
                    class_name = self.model.names[class_id]

                    # Filter by target classes
                    if not self._is_target_class(class_name):
                        continue

                    # Map to ObjectType
                    object_type = self._map_class_to_type(class_name)

                    # Create bounding box array
                    x1, y1, x2, y2 = xyxy
                    bbox = np.array([x1, y1, x2 - x1, y2 - y1])  # [x, y, width, height]

                    # Estimate 3D position (placeholder - will be improved with depth)
                    position = self._estimate_position_2d(bbox, image.shape)

                    # Create DetectedObject
                    detected_obj = DetectedObject(
                        object_type=object_type,
                        position=position,
                        bounding_box=bbox,
                        confidence_score=confidence,
                        timestamp=get_current_timestamp(),
                        id=f"obj_{int(get_current_timestamp() * 1e9)}_{i}"
                    )

                    detected_objects.append(detected_obj)

            return detected_objects

        except Exception as e:
            print(f"Error during detection: {e}")
            return []

    def _is_target_class(self, class_name: str) -> bool:
        """
        Check if a class name is in the target classes.

        Args:
            class_name: Name of the class

        Returns:
            True if class is a target, False otherwise
        """
        if not self.target_classes:
            return True

        # Check direct match or if class maps to a target
        for target in self.target_classes:
            if target.lower() in class_name.lower() or class_name.lower() in target.lower():
                return True

        return False

    def _map_class_to_type(self, class_name: str) -> ObjectType:
        """
        Map a YOLO class name to ObjectType enum.

        Args:
            class_name: YOLO class name

        Returns:
            Corresponding ObjectType
        """
        # Check direct mapping
        class_lower = class_name.lower()
        if class_lower in self.CLASS_MAPPING:
            return self.CLASS_MAPPING[class_lower]

        # Check partial matches
        for key, obj_type in self.CLASS_MAPPING.items():
            if key in class_lower or class_lower in key:
                return obj_type

        return ObjectType.UNKNOWN

    def _estimate_position_2d(self, bbox: np.ndarray, image_shape: Tuple) -> np.ndarray:
        """
        Estimate 2D position from bounding box (placeholder for 3D estimation).

        Args:
            bbox: Bounding box [x, y, width, height]
            image_shape: Shape of the image (height, width, channels)

        Returns:
            Estimated position [x, y, z] (z=0 for 2D)
        """
        # Calculate center of bounding box
        center_x = bbox[0] + bbox[2] / 2
        center_y = bbox[1] + bbox[3] / 2

        # Normalize to image center (0, 0)
        image_height, image_width = image_shape[:2]
        norm_x = (center_x - image_width / 2) / image_width
        norm_y = (center_y - image_height / 2) / image_height

        # Return as 3D position (z=0 for now, will be improved with depth)
        return np.array([norm_x, norm_y, 0.0])

    def get_supported_classes(self) -> List[str]:
        """
        Get list of object classes supported by this model.

        Returns:
            List of class names
        """
        if self.is_loaded and self.model:
            return list(self.model.names.values())
        return list(self.CLASS_MAPPING.keys())

    def visualize_detections(self, image: np.ndarray,
                            detections: List[DetectedObject]) -> np.ndarray:
        """
        Visualize detections on an image.

        Args:
            image: Input image
            detections: List of detected objects

        Returns:
            Image with visualized detections
        """
        vis_image = image.copy()

        for det in detections:
            # Get bounding box
            x, y, w, h = det.bounding_box.astype(int)

            # Choose color based on object type
            color = self._get_color_for_type(det.object_type)

            # Draw rectangle
            cv2.rectangle(vis_image, (x, y), (x + w, y + h), color, 2)

            # Draw label
            label = f"{det.object_type.value}: {det.confidence_score:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

            # Draw label background
            cv2.rectangle(vis_image, (x, y - label_size[1] - 10),
                         (x + label_size[0], y), color, -1)

            # Draw label text
            cv2.putText(vis_image, label, (x, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return vis_image

    def _get_color_for_type(self, object_type: ObjectType) -> Tuple[int, int, int]:
        """
        Get visualization color for object type.

        Args:
            object_type: Type of object

        Returns:
            BGR color tuple
        """
        color_map = {
            ObjectType.HUMAN: (0, 255, 0),      # Green
            ObjectType.CHAIR: (255, 0, 0),      # Blue
            ObjectType.TABLE: (0, 0, 255),      # Red
            ObjectType.DOOR: (255, 255, 0),     # Cyan
            ObjectType.STAIR: (255, 0, 255),    # Magenta
            ObjectType.OBSTACLE: (0, 165, 255), # Orange
            ObjectType.FURNITURE: (128, 0, 128) # Purple
        }

        return color_map.get(object_type, (128, 128, 128))  # Gray for unknown

    def get_detection_statistics(self, detections: List[DetectedObject]) -> Dict[str, Any]:
        """
        Get statistics about detections.

        Args:
            detections: List of detected objects

        Returns:
            Dictionary with detection statistics
        """
        stats = {
            'total_detections': len(detections),
            'by_type': {},
            'average_confidence': 0.0,
            'min_confidence': 1.0,
            'max_confidence': 0.0
        }

        if not detections:
            return stats

        # Count by type
        for det in detections:
            obj_type = det.object_type.value
            stats['by_type'][obj_type] = stats['by_type'].get(obj_type, 0) + 1

        # Calculate confidence statistics
        confidences = [det.confidence_score for det in detections]
        stats['average_confidence'] = np.mean(confidences)
        stats['min_confidence'] = np.min(confidences)
        stats['max_confidence'] = np.max(confidences)

        return stats
