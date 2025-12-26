"""Isaac Sim training module for AI-Robot Brain"""

from .object_detection_model import (
    DetectionResult,
    SimpleObjectDetector,
    ObjectDetectionTrainer,
    ObjectDetectionInference,
    create_default_object_detection_model,
    train_object_detection_model
)

__all__ = [
    'DetectionResult',
    'SimpleObjectDetector',
    'ObjectDetectionTrainer',
    'ObjectDetectionInference',
    'create_default_object_detection_model',
    'train_object_detection_model'
]