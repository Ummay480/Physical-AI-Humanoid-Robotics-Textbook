"""
Synthetic data generation tools for Isaac Sim
Provides utilities for generating synthetic training data with domain randomization
"""
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import json
from pathlib import Path
import random
from datetime import datetime
import cv2  # OpenCV for image processing

from ...common.models.perception_models import (
    SensorData, SensorType, Coordinate, Orientation, SimulationEnvironment
)
from ...common.utils.logger import get_logger


class SyntheticDataGenerator:
    """
    Generator for synthetic training data with domain randomization
    """

    def __init__(self, output_dir: str = "data/synthetic"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger("SyntheticDataGenerator")

        # Define object categories for synthetic data
        self.object_categories = [
            {"id": 0, "name": "cube", "color_range": [(0.6, 1.0), (0.0, 0.4), (0.0, 0.4)]},  # Red variations
            {"id": 1, "name": "sphere", "color_range": [(0.0, 0.4), (0.6, 1.0), (0.0, 0.4)]},  # Green variations
            {"id": 2, "name": "cylinder", "color_range": [(0.0, 0.4), (0.0, 0.4), (0.6, 1.0)]},  # Blue variations
            {"id": 3, "name": "cone", "color_range": [(0.8, 1.0), (0.8, 1.0), (0.0, 0.2)]},  # Yellow variations
            {"id": 4, "name": "capsule", "color_range": [(0.8, 1.0), (0.0, 0.2), (0.8, 1.0)]},  # Magenta variations
        ]

    def generate_random_color(self, color_range: List[Tuple[float, float]]) -> List[float]:
        """
        Generate a random color within the specified range

        Args:
            color_range: List of (min, max) tuples for RGB channels

        Returns:
            RGB color as [R, G, B] list
        """
        return [random.uniform(min_val, max_val) for min_val, max_val in color_range]

    def generate_random_lighting_condition(self) -> Dict[str, Any]:
        """
        Generate random lighting conditions for domain randomization

        Returns:
            Dictionary with lighting parameters
        """
        return {
            "intensity": random.uniform(0.5, 2.0),
            "color_temperature": random.uniform(3000, 8000),  # Kelvin
            "direction": [
                random.uniform(-1, 1),
                random.uniform(-1, 1),
                random.uniform(-1, 0)  # Pointing downward generally
            ]
        }

    def generate_random_material_properties(self) -> Dict[str, Any]:
        """
        Generate random material properties for domain randomization

        Returns:
            Dictionary with material properties
        """
        return {
            "friction": random.uniform(0.1, 0.9),
            "restitution": random.uniform(0.0, 0.5),
            "roughness": random.uniform(0.0, 1.0),
            "metallic": random.uniform(0.0, 1.0)
        }

    def generate_random_object_config(self, object_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a random object configuration

        Args:
            object_type: Specific object type to generate (random if None)

        Returns:
            Dictionary with object configuration
        """
        if object_type is None:
            object_info = random.choice(self.object_categories)
        else:
            object_info = next(
                (cat for cat in self.object_categories if cat["name"] == object_type),
                self.object_categories[0]
            )

        # Generate random position within a reasonable range
        position = [
            random.uniform(-3.0, 3.0),  # x
            random.uniform(-3.0, 3.0),  # y
            random.uniform(0.5, 2.0)    # z (height above ground)
        ]

        # Generate random size/scale
        base_size = random.uniform(0.1, 0.8)
        size = [base_size * random.uniform(0.7, 1.3) for _ in range(3)]  # xyz scale

        # Generate random rotation
        rotation = [
            random.uniform(0, 360),  # x rotation
            random.uniform(0, 360),  # y rotation
            random.uniform(0, 360)   # z rotation
        ]

        # Generate random color based on category
        color = self.generate_random_color(object_info["color_range"])

        return {
            "id": object_info["id"],
            "name": f"{object_info['name']}_{random.randint(1000, 9999)}",
            "type": object_info["name"],
            "position": position,
            "size": size,
            "rotation": rotation,
            "color": color,
            "material": self.generate_random_material_properties(),
            "category": object_info["name"]
        }

    def generate_scene_config(self, num_objects: int = 5) -> Dict[str, Any]:
        """
        Generate a complete scene configuration with multiple objects

        Args:
            num_objects: Number of objects to include in the scene

        Returns:
            Dictionary with scene configuration
        """
        objects = []
        for _ in range(num_objects):
            objects.append(self.generate_random_object_config())

        scene_config = {
            "scene_id": f"scene_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{random.randint(100, 999)}",
            "timestamp": datetime.now().isoformat(),
            "lighting": self.generate_random_lighting_condition(),
            "objects": objects,
            "camera_config": {
                "position": [0.0, 0.0, 1.5],  # Camera position
                "orientation": [0.0, 0.0, 0.0, 1.0],  # Camera orientation (quaternion)
                "fov": random.uniform(60, 120),  # Field of view
                "resolution": [640, 480]  # Image resolution
            },
            "domain_randomization": {
                "lighting_enabled": True,
                "material_enabled": True,
                "object_placement_enabled": True,
                "background_enabled": True
            }
        }

        return scene_config

    def generate_synthetic_image_and_annotations(
        self,
        scene_config: Dict[str, Any],
        image_size: Tuple[int, int] = (640, 480)
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Generate a synthetic image and corresponding annotations

        Args:
            scene_config: Scene configuration dictionary
            image_size: Size of the output image (width, height)

        Returns:
            Tuple of (synthetic image, annotations dictionary)
        """
        width, height = image_size

        # Create a base image with random background
        bg_lighting = scene_config["lighting"]
        bg_brightness = int(bg_lighting["intensity"] * 50)  # Base background brightness

        # Create base image with some random background pattern
        image = np.random.randint(
            bg_brightness, bg_brightness + 30, (height, width, 3), dtype=np.uint8
        )

        # Add some random noise for texture
        noise = np.random.randint(0, 20, (height, width, 3), dtype=np.uint8)
        image = np.clip(image.astype(np.int16) + noise.astype(np.int16), 0, 255).astype(np.uint8)

        # Process each object in the scene
        annotations = {
            "scene_id": scene_config["scene_id"],
            "image_size": image_size,
            "objects": []
        }

        for obj in scene_config["objects"]:
            # Convert object position to 2D image coordinates (simplified)
            # In a real implementation, this would involve proper 3D to 2D projection
            obj_x, obj_y, obj_z = obj["position"]

            # Calculate 2D position (simplified perspective projection)
            scale_factor = 300 / (obj_z + 1)  # Simple perspective scaling
            center_x = int(width / 2 + obj_x * scale_factor)
            center_y = int(height / 2 - obj_y * scale_factor)

            # Calculate size based on distance and 3D size
            avg_size = sum(obj["size"]) / 3
            obj_size = int(avg_size * scale_factor)

            # Make sure the object is within image bounds
            center_x = max(obj_size, min(width - obj_size, center_x))
            center_y = max(obj_size, min(height - obj_size, center_y))

            # Convert color from [0,1] to [0,255]
            color = [int(c * 255) for c in obj["color"]]
            color = tuple(color)

            # Draw the object based on its type
            if obj["type"] == "sphere":
                cv2.circle(image, (center_x, center_y), obj_size, color, -1)
            elif obj["type"] == "cube":
                # Draw a square for cube (simplified)
                pt1 = (center_x - obj_size, center_y - obj_size)
                pt2 = (center_x + obj_size, center_y + obj_size)
                cv2.rectangle(image, pt1, pt2, color, -1)
            elif obj["type"] == "cylinder":
                # Draw an ellipse for cylinder (top view)
                cv2.ellipse(image, (center_x, center_y), (obj_size, obj_size//2), 0, 0, 360, color, -1)
            elif obj["type"] == "cone":
                # Draw a triangle for cone
                points = np.array([
                    [center_x, center_y - obj_size],
                    [center_x - obj_size, center_y + obj_size],
                    [center_x + obj_size, center_y + obj_size]
                ], np.int32)
                cv2.fillPoly(image, [points], color)
            elif obj["type"] == "capsule":
                # Draw a capsule shape (rectangle with rounded ends)
                pt1 = (center_x - obj_size//2, center_y - obj_size//4)
                pt2 = (center_x + obj_size//2, center_y + obj_size//4)
                cv2.rectangle(image, pt1, pt2, color, -1)
                cv2.circle(image, (center_x - obj_size//2, center_y), obj_size//4, color, -1)
                cv2.circle(image, (center_x + obj_size//2, center_y), obj_size//4, color, -1)

            # Add annotation for this object
            bbox = [
                max(0, center_x - obj_size),  # x
                max(0, center_y - obj_size),  # y
                min(width, 2 * obj_size),     # width
                min(height, 2 * obj_size)     # height
            ]

            annotations["objects"].append({
                "name": obj["name"],
                "category": obj["category"],
                "class_id": obj["id"],
                "bbox": {
                    "x": bbox[0],
                    "y": bbox[1],
                    "width": bbox[2],
                    "height": bbox[3]
                },
                "position_3d": obj["position"],
                "size_3d": obj["size"],
                "color": obj["color"],
                "material": obj["material"]
            })

        return image, annotations

    def generate_batch(
        self,
        num_samples: int,
        output_dir: Optional[str] = None,
        image_size: Tuple[int, int] = (640, 480)
    ) -> List[str]:
        """
        Generate a batch of synthetic data samples

        Args:
            num_samples: Number of samples to generate
            output_dir: Directory to save the data (uses default if None)
            image_size: Size of generated images

        Returns:
            List of paths to generated annotation files
        """
        if output_dir is None:
            output_dir = self.output_dir
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        annotation_paths = []

        for i in range(num_samples):
            # Generate scene configuration
            scene_config = self.generate_scene_config(num_objects=random.randint(3, 7))

            # Generate synthetic image and annotations
            image, annotations = self.generate_synthetic_image_and_annotations(
                scene_config, image_size
            )

            # Save the image
            image_filename = f"synthetic_image_{i:05d}.png"
            image_path = output_dir / image_filename
            cv2.imwrite(str(image_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

            # Save the annotations
            annotation_filename = f"annotations_{i:05d}.json"
            annotation_path = output_dir / annotation_filename
            annotation_paths.append(str(annotation_path))

            with open(annotation_path, 'w') as f:
                json.dump(annotations, f, indent=2)

            # Also save the full scene config for reference
            config_filename = f"scene_config_{i:05d}.json"
            config_path = output_dir / config_filename
            with open(config_path, 'w') as f:
                json.dump(scene_config, f, indent=2)

            if i % 100 == 0:
                self.logger.info(f"Generated {i+1}/{num_samples} synthetic samples")

        self.logger.info(f"Synthetic data generation completed. {num_samples} samples saved to {output_dir}")
        return annotation_paths

    def generate_sensor_data_from_config(
        self,
        scene_config: Dict[str, Any],
        sensor_type: SensorType = SensorType.RGB_CAMERA
    ) -> SensorData:
        """
        Generate SensorData entity from scene configuration

        Args:
            scene_config: Scene configuration dictionary
            sensor_type: Type of sensor data to generate

        Returns:
            SensorData entity
        """
        # Generate synthetic sensor data payload based on sensor type
        if sensor_type == SensorType.RGB_CAMERA:
            # For RGB camera, we'd generate an image
            # Here we'll create a dummy payload with some basic structure
            image, _ = self.generate_synthetic_image_and_annotations(
                scene_config, (640, 480)
            )
            # Convert image to bytes (in a real implementation, this would be proper encoding)
            payload = image.tobytes()
        elif sensor_type == SensorType.DEPTH_CAMERA:
            # For depth camera, we'd generate a depth map
            depth_map = np.random.rand(480, 640).astype(np.float32)
            payload = depth_map.tobytes()
        else:
            # For other sensor types, create dummy payload
            payload = b"dummy_sensor_payload"

        # Extract camera position from scene config
        cam_pos = scene_config["camera_config"]["position"]
        cam_orient = scene_config["camera_config"]["orientation"]

        return SensorData(
            id=None,  # Auto-generated
            sensor_type=sensor_type,
            timestamp=datetime.now(),
            data_payload=payload,
            frame_id="camera_frame",
            sensor_position=Coordinate(*cam_pos),
            sensor_orientation=Orientation(*cam_orient)
        )


class DomainRandomizationEngine:
    """
    Engine for applying domain randomization techniques to synthetic data
    """

    def __init__(self):
        self.logger = get_logger("DomainRandomizationEngine")

    def apply_texture_randomization(self, image: np.ndarray) -> np.ndarray:
        """
        Apply random texture variations to an image

        Args:
            image: Input image

        Returns:
            Image with randomized textures
        """
        # Add random noise
        noise = np.random.normal(0, random.uniform(1, 5), image.shape).astype(np.int16)
        randomized = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # Apply random blur
        blur_kernel = random.choice([3, 5, 7])
        if random.random() > 0.7:  # 30% chance of blur
            randomized = cv2.GaussianBlur(randomized, (blur_kernel, blur_kernel), 0)

        return randomized

    def apply_lighting_randomization(self, image: np.ndarray) -> np.ndarray:
        """
        Apply random lighting variations to an image

        Args:
            image: Input image

        Returns:
            Image with randomized lighting
        """
        # Convert to HSV for lighting adjustment
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float64)

        # Randomly adjust value (brightness)
        brightness_factor = random.uniform(0.7, 1.3)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * brightness_factor, 0, 255)

        # Randomly adjust saturation
        saturation_factor = random.uniform(0.8, 1.2)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_factor, 0, 255)

        # Convert back to RGB
        randomized = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

        return randomized

    def apply_color_randomization(self, image: np.ndarray) -> np.ndarray:
        """
        Apply random color variations to an image

        Args:
            image: Input image

        Returns:
            Image with randomized colors
        """
        # Apply random color channel multipliers
        multipliers = [random.uniform(0.8, 1.2) for _ in range(3)]  # RGB multipliers

        randomized = image.astype(np.float64)
        for i in range(3):  # Apply to each channel
            randomized[:, :, i] = np.clip(randomized[:, :, i] * multipliers[i], 0, 255)

        return randomized.astype(np.uint8)


def get_default_synthetic_data_generator() -> SyntheticDataGenerator:
    """
    Get a default synthetic data generator instance

    Returns:
        SyntheticDataGenerator instance
    """
    return SyntheticDataGenerator()


def generate_synthetic_dataset(
    num_samples: int,
    output_dir: str = "data/synthetic",
    image_size: Tuple[int, int] = (640, 480)
) -> List[str]:
    """
    Generate a complete synthetic dataset

    Args:
        num_samples: Number of samples to generate
        output_dir: Directory to save the dataset
        image_size: Size of generated images

    Returns:
        List of paths to generated annotation files
    """
    generator = get_default_synthetic_data_generator()
    return generator.generate_batch(num_samples, output_dir, image_size)