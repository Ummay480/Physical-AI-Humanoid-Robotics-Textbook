"""
Synthetic Data Generation Pipeline for Perception Training
Implements T014: Implement Synthetic Data Generation Pipeline
"""
import os
import json
import numpy as np
from typing import List, Dict, Any, Tuple
from pathlib import Path
from dataclasses import dataclass
from PIL import Image
import cv2


@dataclass
class BoundingBox:
    """Bounding box for object detection."""
    x_min: int
    y_min: int
    x_max: int
    y_max: int
    class_id: int
    confidence: float = 1.0


@dataclass
class SyntheticDataSample:
    """Represents a single synthetic data sample."""
    rgb_image: np.ndarray
    depth_image: np.ndarray
    segmentation_mask: np.ndarray
    bounding_boxes: List[BoundingBox]
    image_id: str
    annotations: Dict[str, Any]


class SyntheticDataGenerator:
    """Generates labeled synthetic data for perception model training."""

    def __init__(self, output_dir: str = "data/synthetic_perception_dataset"):
        """
        Initialize the synthetic data generator.

        Args:
            output_dir: Directory to save generated data
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories for different data types
        self.rgb_dir = self.output_dir / "rgb"
        self.depth_dir = self.output_dir / "depth"
        self.segmentation_dir = self.output_dir / "segmentation"
        self.annotations_dir = self.output_dir / "annotations"

        for directory in [self.rgb_dir, self.depth_dir, self.segmentation_dir, self.annotations_dir]:
            directory.mkdir(exist_ok=True)

        # Define object classes for detection
        self.object_classes = {
            0: "background",
            1: "human",
            2: "chair",
            3: "table",
            4: "door",
            5: "stair",
            6: "obstacle",
            7: "robot",
            8: "box",
            9: "plant"
        }

        self.class_count = len(self.object_classes)

    def generate_camera_data(self, width: int = 640, height: int = 480) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate synthetic camera data (RGB, depth, segmentation).

        Args:
            width: Image width
            height: Image height

        Returns:
            Tuple of (RGB image, depth image, segmentation mask)
        """
        # Generate synthetic RGB image
        rgb_image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)

        # Add some synthetic objects with different colors
        for _ in range(np.random.randint(3, 8)):  # 3-8 objects
            # Random position and size
            center_x = np.random.randint(50, width - 50)
            center_y = np.random.randint(50, height - 50)
            radius = np.random.randint(20, 60)

            # Random color
            color = np.random.randint(0, 255, 3, dtype=np.uint8)

            # Draw filled circle
            cv2.circle(rgb_image, (center_x, center_y), radius, color.tolist(), -1)

        # Generate synthetic depth image
        depth_image = np.random.uniform(0.1, 10.0, (height, width)).astype(np.float32)

        # Add some depth variation to match RGB objects
        for _ in range(np.random.randint(3, 8)):
            center_x = np.random.randint(50, width - 50)
            center_y = np.random.randint(50, height - 50)
            radius = np.random.randint(20, 60)
            depth_value = np.random.uniform(0.5, 8.0)

            # Create a circular depth region
            y, x = np.ogrid[:height, :width]
            mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
            depth_image[mask] = depth_value

        # Generate synthetic segmentation mask
        segmentation_mask = np.zeros((height, width), dtype=np.uint8)

        # Add labeled regions for different objects
        for class_id in range(1, self.class_count):  # Skip background
            if np.random.random() > 0.3:  # 70% chance to include each class
                center_x = np.random.randint(50, width - 50)
                center_y = np.random.randint(50, height - 50)
                radius = np.random.randint(15, 50)

                y, x = np.ogrid[:height, :width]
                mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
                segmentation_mask[mask] = class_id

        return rgb_image, depth_image, segmentation_mask

    def generate_bounding_boxes(self, segmentation_mask: np.ndarray) -> List[BoundingBox]:
        """
        Generate bounding boxes from segmentation mask.

        Args:
            segmentation_mask: Segmentation mask with class labels

        Returns:
            List of bounding boxes
        """
        bounding_boxes = []

        for class_id in range(1, self.class_count):  # Skip background
            # Find contours for this class
            class_mask = (segmentation_mask == class_id).astype(np.uint8)

            if np.any(class_mask):
                # Find contours
                contours, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for contour in contours:
                    if len(contour) >= 3:  # Valid contour
                        x, y, w, h = cv2.boundingRect(contour)
                        bbox = BoundingBox(
                            x_min=x,
                            y_min=y,
                            x_max=x + w,
                            y_max=y + h,
                            class_id=class_id,
                            confidence=1.0
                        )
                        bounding_boxes.append(bbox)

        return bounding_boxes

    def apply_data_augmentation(self, rgb_image: np.ndarray, depth_image: np.ndarray,
                               segmentation_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply data augmentation to synthetic data.

        Args:
            rgb_image: RGB image
            depth_image: Depth image
            segmentation_mask: Segmentation mask

        Returns:
            Augmented images
        """
        # Random rotation
        if np.random.random() > 0.5:
            angle = np.random.uniform(-10, 10)  # -10 to 10 degrees
            h, w = rgb_image.shape[:2]
            center = (w // 2, h // 2)

            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            rgb_image = cv2.warpAffine(rgb_image, rotation_matrix, (w, h))
            depth_image = cv2.warpAffine(depth_image, rotation_matrix, (w, h),
                                        flags=cv2.INTER_NEAREST)  # Use nearest for segmentation
            segmentation_mask = cv2.warpAffine(segmentation_mask, rotation_matrix, (w, h),
                                             flags=cv2.INTER_NEAREST)

        # Random brightness adjustment
        if np.random.random() > 0.5:
            brightness_factor = np.random.uniform(0.8, 1.2)
            rgb_image = np.clip(rgb_image * brightness_factor, 0, 255).astype(np.uint8)

        # Random noise
        if np.random.random() > 0.5:
            noise = np.random.normal(0, 5, rgb_image.shape).astype(np.int16)
            rgb_image = np.clip(rgb_image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        return rgb_image, depth_image, segmentation_mask

    def export_to_coco_format(self, samples: List[SyntheticDataSample], output_path: str):
        """
        Export synthetic data samples to COCO format.

        Args:
            samples: List of synthetic data samples
            output_path: Path to save COCO format JSON
        """
        coco_data = {
            "info": {
                "description": "Synthetic Perception Dataset",
                "version": "1.0",
                "year": 2025,
                "contributor": "AI-Robot Brain Team",
                "date_created": "2025-01-01"
            },
            "licenses": [
                {
                    "id": 1,
                    "name": "MIT License",
                    "url": "https://opensource.org/licenses/MIT"
                }
            ],
            "categories": [
                {
                    "id": class_id,
                    "name": class_name,
                    "supercategory": "object"
                }
                for class_id, class_name in self.object_classes.items()
                if class_id > 0  # Skip background
            ],
            "images": [],
            "annotations": []
        }

        annotation_id = 1
        for i, sample in enumerate(samples):
            # Add image info
            image_info = {
                "id": i + 1,
                "file_name": f"rgb/{sample.image_id}.jpg",
                "width": sample.rgb_image.shape[1],
                "height": sample.rgb_image.shape[0],
                "date_captured": "2025-01-01",
                "license": 1,
                "coco_url": "",
                "flickr_url": ""
            }
            coco_data["images"].append(image_info)

            # Add annotations for bounding boxes
            for bbox in sample.bounding_boxes:
                annotation = {
                    "id": annotation_id,
                    "image_id": i + 1,
                    "category_id": bbox.class_id,
                    "bbox": [bbox.x_min, bbox.y_min,
                            bbox.x_max - bbox.x_min, bbox.y_max - bbox.y_min],
                    "area": (bbox.x_max - bbox.x_min) * (bbox.y_max - bbox.y_min),
                    "iscrowd": 0
                }
                coco_data["annotations"].append(annotation)
                annotation_id += 1

        # Save COCO format JSON
        with open(output_path, 'w') as f:
            json.dump(coco_data, f, indent=2)

    def generate_dataset(self, num_samples: int = 10000, apply_augmentation: bool = True) -> List[SyntheticDataSample]:
        """
        Generate a complete synthetic dataset.

        Args:
            num_samples: Number of samples to generate
            apply_augmentation: Whether to apply data augmentation

        Returns:
            List of synthetic data samples
        """
        print(f"Generating {num_samples} synthetic data samples...")

        samples = []
        for i in range(num_samples):
            if i % 1000 == 0:
                print(f"Generated {i}/{num_samples} samples...")

            # Generate raw camera data
            rgb_image, depth_image, segmentation_mask = self.generate_camera_data()

            # Apply data augmentation if enabled
            if apply_augmentation:
                rgb_image, depth_image, segmentation_mask = self.apply_data_augmentation(
                    rgb_image, depth_image, segmentation_mask
                )

            # Generate bounding boxes
            bounding_boxes = self.generate_bounding_boxes(segmentation_mask)

            # Create image ID
            image_id = f"synthetic_{i:06d}"

            # Create synthetic data sample
            sample = SyntheticDataSample(
                rgb_image=rgb_image,
                depth_image=depth_image,
                segmentation_mask=segmentation_mask,
                bounding_boxes=bounding_boxes,
                image_id=image_id,
                annotations={}
            )

            samples.append(sample)

            # Save the data
            self._save_sample(sample)

        print(f"Dataset generation complete. Generated {len(samples)} samples.")

        # Export to COCO format
        coco_path = self.output_dir / "annotations" / "instances_synthetic.json"
        self.export_to_coco_format(samples, str(coco_path))

        return samples

    def _save_sample(self, sample: SyntheticDataSample):
        """Save a synthetic data sample to disk."""
        # Save RGB image
        rgb_path = self.rgb_dir / f"{sample.image_id}.jpg"
        Image.fromarray(sample.rgb_image).save(rgb_path)

        # Save depth image (as 16-bit PNG)
        depth_path = self.depth_dir / f"{sample.image_id}.png"
        depth_normalized = ((sample.depth_image - sample.depth_image.min()) /
                           (sample.depth_image.max() - sample.depth_image.min()) * 65535).astype(np.uint16)
        Image.fromarray(depth_normalized).save(depth_path)

        # Save segmentation mask
        segmentation_path = self.segmentation_dir / f"{sample.image_id}.png"
        Image.fromarray(sample.segmentation_mask).save(segmentation_path)

        # Save annotations
        annotations = {
            "image_id": sample.image_id,
            "width": sample.rgb_image.shape[1],
            "height": sample.rgb_image.shape[0],
            "bounding_boxes": [
                {
                    "x_min": bbox.x_min,
                    "y_min": bbox.y_min,
                    "x_max": bbox.x_max,
                    "y_max": bbox.y_max,
                    "class_id": bbox.class_id,
                    "class_name": self.object_classes[bbox.class_id],
                    "confidence": bbox.confidence
                }
                for bbox in sample.bounding_boxes
            ]
        }

        annotations_path = self.annotations_dir / f"{sample.image_id}.json"
        with open(annotations_path, 'w') as f:
            json.dump(annotations, f, indent=2)

    def run_data_generation(self, num_samples: int = 10000):
        """
        Run the complete data generation pipeline.

        Args:
            num_samples: Number of samples to generate
        """
        print("Starting synthetic data generation pipeline...")

        # Generate the dataset
        samples = self.generate_dataset(num_samples=num_samples)

        # Generate quality report
        self._generate_quality_report(samples)

        print(f"Synthetic data generation complete. Dataset saved to: {self.output_dir}")

    def _generate_quality_report(self, samples: List[SyntheticDataSample]):
        """Generate a quality report for the generated dataset."""
        report = {
            "total_samples": len(samples),
            "class_distribution": {},
            "average_objects_per_image": 0,
            "image_resolution": f"{samples[0].rgb_image.shape[1]}x{samples[0].rgb_image.shape[0]}" if samples else "N/A"
        }

        # Count class distribution
        total_objects = 0
        for sample in samples:
            for bbox in sample.bounding_boxes:
                class_name = self.object_classes[bbox.class_id]
                if class_name not in report["class_distribution"]:
                    report["class_distribution"][class_name] = 0
                report["class_distribution"][class_name] += 1
                total_objects += 1

        report["average_objects_per_image"] = total_objects / len(samples) if samples else 0

        # Save report
        report_path = self.output_dir / "quality_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)


def main():
    """Main function to run the synthetic data generation pipeline."""
    generator = SyntheticDataGenerator()
    generator.run_data_generation(num_samples=100)  # Using 100 for testing, can be increased to 10000+


if __name__ == "__main__":
    main()