# Perception Module - Quickstart Guide

A practical guide to get started with the Perception and Sensors module in minutes.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Basic Setup](#basic-setup)
4. [Running Your First Perception Pipeline](#running-your-first-perception-pipeline)
5. [Common Use Cases](#common-use-cases)
6. [Troubleshooting](#troubleshooting)

## Prerequisites

Before starting, ensure you have:

- Ubuntu 22.04 LTS (recommended)
- ROS 2 Humble installed
- Python 3.10+
- NVIDIA GPU (optional, for YOLO acceleration)

## Installation

### 1. Install System Dependencies

```bash
# Install ROS 2 Humble (if not already installed)
sudo apt update
sudo apt install ros-humble-desktop

# Install perception dependencies
sudo apt install -y \
    ros-humble-cv-bridge \
    ros-humble-sensor-msgs \
    ros-humble-vision-msgs \
    python3-pip \
    python3-opencv
```

### 2. Install Python Dependencies

```bash
cd /path/to/hackathon
pip install -r requirements.txt

# Key packages installed:
# - ultralytics (YOLO)
# - numpy, opencv-python
# - scikit-learn
# - pyyaml
```

### 3. Build the Perception Package

```bash
# From workspace root
cd /path/to/hackathon

# Build perception package
colcon build --packages-select perception

# Source the workspace
source install/setup.bash
```

### 4. Download YOLO Model (Optional)

```bash
# Download YOLOv8 nano model
mkdir -p models
cd models
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
cd ..
```

## Basic Setup

### Configure Your Sensors

Edit sensor configuration files in `config/`:

**Camera Configuration** (`config/camera_front.yaml`):
```yaml
sensor_id: "camera_front"
sensor_type: "camera"

operational_params:
  frequency: 30.0
  resolution: [640, 480]
  format: "bgr8"
  topic: "/camera_front/image_raw"
```

**LIDAR Configuration** (`config/lidar_3d.yaml`):
```yaml
sensor_id: "lidar_3d"
sensor_type: "lidar"

operational_params:
  frequency: 10.0
  range_min: 0.1
  range_max: 30.0
  topic: "/lidar_3d/scan"
```

**IMU Configuration** (`config/imu_main.yaml`):
```yaml
sensor_id: "imu_main"
sensor_type: "imu"

operational_params:
  frequency: 100.0
  topic: "/imu/data"
```

### Verify Configuration

```bash
# Check if config files are valid
python3 -c "
from src.perception.common.config_handler import ConfigHandler
config = ConfigHandler()
print('Camera config:', config.load_sensor_config('config/camera_front.yaml'))
"
```

## Running Your First Perception Pipeline

### Quick Start - Simulated Data

Run the perception pipeline with test/simulated data:

```bash
# Terminal 1: Launch the perception pipeline
ros2 launch src/launch/perception_pipeline.launch.py

# Terminal 2: Publish test camera data
ros2 run image_tools cam2image --ros-args -r /image:=/camera_front/image_raw

# Terminal 3: Monitor detected objects
ros2 topic echo /perception/objects_detected
```

### Production Start - Real Sensors

```bash
# Launch with your sensor drivers
ros2 launch perception perception_pipeline.launch.py \
    camera_config:=config/camera_front.yaml \
    lidar_config:=config/lidar_3d.yaml \
    imu_config:=config/imu_main.yaml \
    model_path:=models/yolov8n.pt \
    enable_visualization:=true
```

### Verify Pipeline is Running

```bash
# Check all nodes are active
ros2 node list
# Expected output:
# /perception/sensor_acquisition_node
# /perception/object_detection_node
# /perception/sensor_fusion_node

# Check topics are publishing
ros2 topic list
# Should include:
# /perception/objects_detected
# /perception/environmental_map
# /perception/fused_data
# /perception/system_status

# Monitor update rates
ros2 topic hz /perception/objects_detected
# Expected: ~30 Hz
```

## Common Use Cases

### Use Case 1: Object Detection Only

If you only need object detection without fusion:

```python
#!/usr/bin/env python3
"""Simple object detection example."""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import json

from src.perception.computer_vision.object_detector import YOLODetector

class SimpleDetectionNode(Node):
    def __init__(self):
        super().__init__('simple_detection')

        # Create detector
        self.detector = YOLODetector(
            model_path='models/yolov8n.pt',
            confidence_threshold=0.5
        )

        # Setup ROS 2
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(
            Image,
            '/camera_front/image_raw',
            self.image_callback,
            10
        )

        self.publisher = self.create_publisher(
            String,
            '/detections',
            10
        )

    def image_callback(self, msg):
        # Convert ROS image to OpenCV
        cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

        # Run detection
        detections = self.detector.detect(cv_image)

        # Publish results
        results = {
            'timestamp': self.get_clock().now().to_msg(),
            'count': len(detections),
            'objects': [
                {
                    'class': det.object_type.value,
                    'confidence': det.confidence,
                    'position': det.position.tolist()
                }
                for det in detections
            ]
        }

        self.publisher.publish(json.dumps(results))
        self.get_logger().info(f'Detected {len(detections)} objects')

def main():
    rclpy.init()
    node = SimpleDetectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

Run it:
```bash
chmod +x simple_detection.py
ros2 run python simple_detection.py
```

### Use Case 2: Sensor Fusion for Position Tracking

Track object positions using multi-sensor fusion:

```python
#!/usr/bin/env python3
"""Multi-sensor position tracking example."""

import numpy as np
from src.perception.sensor_fusion.kalman_filter import MultiSensorKalmanFilter
from src.perception.sensor_fusion.data_fusion import SensorDataFusion

# Create multi-sensor Kalman filter
kf = MultiSensorKalmanFilter(dim_x=6, dim_z=3)  # Track [x, y, z, vx, vy, vz]

# Initialize with first measurement
initial_position = np.array([0.0, 0.0, 0.0])
kf.initialize(initial_position)

# Create fusion engine
fusion = SensorDataFusion()

# Simulate sensor measurements
camera_position = np.array([1.5, 0.2, 0.1])
lidar_position = np.array([1.48, 0.18, 0.12])

# Fuse positions with confidence weights
positions = {
    'camera': camera_position,
    'lidar': lidar_position
}
confidences = {
    'camera': 0.7,  # 70% confidence
    'lidar': 0.9    # 90% confidence
}

fused_position, fused_confidence = fusion.fuse_position_estimates(
    positions, confidences
)

print(f"Fused position: {fused_position}")
print(f"Fused confidence: {fused_confidence:.2f}")

# Update Kalman filter
kf.update('sensor_1', fused_position, timestamp=0.1)
kf.predict()

# Get filtered state
state = kf.get_state()
print(f"Filtered position: {state[:3]}")
print(f"Estimated velocity: {state[3:]}")
```

### Use Case 3: 3D Environmental Mapping

Build a 3D occupancy map from LIDAR data:

```python
#!/usr/bin/env python3
"""3D mapping example."""

import numpy as np
from src.perception.sensor_fusion.mapping import OccupancyGrid3D, PointCloudProcessor

# Create 3D occupancy grid (5cm resolution, 10m range)
grid = OccupancyGrid3D(
    resolution=0.05,
    size=(200, 200, 100),  # 10m x 10m x 5m
    origin=np.array([-5.0, -5.0, 0.0])
)

# Simulate LIDAR point cloud
num_points = 1000
points = np.random.rand(num_points, 3) * 10 - 5  # Random points in range

# Process point cloud
processor = PointCloudProcessor()

# Downsample
downsampled = processor.downsample(points, voxel_size=0.1)
print(f"Downsampled from {len(points)} to {len(downsampled)} points")

# Remove outliers
cleaned = processor.remove_outliers(downsampled)
print(f"Removed {len(downsampled) - len(cleaned)} outliers")

# Update occupancy grid
sensor_origin = np.array([0.0, 0.0, 1.0])
grid.update_with_point_cloud(cleaned, sensor_origin)

# Mark detected object as occupied
object_position = np.array([2.0, 1.0, 0.5])
object_size = np.array([0.5, 0.5, 1.0])
grid.mark_object(object_position, object_size)

# Query occupancy
test_point = np.array([2.0, 1.0, 0.5])
occupancy_prob = grid.get_occupancy(test_point)
print(f"Occupancy probability at {test_point}: {occupancy_prob:.2f}")

# Export map
grid.export_to_file('environmental_map.npy')
print("Map saved to environmental_map.npy")
```

### Use Case 4: Performance Monitoring

Monitor and optimize perception pipeline performance:

```python
#!/usr/bin/env python3
"""Performance monitoring example."""

from src.perception.monitoring.logger import (
    PerceptionLogger,
    PerformanceMonitor,
    HealthMonitor,
    TimingContext
)
from src.perception.monitoring.performance_optimizer import PerformanceOptimizer
import time

# Setup logging
logger = PerceptionLogger('perception_monitor', log_dir='logs')

# Setup monitoring
perf_monitor = PerformanceMonitor(window_size=1000)
health_monitor = HealthMonitor()

# Create optimizer
optimizer = PerformanceOptimizer()

# Simulate processing loop
for i in range(100):
    # Time the operation
    with TimingContext('detection_cycle', perf_monitor, logger):
        # Simulate detection work
        time.sleep(0.01)  # 10ms processing time

    # Record custom metrics
    perf_monitor.record_metric('fps', 30.0, 'Hz')
    perf_monitor.record_metric('cpu_usage', 45.5, '%')

    # Check health periodically
    if i % 10 == 0:
        health_results = health_monitor.check_health(perf_monitor)
        logger.info(f"Health check: {health_results['overall']}")

# Get statistics
stats = perf_monitor.get_all_statistics()
print("\nPerformance Statistics:")
for metric, values in stats.items():
    print(f"{metric}:")
    print(f"  Mean: {values['mean']:.2f} {values.get('unit', '')}")
    print(f"  P95: {values['p95']:.2f}")
    print(f"  P99: {values['p99']:.2f}")

# Get optimization suggestions
report = optimizer.get_optimization_report()
suggestions = optimizer.suggest_optimizations()

print("\nOptimization Suggestions:")
for suggestion in suggestions:
    print(f"  - {suggestion}")
```

### Use Case 5: Custom Feature Extraction

Extract visual features for custom processing:

```python
#!/usr/bin/env python3
"""Feature extraction example."""

import cv2
import numpy as np
from src.perception.computer_vision.feature_extractor import FeatureExtractor
from src.perception.computer_vision.cv_utils import preprocess_image

# Load test image
image = cv2.imread('test_image.jpg')

# Preprocess
preprocessed = preprocess_image(
    image,
    target_size=(640, 480),
    normalize=True,
    enhance=True
)

# Create feature extractor
extractor = FeatureExtractor()

# Extract SIFT features
sift_keypoints, sift_descriptors = extractor.extract_sift(preprocessed)
print(f"Detected {len(sift_keypoints)} SIFT keypoints")

# Extract ORB features
orb_keypoints, orb_descriptors = extractor.extract_orb(preprocessed)
print(f"Detected {len(orb_keypoints)} ORB keypoints")

# Extract HOG features
hog_features = extractor.extract_hog(preprocessed)
print(f"HOG feature vector size: {len(hog_features)}")

# Extract edges
edges = extractor.extract_edges(preprocessed, method='canny')

# Extract corners
corners = extractor.extract_corners(preprocessed, method='harris', max_corners=100)
print(f"Detected {len(corners)} corners")

# Visualize features
vis_image = cv2.drawKeypoints(
    preprocessed,
    sift_keypoints,
    None,
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

cv2.imwrite('features_visualization.jpg', vis_image)
print("Visualization saved to features_visualization.jpg")
```

## Troubleshooting

### Problem: "No sensor data received"

**Symptoms:** No data on `/perception/objects_detected` topic

**Solutions:**
```bash
# 1. Check if sensor topics are publishing
ros2 topic list | grep -E "camera|lidar|imu"

# 2. Check topic rates
ros2 topic hz /camera_front/image_raw

# 3. Verify sensor configurations
cat config/camera_front.yaml

# 4. Check node logs
ros2 node info /perception/sensor_acquisition_node

# 5. Enable debug logging
ros2 launch perception perception_pipeline.launch.py log_level:=DEBUG
```

### Problem: "Low detection accuracy"

**Symptoms:** Objects not detected or many false positives

**Solutions:**
```python
# 1. Adjust confidence threshold
detector = YOLODetector(
    model_path='models/yolov8n.pt',
    confidence_threshold=0.3  # Lower for more detections (more false positives)
)

# 2. Try a larger YOLO model
# Download YOLOv8 medium model (better accuracy, slower)
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt

# 3. Improve lighting/image quality
preprocessed = preprocess_image(
    image,
    enhance=True,  # Enable CLAHE enhancement
    denoise=True   # Enable denoising
)

# 4. Calibrate camera
# Run camera calibration and update camera_front.yaml
```

### Problem: "High latency (>20ms)"

**Symptoms:** System feels sluggish, latency warnings in logs

**Solutions:**
```yaml
# 1. Reduce sensor frequencies in config/perception_params.yaml
sensors:
  camera:
    frequency: 15.0  # Reduce from 30Hz
  lidar:
    frequency: 5.0   # Reduce from 10Hz

# 2. Lower image resolution
operational_params:
  resolution: [320, 240]  # Reduce from [640, 480]

# 3. Disable visualization
ros2 launch perception perception_pipeline.launch.py enable_visualization:=false

# 4. Use smaller YOLO model
model_path: "models/yolov8n.pt"  # Nano is fastest
```

### Problem: "Memory usage growing over time"

**Symptoms:** System slows down, eventual crash

**Solutions:**
```python
# 1. Enable memory monitoring
from src.perception.monitoring.performance_optimizer import MemoryOptimizer

mem_optimizer = MemoryOptimizer()

# Check periodically
if mem_optimizer.check_memory_leak(threshold_mb=200):
    logger.warning("Potential memory leak detected")

# 2. Optimize numpy arrays
import numpy as np
array = mem_optimizer.optimize_numpy_array(large_array)

# 3. Reduce history window sizes
perf_monitor = PerformanceMonitor(window_size=100)  # Reduce from 1000

# 4. Clear caches periodically
cache_manager.clear()
```

### Problem: "ROS 2 nodes not communicating"

**Symptoms:** Nodes running but no data flow

**Solutions:**
```bash
# 1. Check ROS 2 domain
echo $ROS_DOMAIN_ID
# Ensure all nodes use same domain

# 2. Verify QoS compatibility
ros2 topic info /perception/objects_detected -v

# 3. Check network configuration
ros2 doctor

# 4. Restart ROS 2 daemon
ros2 daemon stop
ros2 daemon start
```

## Next Steps

- Join discussions on [ROS Discourse](https://discourse.ros.org/)
- Refer to [ROS 2 Humble Documentation](https://docs.ros.org/en/humble/) for more information

## Support

For issues and questions:
- GitHub Issues: [hackathon/issues](https://github.com/your-org/hackathon/issues)
- ROS 2 Humble Docs: https://docs.ros.org/en/humble/

---

**Happy Coding!** 🤖
