# Perception and Sensors Module - Quickstart Guide

This guide will get you up and running with the Perception and Sensors Module in 5 minutes.

## Prerequisites

- ROS 2 Humble installed
- Python 3.10+
- CUDA-capable GPU (optional, for YOLO acceleration)

## Quick Installation

```bash
# 1. Install system dependencies
sudo apt-get update
sudo apt-get install -y ros-humble-sensor-msgs ros-humble-cv-bridge \
    ros-humble-pcl-ros python3-opencv

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Build the workspace
cd /mnt/d/aidd/hackathon
colcon build --packages-select perception_module
source install/setup.bash
```

## Running the Complete Pipeline

### Option 1: Launch Everything (Recommended)

```bash
# Launch the complete perception pipeline
ros2 launch perception_module perception_pipeline.launch.py
```

This starts:
- Camera acquisition node
- LIDAR acquisition node
- IMU acquisition node
- Object detection node
- Sensor fusion node

### Option 2: Launch Individual Components

```bash
# Terminal 1: Sensor acquisition
ros2 run perception_module sensor_acquisition_node

# Terminal 2: Object detection
ros2 run perception_module object_detection_node

# Terminal 3: Sensor fusion
ros2 run perception_module sensor_fusion_node
```

## Quick Examples

### Example 1: Monitor Detected Objects

```bash
# Subscribe to detected objects
ros2 topic echo /perception/objects
```

Expected output:
```yaml
objects:
  - object_type: HUMAN
    confidence: 0.95
    position: {x: 2.5, y: 1.0, z: 0.0}
    bounding_box: {x_min: 100, y_min: 200, x_max: 300, y_max: 500}
```

### Example 2: Visualize 3D Occupancy Map

```bash
# Launch RViz with perception config
rviz2 -d config/perception.rviz
```

### Example 3: Monitor System Health

```python
from perception.monitoring.logger import get_health_monitor, get_performance_monitor

# Get health status
health = get_health_monitor()
status = health.get_overall_health()
print(f"System health: {status}")

# Get performance metrics
perf = get_performance_monitor()
stats = perf.get_statistics('processing_latency')
print(f"Average latency: {stats['mean']:.2f}ms")
```

### Example 4: Record Sensor Data

```bash
# Record 30 seconds of data
ros2 bag record -a -d 30
```

### Example 5: Test with Simulated Data

```python
from perception.sensor_acquisition.sensor_manager import SensorManager
from perception.common.config_handler import ConfigHandler
from rclpy.node import Node
import rclpy

# Initialize ROS 2
rclpy.init()
node = Node('test_node')

# Load configuration
config = ConfigHandler('config/camera_front.yaml')
sensor_config = config.load_sensor_config()

# Create sensor manager
manager = SensorManager(node)
manager.add_sensor(sensor_config)

# Start acquisition
manager.start_all()

# Let it run for 10 seconds
import time
time.sleep(10)

# Check statistics
stats = manager.get_statistics()
print(f"Frames captured: {stats['camera_front']['frames_processed']}")

# Cleanup
manager.stop_all()
rclpy.shutdown()
```

## Configuration

### Sensor Configuration

Edit sensor configs in `config/`:
- `camera_front.yaml` - Camera settings
- `lidar_3d.yaml` - LIDAR settings
- `imu_main.yaml` - IMU settings

Example camera config:
```yaml
sensor_id: "camera_front"
sensor_type: "camera"
operational_params:
  frequency: 30.0
  resolution: [640, 480]
  format: "bgr8"
  topic: "/camera/front/image_raw"
```

### System Parameters

Edit `config/perception_params.yaml`:
```yaml
perception:
  object_detection:
    model_type: "yolov8"
    confidence_threshold: 0.5

  sensor_fusion:
    fusion_rate: 50.0
    kalman_process_noise: 0.01
    kalman_measurement_noise: 0.1

  mapping:
    grid_resolution: 0.05
    grid_size: [10.0, 10.0, 3.0]
```

## Monitoring & Debugging

### Check Node Status

```bash
# List running nodes
ros2 node list

# Check node info
ros2 node info /sensor_acquisition_node
```

### Monitor Performance

```bash
# Watch latency metrics
ros2 topic hz /perception/objects

# Check bandwidth
ros2 topic bw /camera/front/image_raw
```

### Enable Debug Logging

```bash
# Set log level to DEBUG
ros2 run perception_module sensor_acquisition_node --ros-args --log-level DEBUG
```

### View Logs

```bash
# Application logs
cat logs/perception.log

# ROS 2 logs
ros2 run rqt_console rqt_console
```

## Performance Tuning

### For Low Latency (\<10ms)

```yaml
# config/perception_params.yaml
perception:
  object_detection:
    enable_gpu: true
    batch_size: 1
  sensor_fusion:
    fusion_rate: 100.0
  performance:
    target_latency_ms: 10.0
```

### For High Accuracy

```yaml
perception:
  object_detection:
    confidence_threshold: 0.8
    nms_threshold: 0.3
  sensor_fusion:
    kalman_process_noise: 0.001
```

### For Resource-Constrained Systems

```yaml
perception:
  object_detection:
    model_type: "yolov8n"  # Nano model
    enable_gpu: false
  mapping:
    grid_resolution: 0.1  # Larger voxels
    enable_mapping: false  # Disable if not needed
```

## Common Issues

### Issue: Camera not detected

```bash
# Check camera devices
v4l2-ctl --list-devices

# Verify ROS 2 camera driver
ros2 run usb_cam usb_cam_node_exe
```

### Issue: High latency

```bash
# Check system load
top

# Verify GPU usage (if applicable)
nvidia-smi

# Reduce detection frequency
ros2 param set /object_detection_node detection_frequency 15.0
```

### Issue: Objects not detected

```bash
# Check camera feed
ros2 run rqt_image_view rqt_image_view

# Verify model is loaded
ros2 service call /object_detection_node/get_parameters rcl_interfaces/srv/GetParameters "{names: ['model_path']}"

# Lower confidence threshold
ros2 param set /object_detection_node confidence_threshold 0.3
```

## Next Steps

1. **Customize Detection**: Train custom YOLO model for specific objects
2. **Add Sensors**: Integrate additional cameras or depth sensors
3. **Extend Mapping**: Implement SLAM or semantic mapping
4. **Optimize**: Profile and optimize for your specific hardware

## Resources

- [ROS 2 Humble Documentation](https://docs.ros.org/en/humble/)

## Support

- Report issues: [GitHub Issues](https://github.com/your-repo/issues)
- Ask questions: [Discussions](https://github.com/your-repo/discussions)
- Email: support@your-project.org
