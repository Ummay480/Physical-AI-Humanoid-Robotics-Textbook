# Actuation & Sensors Quickstart Guide

## Quick Reference

**Launch Simulation:**
```bash
ros2 launch robot_description gazebo_sim.launch.py
```

**Validate Everything:**
```bash
python3 scripts/validate_actuation_and_sensors.py
```

**Test Joint Movement:**
```bash
ros2 topic pub --once /position_controller/commands std_msgs/msg/Float64MultiArray \
  "{data: [0.0, 0.5, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0]}"
```

## Overview

This guide covers testing the Phase-3 implementation:
- 10 actuated joints (hips, knees, ankles, shoulders, elbows)
- ros2_control integration
- IMU sensor
- LiDAR sensor
- RGB-D camera

## Prerequisites

- ROS 2 Humble
- Gazebo Classic
- robot_description package built
- All Phase-1 and Phase-2 requirements

## Launch Simulation

```bash
# Terminal 1: Launch Gazebo with controllers
source install/setup.bash
ros2 launch robot_description gazebo_sim.launch.py
```

**Expected startup sequence:**
1. Gazebo window opens
2. World loads (ground plane, lighting)
3. Robot spawns at z=1.5m
4. Robot falls and lands
5. Controllers load automatically
6. Sensors start publishing

## Verify Controllers

```bash
# Terminal 2: Check controllers
ros2 control list_controllers
```

**Expected output:**
```
joint_state_broadcaster[joint_state_broadcaster/JointStateBroadcaster] active
position_controller[position_controllers/JointGroupPositionController] active
```

## Monitor Joint States

```bash
# View joint states
ros2 topic echo /joint_states

# Check frequency
ros2 topic hz /joint_states
# Expected: ~50 Hz
```

**Joint order:**
1. left_hip_joint
2. left_knee_joint
3. left_ankle_joint
4. right_hip_joint
5. right_knee_joint
6. right_ankle_joint
7. left_shoulder_joint
8. left_elbow_joint
9. right_shoulder_joint
10. right_elbow_joint

## Test Joint Commands

### Basic Movement Test

```bash
# Move both knees (slight squat)
ros2 topic pub --once /position_controller/commands std_msgs/msg/Float64MultiArray \
  "{data: [0.0, 0.5, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0]}"
```

**Watch in Gazebo:** Knees should bend forward.

### Return to Zero Position

```bash
ros2 topic pub --once /position_controller/commands std_msgs/msg/Float64MultiArray \
  "{data: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}"
```

### Arm Movement Test

```bash
# Bend both elbows
ros2 topic pub --once /position_controller/commands std_msgs/msg/Float64MultiArray \
  "{data: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0]}"
```

### Complex Pose Example

```bash
# Squat with arms forward
ros2 topic pub --once /position_controller/commands std_msgs/msg/Float64MultiArray \
  "{data: [0.3, 1.2, -0.1, 0.3, 1.2, -0.1, 0.5, 0.5, 0.5, 0.5]}"
```

## Monitor Sensors

### IMU Sensor

```bash
# View IMU data
ros2 topic echo /imu/data

# Check frequency
ros2 topic hz /imu/data
# Expected: ~100 Hz
```

**Data includes:**
- Orientation (quaternion)
- Angular velocity (x, y, z rad/s)
- Linear acceleration (x, y, z m/s²)

### LiDAR Sensor

```bash
# View scan data
ros2 topic echo /scan

# Check frequency
ros2 topic hz /scan
# Expected: ~10 Hz
```

**Data includes:**
- 720 range measurements
- 360° coverage (0.5° resolution)
- Range: 0.1m to 30.0m

**Visualize in RViz:**
```bash
ros2 run rviz2 rviz2
# Add LaserScan display
# Topic: /scan
# Fixed Frame: head
```

### Camera Sensors

```bash
# RGB image
ros2 topic hz /camera/image_raw
# Expected: ~30 Hz

# Depth image
ros2 topic hz /camera/depth/image_raw
# Expected: ~30 Hz

# Point cloud
ros2 topic hz /camera/points
# Expected: ~30 Hz
```

**Visualize in RViz:**
```bash
ros2 run rviz2 rviz2
# Add Image display for /camera/image_raw
# Add PointCloud2 display for /camera/points
# Fixed Frame: head
```

## Run Full Validation

```bash
# Terminal 3: Run validation script
python3 scripts/validate_actuation_and_sensors.py
```

**Expected output:**
```
ACTUATION & SENSOR VALIDATION RESULTS - PHASE 3
================================================================

--- ACTUATION TESTS ---
✓ Joint States Publishing: PASSED (freq=50.0 Hz)
✓ Actuated Joints Present: PASSED (all 10 actuated joints found)
✓ Joint Position Data Valid: PASSED (10 positions)
✓ Joint Velocity Data Present: PASSED (10 velocities)

--- SENSOR TESTS ---
✓ IMU Sensor Publishing: PASSED (freq=100.0 Hz)
✓ LiDAR Sensor Publishing: PASSED (freq=10.0 Hz, 720 points)
✓ RGB Camera Publishing: PASSED (freq=30.0 Hz, 640x480)
✓ Depth Camera Publishing: PASSED (freq=30.0 Hz, 640x480)
✓ Point Cloud Publishing: PASSED (freq=30.0 Hz, 640x480 points)

--- SYSTEM TESTS ---
✓ Simulation Clock Running: PASSED (advanced 10.00s)

SUMMARY: 10/10 tests passed

✅ All tests passed! Digital Twin (Module-2) is fully operational.
```

## Joint Limits Reference

### Leg Joints

| Joint | Min (rad) | Max (rad) | Min (deg) | Max (deg) |
|-------|-----------|-----------|-----------|-----------|
| Hip | -1.047 | 1.571 | -60° | 90° |
| Knee | 0.0 | 2.094 | 0° | 120° |
| Ankle | -0.785 | 0.785 | -45° | 45° |

### Arm Joints

| Joint | Min (rad) | Max (rad) | Min (deg) | Max (deg) |
|-------|-----------|-----------|-----------|-----------|
| Shoulder | -1.571 | 3.142 | -90° | 180° |
| Elbow | 0.0 | 2.356 | 0° | 135° |

## Advanced Testing

### Create a Python Control Script

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
import time

class JointController(Node):
    def __init__(self):
        super().__init__('joint_controller')
        self.publisher = self.create_publisher(
            Float64MultiArray,
            '/position_controller/commands',
            10
        )

    def send_command(self, positions):
        msg = Float64MultiArray()
        msg.data = positions
        self.publisher.publish(msg)
        self.get_logger().info(f'Sent: {positions}')

def main():
    rclpy.init()
    controller = JointController()

    # Squat motion
    controller.send_command([0.3, 1.2, 0.0, 0.3, 1.2, 0.0, 0.0, 0.0, 0.0, 0.0])
    time.sleep(2)

    # Return to neutral
    controller.send_command([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Monitor TF Frames

```bash
# Generate TF tree
ros2 run tf2_tools view_frames

# View PDF
evince frames.pdf
```

### Record and Playback

```bash
# Record all topics for 30 seconds
ros2 bag record -a --duration 30

# Playback
ros2 bag play <bag_file>
```

## Troubleshooting

### Problem: Controllers not loading

```bash
# Check controller manager node
ros2 node list | grep controller_manager

# Manually load controllers
ros2 control load_controller joint_state_broadcaster
ros2 control load_controller position_controller

# Activate controllers
ros2 control set_controller_state joint_state_broadcaster active
ros2 control set_controller_state position_controller active
```

### Problem: Joint commands not working

**Check:**
1. Controller is active: `ros2 control list_controllers`
2. Array size is correct (10 values)
3. Values within joint limits
4. No physics explosions in Gazebo

**Solution:**
```bash
# Restart controller
ros2 control unload_controller position_controller
ros2 control load_controller position_controller
ros2 control set_controller_state position_controller active
```

### Problem: Sensor topics not publishing

**Check:**
```bash
# List all topics
ros2 topic list | grep -E "(imu|scan|camera)"

# Expected:
# /imu/data
# /scan
# /camera/image_raw
# /camera/depth/image_raw
# /camera/points
# /camera/camera_info
# /camera/depth/camera_info
```

**Solution:**
- Restart Gazebo
- Check Gazebo console for plugin errors
- Verify sensor frames: `ros2 run tf2_tools view_frames`

### Problem: Low sensor frequency

**Possible causes:**
- CPU overload
- Gazebo simulation running slow
- Too many plugins loaded

**Solution:**
- Close unnecessary applications
- Reduce Gazebo GUI rendering
- Check real-time factor: should be ~1.0

## Performance Tips

1. **Reduce sensor rates if needed:**
   - Edit URDF update_rate parameters
   - Rebuild package

2. **Use headless Gazebo for better performance:**
   ```bash
   ros2 launch robot_description gazebo_sim.launch.py gui:=false
   ```

3. **Monitor system resources:**
   ```bash
   htop  # CPU/RAM usage
   ```

## Integration Examples

### Subscribe to Joint States

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

class JointStateMonitor(Node):
    def __init__(self):
        super().__init__('joint_state_monitor')
        self.subscription = self.create_subscription(
            JointState,
            '/joint_states',
            self.callback,
            10
        )

    def callback(self, msg):
        for i, name in enumerate(msg.name):
            pos = msg.position[i]
            vel = msg.velocity[i]
            print(f'{name}: pos={pos:.3f}, vel={vel:.3f}')

def main():
    rclpy.init()
    monitor = JointStateMonitor()
    rclpy.spin(monitor)
    monitor.destroy_node()
    rclpy.shutdown()
```

### Subscribe to IMU Data

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
import math

class IMUMonitor(Node):
    def __init__(self):
        super().__init__('imu_monitor')
        self.subscription = self.create_subscription(
            Imu,
            '/imu/data',
            self.callback,
            10
        )

    def callback(self, msg):
        # Convert quaternion to euler angles (simplified)
        q = msg.orientation
        roll = math.atan2(2*(q.w*q.x + q.y*q.z), 1 - 2*(q.x**2 + q.y**2))
        pitch = math.asin(2*(q.w*q.y - q.z*q.x))
        yaw = math.atan2(2*(q.w*q.z + q.x*q.y), 1 - 2*(q.y**2 + q.z**2))

        print(f'Roll: {math.degrees(roll):.1f}°')
        print(f'Pitch: {math.degrees(pitch):.1f}°')
        print(f'Yaw: {math.degrees(yaw):.1f}°')

def main():
    rclpy.init()
    monitor = IMUMonitor()
    rclpy.spin(monitor)
```

## Next Steps

Once all tests pass:
1. ✅ Digital Twin (Module-2) complete
2. ➡️ Proceed to Module-3: AI-Robot Brain
3. Integrate perception algorithms
4. Implement high-level controllers
5. Add SLAM capabilities
6. Develop vision-based navigation

## Quick Command Reference

```bash
# Launch
ros2 launch robot_description gazebo_sim.launch.py

# Controllers
ros2 control list_controllers
ros2 control load_controller <name>
ros2 control set_controller_state <name> active

# Topics
ros2 topic list
ros2 topic echo <topic>
ros2 topic hz <topic>

# Nodes
ros2 node list
ros2 node info <node>

# TF
ros2 run tf2_tools view_frames
ros2 run tf2_ros tf2_echo <source> <target>

# Validation
python3 scripts/validate_actuation_and_sensors.py
```

## Success Criteria

- [ ] Gazebo launches without errors
- [ ] Robot spawns and settles on ground
- [ ] Both controllers active
- [ ] Joint states at 50 Hz
- [ ] All sensors publishing at expected rates
- [ ] Joint commands move joints smoothly
- [ ] No physics instabilities
- [ ] Validation script passes 10/10 tests

---

**Digital Twin Status:** ✅ Operational
**Next:** Module-3 AI Integration
