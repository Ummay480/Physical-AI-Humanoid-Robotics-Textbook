# Gazebo Physics Simulation Quickstart

## Overview
This guide covers Phase-2 of Module-2: Gazebo Physics Simulation Setup for the humanoid robot. The setup enables physics-based simulation with gravity, collisions, and ground interaction.

## Prerequisites
- ROS 2 Humble (or compatible version)
- Gazebo Classic or Gazebo Fortress
- robot_description package (Phase-1 completed)

## What Was Implemented

### 1. Gazebo World File
**Location:** `src/robot_description/worlds/humanoid_world.world`

Features:
- Ground plane with friction coefficients (mu=100, mu2=50)
- Gravity enabled (9.81 m/s²)
- Directional sun light with shadows
- Point light for ambient illumination
- Physics engine: ODE with 1ms step size
- Real-time factor: 1.0

### 2. Enhanced URDF/XACRO with Gazebo Tags
**Location:** `src/robot_description/urdf/humanoid.urdf.xacro`

Added Gazebo-specific properties:
- **Materials:** Gazebo/White, Gazebo/Blue, Gazebo/Red
- **Friction coefficients:**
  - Feet: mu1=1.0, mu2=1.0 (high friction for stability)
  - Body/legs: mu1=0.8, mu2=0.8
  - Arms/head: mu1=0.5, mu2=0.5
- **Contact properties:**
  - kp=1e7 (spring constant)
  - kd=1 (damping coefficient)
  - minDepth=0.001 (minimum collision depth)
- **Gazebo plugins:**
  - gazebo_ros_joint_state_publisher (50 Hz update rate)
  - gazebo_ros2_control (for future control integration)

### 3. Gazebo Launch File
**Location:** `src/robot_description/launch/gazebo_sim.launch.py`

Launch components:
- Gazebo server (gzserver) with custom world
- Gazebo client (gzclient) for visualization
- Robot state publisher (publishes TF transforms)
- Joint state publisher (publishes joint states)
- Robot spawner (spawns robot at z=1.5m to demonstrate falling physics)

### 4. Updated Dependencies
**Files:** `package.xml`, `CMakeLists.txt`

Added dependencies:
- gazebo_ros
- gazebo_ros_pkgs
- gazebo_plugins

Installed directories:
- worlds (Gazebo world files)
- launch (includes new gazebo_sim.launch.py)

## Build Instructions

```bash
# Navigate to workspace root
cd /mnt/d/aidd/hackathon

# Source ROS 2
source /opt/ros/humble/setup.bash

# Build the robot_description package
colcon build --packages-select robot_description --symlink-install

# Source the workspace
source install/setup.bash
```

## Running the Simulation

### Launch Gazebo with Robot

```bash
# Source your workspace
source install/setup.bash

# Launch Gazebo simulation
ros2 launch robot_description gazebo_sim.launch.py
```

### Expected Behavior

1. **Gazebo starts** with the custom world (ground plane, lighting)
2. **Robot spawns** at height z=1.5m above the ground
3. **Robot falls** due to gravity and lands on its feet
4. **Physics interactions:**
   - Collisions between robot and ground
   - Friction forces keeping robot stable
   - Gravity pulling robot down (9.81 m/s²)
5. **Joint states** published at 50 Hz
6. **TF transforms** published from all robot links

## Verification Steps

### 1. Check Robot Spawned Correctly

```bash
# List entities in Gazebo
gz model --list
# Should show: humanoid_robot
```

### 2. Verify Joint State Publishing

```bash
# Monitor joint states
ros2 topic echo /joint_states

# Expected output: joint positions and velocities
```

### 3. Verify TF Transforms

```bash
# List all TF frames
ros2 run tf2_tools view_frames

# Check TF tree
ros2 run tf2_ros tf2_echo base_link left_foot
```

### 4. Monitor Physics Performance

```bash
# Check Gazebo topics
ros2 topic list | grep gazebo

# Monitor clock
ros2 topic echo /clock

# Should update at real-time rate
```

### 5. Visualize in RViz (Optional)

```bash
# Launch RViz alongside Gazebo
ros2 run rviz2 rviz2

# Add displays:
# - RobotModel (topic: /robot_description)
# - TF
# - Set Fixed Frame: base_link or odom
```

## Testing Physics Interactions

### Test 1: Falling Physics
**Objective:** Verify gravity and collision detection

1. Robot should fall from spawn height (1.5m)
2. Robot should collide with ground plane
3. Robot should remain stable on ground (not sink or bounce excessively)

**Success Criteria:**
- Robot settles on ground within 2-3 seconds
- No jittering or unstable behavior
- Feet make proper contact with ground

### Test 2: Joint State Publishing
**Objective:** Verify Gazebo publishes joint states

```bash
ros2 topic hz /joint_states
# Expected: ~50 Hz
```

**Success Criteria:**
- Joint states published at 50 Hz
- All joints present in message

### Test 3: TF Broadcasting
**Objective:** Verify coordinate frame transforms

```bash
ros2 run tf2_ros tf2_echo base_link left_foot
```

**Success Criteria:**
- Transforms available for all links
- Transform updates at high frequency
- No TF errors or warnings

### Test 4: Ground Contact
**Objective:** Verify friction and contact forces

**Manual Test:**
1. In Gazebo GUI, try to apply forces to robot (if interactive mode available)
2. Robot should resist sliding due to friction
3. Robot should maintain balance

**Success Criteria:**
- Feet don't slide on ground
- Robot remains stable
- Friction coefficients working correctly

## Troubleshooting

### Issue: Robot falls through ground
**Solution:**
- Check collision geometries in URDF
- Verify ground plane in world file
- Increase contact stiffness (kp) in Gazebo tags

### Issue: Robot bounces or jitters
**Solution:**
- Increase damping (kd) in Gazebo tags
- Reduce physics step size in world file
- Check inertial properties (mass, inertia)

### Issue: Joint states not published
**Solution:**
- Verify gazebo_ros_joint_state_publisher plugin in URDF
- Check plugin is loaded: `ros2 topic list | grep joint_states`
- Restart Gazebo

### Issue: TF transforms missing
**Solution:**
- Verify robot_state_publisher is running
- Check robot_description parameter is set
- Use `ros2 run tf2_tools view_frames` to debug

### Issue: Slow simulation
**Solution:**
- Reduce physics update rate in world file
- Simplify collision geometries
- Close unnecessary GUI windows

## Configuration Files Reference

### World File Settings
```xml
<physics type="ode">
  <max_step_size>0.001</max_step_size>          <!-- 1ms time step -->
  <real_time_factor>1.0</real_time_factor>      <!-- Real-time simulation -->
  <real_time_update_rate>1000.0</real_time_update_rate>  <!-- 1000 Hz -->
  <gravity>0 0 -9.81</gravity>                  <!-- Earth gravity -->
</physics>
```

### Friction Coefficients
- **mu1, mu2:** Coulomb friction coefficients
  - 1.0 = high friction (feet)
  - 0.8 = medium-high friction (legs, body)
  - 0.5 = medium friction (arms, head)

### Contact Properties
- **kp (spring constant):** 1e7-1e8 (stiffness)
- **kd (damping):** 1 (damping coefficient)
- **minDepth:** 0.001 (penetration tolerance)

## Next Steps (Phase-3)

Once physics simulation is verified:
1. Add actuated joints (revolute joints for hips, knees, ankles)
2. Implement joint controllers (position, velocity, effort)
3. Add sensor integration (IMU, force sensors)
4. Develop balance controllers
5. Test walking gaits

## Important Notes

- **Phase-2 Focus:** Physics simulation ONLY
- **No sensors yet:** Camera, lidar, IMU come in later phases
- **No AI/control:** Basic physics testing only
- **No SLAM:** Navigation comes after sensor integration
- **No Isaac Sim yet:** Using Gazebo Classic for now

## File Summary

Created/Modified files:
```
src/robot_description/
├── worlds/
│   └── humanoid_world.world              [NEW] Gazebo world file
├── urdf/
│   └── humanoid.urdf.xacro              [MODIFIED] Added Gazebo tags
├── launch/
│   ├── gazebo_sim.launch.py             [NEW] Gazebo launch file
│   └── view_robot.launch.py             [EXISTING] RViz launch
├── package.xml                          [MODIFIED] Added Gazebo deps
└── CMakeLists.txt                       [MODIFIED] Install worlds dir
```

## Contact Physics Parameters

Recommended tuning for humanoid stability:

| Link | mu1 | mu2 | kp | kd | Purpose |
|------|-----|-----|----|----|---------|
| Feet | 1.0 | 1.0 | 1e8 | 1 | Max traction, no slip |
| Legs | 0.8 | 0.8 | 1e7 | 1 | General contact |
| Body | 0.8 | 0.8 | 1e7 | 1 | General contact |
| Arms | 0.5 | 0.5 | 1e7 | 1 | Low friction |
| Head | 0.5 | 0.5 | 1e7 | 1 | Low friction |

## Success Criteria Checklist

- [ ] Gazebo launches without errors
- [ ] Robot spawns at correct position (0, 0, 1.5)
- [ ] Robot falls and lands on ground
- [ ] Robot remains stable (no sinking/bouncing)
- [ ] Joint states published at 50 Hz
- [ ] TF transforms available for all links
- [ ] Ground plane visible and collidable
- [ ] Lighting and shadows working
- [ ] Physics runs in real-time
- [ ] No error messages in console

---

**Phase-2 Complete!** Ready for Phase-3: Joint Control and Actuation
