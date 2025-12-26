# Robot Description Package

ROS 2 package containing URDF/XACRO robot description and simulation launch files for the humanoid robot.

## Package Contents

```
robot_description/
├── urdf/
│   └── humanoid.urdf.xacro          # Main robot description with Gazebo tags
├── worlds/
│   └── humanoid_world.world         # Gazebo world file
├── launch/
│   ├── view_robot.launch.py         # RViz visualization
│   └── gazebo_sim.launch.py         # Gazebo physics simulation
├── rviz/
│   └── robot_view.rviz              # RViz configuration
├── meshes/                          # (Optional) Robot mesh files
├── package.xml                      # Package dependencies
├── CMakeLists.txt                   # Build configuration
└── README.md                        # This file
```

## Quick Start

### View Robot in RViz (Phase-1)

```bash
source install/setup.bash
ros2 launch robot_description view_robot.launch.py
```

This launches:
- robot_state_publisher (publishes TF)
- joint_state_publisher (publishes joint states)
- RViz2 (visualization)

### Run Physics Simulation in Gazebo (Phase-2)

```bash
source install/setup.bash
ros2 launch robot_description gazebo_sim.launch.py
```

This launches:
- Gazebo server with custom world
- Gazebo client (GUI)
- robot_state_publisher
- joint_state_publisher
- Robot spawner

## Robot Specifications

### Physical Properties
- **Total Mass:** 19 kg
  - Base: 10 kg
  - Legs: 2 kg each
  - Feet: 1 kg each
  - Arms: 1 kg each
  - Head: 2 kg

### Dimensions
- **Height:** ~1.45 m (base_height + leg_length + foot_height)
- **Base:** 0.5m × 0.3m × 0.8m (L × W × H)
- **Leg Length:** 0.6m
- **Foot Size:** 0.15m × 0.1m × 0.05m

### Joints (Phase-3: Actuated)
**Revolute Joints (10):**
1. left_hip_joint (revolute, Y-axis)
2. left_knee_joint (revolute, Y-axis)
3. left_ankle_joint (revolute, Y-axis)
4. right_hip_joint (revolute, Y-axis)
5. right_knee_joint (revolute, Y-axis)
6. right_ankle_joint (revolute, Y-axis)
7. left_shoulder_joint (revolute, Y-axis)
8. left_elbow_joint (revolute, Y-axis)
9. right_shoulder_joint (revolute, Y-axis)
10. right_elbow_joint (revolute, Y-axis)

**Fixed Joints (1):**
11. head_joint (fixed)

## Gazebo Physics Properties

### Friction Coefficients
- **Feet:** μ₁=1.0, μ₂=1.0 (maximum traction)
- **Legs/Body:** μ₁=0.8, μ₂=0.8
- **Arms/Head:** μ₁=0.5, μ₂=0.5

### Contact Properties
- **Spring Constant (kp):**
  - Feet: 1e8 (very stiff)
  - Other: 1e7 (stiff)
- **Damping (kd):** 1.0
- **Min Depth:** 0.001m

### World Settings
- **Gravity:** [0, 0, -9.81] m/s²
- **Time Step:** 1ms
- **Update Rate:** 1000 Hz
- **Real-Time Factor:** 1.0

## Verification

### Check Topics
```bash
# Joint states (should be ~50 Hz)
ros2 topic hz /joint_states

# TF transforms
ros2 run tf2_tools view_frames

# Clock (simulation time)
ros2 topic echo /clock
```

### Run Validation Script
```bash
python3 scripts/validate_gazebo_physics.py
```

## Troubleshooting

### Robot falls through ground
- Check collision geometries in URDF
- Verify ground plane in world file
- Increase kp in Gazebo tags

### Robot bounces or jitters
- Increase damping (kd)
- Check inertial properties
- Reduce time step

### No joint states published
- Verify plugin loaded: `ros2 topic list`
- Check Gazebo console for errors
- Restart Gazebo

### TF errors
- Verify robot_state_publisher is running
- Check robot_description parameter
- Use `ros2 run tf2_tools view_frames`

## Dependencies

- `ament_cmake`
- `robot_state_publisher`
- `joint_state_publisher`
- `xacro`
- `rviz2`
- `gazebo_ros`
- `gazebo_ros_pkgs`
- `gazebo_plugins`

## Building

```bash
cd /path/to/workspace
colcon build --packages-select robot_description --symlink-install
source install/setup.bash
```

## Phase Status

- ✅ **Phase-1:** URDF/XACRO model with RViz visualization
- ✅ **Phase-2:** Gazebo physics simulation
- ✅ **Phase-3:** Joint actuation & sensor simulation (COMPLETE)
- ⏳ **Phase-4:** Advanced control & AI integration (Module-3)

## Documentation

- **Phase-2 Physics Guide:** `docs/gazebo_physics_quickstart.md`
- **Phase-3 Actuation Guide:** `docs/actuation_sensors_quickstart.md`
- **Phase-2 Summary:** `PHASE2_GAZEBO_COMPLETE.md`
- **Phase-3 Summary:** `PHASE3_ACTUATION_SENSORS_COMPLETE.md`
- **Validation Scripts:**
  - `scripts/validate_gazebo_physics.py` (Phase-2)
  - `scripts/validate_actuation_and_sensors.py` (Phase-3)

## Notes

- **Total DOF:** 10 actuated joints
- **Sensors:** IMU (100 Hz), LiDAR (10 Hz), RGB-D Camera (30 Hz)
- **Controllers:** ros2_control with position and effort interfaces
- **Physics:** ODE engine with realistic friction and damping
- **Geometry:** Basic shapes (can add meshes later)

## Future Enhancements

1. Add actuated joints (revolute for hips, knees, ankles)
2. Implement joint controllers (position/velocity/effort)
3. Add sensors (camera, lidar, IMU)
4. Integrate Isaac Sim for advanced physics
5. Add detailed mesh models
6. Implement balance and locomotion controllers

## License

TODO

## Maintainer

robot@todo.todo
