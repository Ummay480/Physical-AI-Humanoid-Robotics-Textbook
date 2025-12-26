# URDF for Humanoid Robot Control

## Introduction to URDF

URDF (Unified Robot Description Format) is an XML-based format used in ROS to describe robot models. For humanoid robots, URDF defines the physical structure, including links (rigid bodies), joints (connections between links), and their properties. Understanding URDF is crucial for humanoid robot control as it provides the geometric and kinematic model that controllers use to understand the robot's structure.

## URDF Structure for Humanoid Robots

A humanoid robot URDF typically includes:

1. **Links**: Represent rigid bodies (torso, head, arms, legs)
2. **Joints**: Define connections between links (shoulders, elbows, knees, etc.)
3. **Visual and Collision Elements**: Define how the robot appears visually and how collisions are detected
4. **Inertial Properties**: Define mass, center of mass, and inertia for each link
5. **ros2_control Interface**: Define control interfaces for each joint

## Basic URDF Components

### Links

Links represent rigid bodies in the robot. Each link must have:
- A unique name
- Visual elements (for display)
- Collision elements (for collision detection)
- Inertial properties (for physics simulation)

```xml
<link name="base_link">
  <visual>
    <geometry>
      <cylinder length="0.6" radius="0.2"/>
    </geometry>
    <material name="blue">
      <color rgba="0 0 0.8 1"/>
    </material>
  </visual>
  <collision>
    <geometry>
      <cylinder length="0.6" radius="0.2"/>
    </geometry>
  </collision>
  <inertial>
    <mass value="10"/>
    <origin xyz="0 0 0"/>
    <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0"/>
  </inertial>
</link>
```

### Joints

Joints connect links and define their relative motion. Common joint types for humanoid robots:
- **revolute**: Rotational joint with limits
- **continuous**: Rotational joint without limits
- **prismatic**: Linear sliding joint
- **fixed**: No motion between links

```xml
<joint name="hip_joint" type="revolute">
  <parent link="torso"/>
  <child link="left_leg"/>
  <origin xyz="0 -0.1 -0.3" rpy="0 0 0"/>
  <axis xyz="0 0 1"/>
  <limit lower="-1.57" upper="1.57" effort="100" velocity="1"/>
</joint>
```

## Complete Humanoid Robot URDF Example

Here's a simplified humanoid robot URDF with minimum 5 links and 4 joints as required by the specification:

```xml
<?xml version="1.0"?>
<robot name="simple_humanoid" xmlns:xacro="http://www.ros.org/wiki/xacro">
  <!-- Base link - torso -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.3 0.2 0.6"/>
      </geometry>
      <material name="light_grey">
        <color rgba="0.7 0.7 0.7 1.0"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.3 0.2 0.6"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="10.0"/>
      <origin xyz="0 0 0"/>
      <inertia ixx="0.5" ixy="0.0" ixz="0.0" iyy="0.6" iyz="0.0" izz="0.3"/>
    </inertial>
  </link>

  <!-- Head -->
  <link name="head">
    <visual>
      <geometry>
        <sphere radius="0.1"/>
      </geometry>
      <material name="white">
        <color rgba="1.0 1.0 1.0 1.0"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <sphere radius="0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="2.0"/>
      <origin xyz="0 0 0"/>
      <inertia ixx="0.02" ixy="0.0" ixz="0.0" iyy="0.02" iyz="0.0" izz="0.02"/>
    </inertial>
  </link>

  <!-- Left Arm -->
  <link name="left_arm">
    <visual>
      <geometry>
        <cylinder length="0.4" radius="0.05"/>
      </geometry>
      <material name="blue">
        <color rgba="0.0 0.0 1.0 1.0"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.4" radius="0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <origin xyz="0 0 -0.2"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.001"/>
    </inertial>
  </link>

  <!-- Right Arm -->
  <link name="right_arm">
    <visual>
      <geometry>
        <cylinder length="0.4" radius="0.05"/>
      </geometry>
      <material name="blue">
        <color rgba="0.0 0.0 1.0 1.0"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.4" radius="0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <origin xyz="0 0 -0.2"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.001"/>
    </inertial>
  </link>

  <!-- Left Leg -->
  <link name="left_leg">
    <visual>
      <geometry>
        <cylinder length="0.5" radius="0.06"/>
      </geometry>
      <material name="red">
        <color rgba="1.0 0.0 0.0 1.0"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.5" radius="0.06"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="2.0"/>
      <origin xyz="0 0 -0.25"/>
      <inertia ixx="0.05" ixy="0.0" ixz="0.0" iyy="0.05" iyz="0.0" izz="0.002"/>
    </inertial>
  </link>

  <!-- Right Leg -->
  <link name="right_leg">
    <visual>
      <geometry>
        <cylinder length="0.5" radius="0.06"/>
      </geometry>
      <material name="red">
        <color rgba="1.0 0.0 0.0 1.0"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.5" radius="0.06"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="2.0"/>
      <origin xyz="0 0 -0.25"/>
      <inertia ixx="0.05" ixy="0.0" ixz="0.0" iyy="0.05" iyz="0.0" izz="0.002"/>
    </inertial>
  </link>

  <!-- Joints connecting the links -->
  <joint name="torso_head_joint" type="revolute">
    <parent link="base_link"/>
    <child link="head"/>
    <origin xyz="0 0 0.35"/>
    <axis xyz="0 1 0"/>
    <limit lower="-0.5" upper="0.5" effort="10" velocity="1"/>
  </joint>

  <joint name="torso_left_arm_joint" type="revolute">
    <parent link="base_link"/>
    <child link="left_arm"/>
    <origin xyz="0.15 0 0.1" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="10" velocity="1"/>
  </joint>

  <joint name="torso_right_arm_joint" type="revolute">
    <parent link="base_link"/>
    <child link="right_arm"/>
    <origin xyz="-0.15 0 0.1" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="10" velocity="1"/>
  </joint>

  <joint name="torso_left_leg_joint" type="revolute">
    <parent link="base_link"/>
    <child link="left_leg"/>
    <origin xyz="0.05 0 -0.3"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="20" velocity="1"/>
  </joint>

  <!-- Additional joint for right leg -->
  <joint name="torso_right_leg_joint" type="revolute">
    <parent link="base_link"/>
    <child link="right_leg"/>
    <origin xyz="-0.05 0 -0.3"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="20" velocity="1"/>
  </joint>

  <!-- ros2_control interface for robot control -->
  <ros2_control name="GazeboSystem" type="system">
    <hardware>
      <plugin>gazebo_ros2_control/GazeboSystem</plugin>
    </hardware>

    <!-- Head joint interface -->
    <joint name="torso_head_joint">
      <command_interface name="position">
        <param name="min">-0.5</param>
        <param name="max">0.5</param>
      </command_interface>
      <state_interface name="position"/>
      <state_interface name="velocity"/>
      <state_interface name="effort"/>
    </joint>

    <!-- Left arm joint interface -->
    <joint name="torso_left_arm_joint">
      <command_interface name="position">
        <param name="min">-1.57</param>
        <param name="max">1.57</param>
      </command_interface>
      <state_interface name="position"/>
      <state_interface name="velocity"/>
      <state_interface name="effort"/>
    </joint>

    <!-- Right arm joint interface -->
    <joint name="torso_right_arm_joint">
      <command_interface name="position">
        <param name="min">-1.57</param>
        <param name="max">1.57</param>
      </command_interface>
      <state_interface name="position"/>
      <state_interface name="velocity"/>
      <state_interface name="effort"/>
    </joint>

    <!-- Left leg joint interface -->
    <joint name="torso_left_leg_joint">
      <command_interface name="position">
        <param name="min">-1.57</param>
        <param name="max">1.57</param>
      </command_interface>
      <state_interface name="position"/>
      <state_interface name="velocity"/>
      <state_interface name="effort"/>
    </joint>

    <!-- Right leg joint interface -->
    <joint name="torso_right_leg_joint">
      <command_interface name="position">
        <param name="min">-1.57</param>
        <param name="max">1.57</param>
      </command_interface>
      <state_interface name="position"/>
      <state_interface name="velocity"/>
      <state_interface name="effort"/>
    </joint>
  </ros2_control>
</robot>
```

## URDF Best Practices for Humanoid Robots

### 1. Proper Inertial Properties
Accurate inertial properties are crucial for stable simulation and control:
- Mass values should reflect actual robot components
- Center of mass should be correctly positioned
- Inertia tensor should be physically realistic

### 2. Joint Limits and Safety
Set appropriate joint limits to prevent damage:
- Limits should reflect physical constraints
- Include safety margins to prevent extreme positions
- Consider the robot's intended range of motion

### 3. Collision and Visual Separation
Separate collision and visual geometry appropriately:
- Visual geometry for display purposes
- Collision geometry for physics simulation (simpler shapes for performance)

### 4. Naming Conventions
Use consistent naming conventions:
- Descriptive names that indicate function
- Consistent prefixes/suffixes for related components
- Follow ROS naming conventions (underscores, lowercase)

## Integration with ROS 2 Control

The `<ros2_control>` element in URDF defines how the robot connects to the ROS 2 Control framework:

### Command Interfaces
Define how commands are sent to joints:
- `position`: Position control
- `velocity`: Velocity control
- `effort`: Torque/force control

### State Interfaces
Define how sensor data is published from joints:
- `position`: Current joint position
- `velocity`: Current joint velocity
- `effort`: Current joint effort/torque

## URDF Validation and Testing

### Validation Tools
- Use `check_urdf` to validate URDF syntax
- Use `urdf_to_graphiz` to visualize the kinematic tree
- Load in RViz to check visual appearance

### Simulation Testing
- Test in Gazebo to verify physics behavior
- Validate joint limits and ranges of motion
- Check that ros2_control interfaces work correctly

## Common URDF Issues and Solutions

### 1. Floating Point Precision
- Use appropriate precision for transformations
- Avoid extremely small or large values

### 2. Kinematic Loops
- Ensure the robot model is a tree structure
- Use mimic joints for coupled joints if needed

### 3. Mass Distribution
- Ensure the robot is stable in simulation
- Verify center of mass is in appropriate location

## Summary

URDF is fundamental to humanoid robot control, providing the geometric and kinematic model that controllers use to understand the robot's structure. A well-designed URDF with proper inertial properties, joint limits, and ros2_control interfaces enables stable simulation and effective control of humanoid robots. Understanding URDF structure and best practices is essential for creating humanoid robots that can be effectively controlled using ROS 2.