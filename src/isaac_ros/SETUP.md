# Isaac ROS Setup Guide

This guide describes the process for installing and validating Isaac ROS packages as part of T002: Install Isaac ROS Packages and Dependencies.

## Overview

Isaac ROS provides hardware-accelerated perception and Visual SLAM capabilities for the AI-Robot Brain module. This setup includes:

- Isaac ROS common packages
- Isaac ROS perception packages
- Isaac ROS visual SLAM packages
- Required dependencies and configurations

## Installation Process

The installation involves these main steps:

1. Install ROS 2 Humble Hawksbill
2. Install Isaac ROS packages
3. Create Isaac ROS workspace (`~/isaac_ros_ws`)
4. Install dependencies (cv_bridge, image_transport, etc.)
5. Verify workspace builds successfully
6. Validate Isaac ROS nodes can be launched
7. Verify GPU acceleration

## Setup Script

A setup script is provided to automate the installation process:

```bash
# Make sure ROS 2 Humble is sourced first
source /opt/ros/humble/setup.bash

# Run the setup script
./scripts/setup_isaac_ros.sh
```

## Validation

After installation, validate the setup:

```bash
# Test Isaac ROS installation
python3 src/isaac_ros/test_installation.py
```

## Launch Files

Example launch files are provided to test Isaac ROS functionality:

```bash
# Example Isaac ROS Visual SLAM launch
ros2 launch isaac_ros_visual_slam.launch.py
```

## Dependencies

This installation requires:
- Ubuntu 22.04 LTS
- ROS 2 Humble Hawksbill
- NVIDIA GPU with CUDA support
- Isaac Sim (for simulation integration)