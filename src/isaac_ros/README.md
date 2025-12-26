# Isaac ROS Package

This package provides integration with NVIDIA Isaac ROS for hardware-accelerated perception and Visual SLAM (VSLAM) capabilities.

## Overview

The Isaac ROS package enables:
- Hardware-accelerated perception using GPU computing
- Visual SLAM for real-time mapping and localization
- Integration with Isaac Sim for simulation-based development
- ROS 2 interfaces for perception and navigation systems

## Installation

This package requires NVIDIA Isaac ROS to be installed separately. Follow the official NVIDIA Isaac ROS installation guide for your platform.

### Prerequisites
- ROS 2 Humble Hawksbill
- Isaac ROS packages (isaac_ros_common, isaac_ros_perception, etc.)
- NVIDIA GPU with RTX 30/40 series or equivalent
- CUDA 11.8+
- Compatible GPU drivers

## Usage

This package is part of the larger AI-Robot Brain system. For complete usage instructions, see the main documentation.

## Components

- `perception/` - Perception pipeline nodes and algorithms
- `vslam/` - Visual SLAM implementations and mapping systems
- `hardware_interface/` - GPU acceleration interfaces and CUDA kernels

## Integration

This package integrates with:
- Isaac Sim for simulation and synthetic data
- Navigation2 (Nav2) for path planning
- Module-2 perception systems for fused data