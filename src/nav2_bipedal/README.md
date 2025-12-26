# Nav2 Bipedal Package

This package provides Navigation2 (Nav2) integration for bipedal humanoid path planning with specialized kinematic constraints.

## Overview

The Nav2 Bipedal package enables:
- Path planning optimized for bipedal locomotion
- Kinematic constraints for humanoid robot movement
- Integration with Isaac ROS VSLAM for real-time navigation
- Dynamic obstacle avoidance for bipedal robots

## Installation

This package requires Navigation2 for ROS 2 Humble to be installed separately. Follow the official Navigation2 installation guide.

### Prerequisites
- ROS 2 Humble Hawksbill
- Navigation2 packages (navigation2, nav2-bringup)
- TF2 for coordinate transformations
- Isaac ROS VSLAM (for map integration)

## Usage

This package is part of the larger AI-Robot Brain system. For complete usage instructions, see the main documentation.

## Components

- `path_planning/` - Bipedal-specific path planning algorithms
- `locomotion/` - Bipedal locomotion controllers and gait patterns
- `constraints/` - Kinematic constraint implementations for bipedal movement

## Integration

This package integrates with:
- Isaac ROS VSLAM for real-time map updates
- Isaac Sim for simulation-based navigation testing
- Module-2 perception systems for obstacle detection