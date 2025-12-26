# Isaac Sim Package

This package provides integration with NVIDIA Isaac Sim for photorealistic simulation and synthetic data generation for the AI-Robot Brain module.

## Overview

The Isaac Sim package enables:
- Photorealistic simulation environments for training perception systems
- Synthetic data generation with domain randomization
- Integration with ROS 2 for perception pipeline testing

## Installation

This package requires NVIDIA Isaac Sim to be installed separately. Follow the official NVIDIA Isaac Sim installation guide for your platform.

### Prerequisites
- NVIDIA GPU with RTX 30/40 series or equivalent
- NVIDIA GPU drivers (525+)
- CUDA 11.8+
- Isaac Sim 2023.1+

## Usage

This package is part of the larger AI-Robot Brain system. For complete usage instructions, see the main documentation.

## Components

- `simulation_envs/` - Simulation environment definitions and assets
- `synthetic_data/` - Synthetic data generation tools
- `training/` - Training scripts for perception models

## Validation

Run the installation validation script to check basic Isaac Sim functionality:

```bash
python3 src/isaac_sim/test_installation.py
```

Note: Full Isaac Sim functionality requires running in a proper Isaac Sim environment.