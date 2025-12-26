# Sensor Fusion Module Documentation

## Overview
The sensor fusion module provides a foundation for combining data from multiple sensors (LIDAR, IMU, etc.) to create a unified understanding of the environment. This implementation is currently at a validation level with basic timestamp-based synchronization.

## Components

### Base Fusion Engine
- `FusionEngine`: Abstract base class defining the fusion interface
- `SimpleTimestampFusionEngine`: Basic implementation focusing on synchronization validation

### Sensor Fusion Node
- Subscribes to sensor acquisition outputs (LIDAR, IMU processed data)
- Performs basic timestamp-based fusion
- Publishes fused state on `/perception/fused_state`
- Publishes fusion status on `/perception/fusion_status`

## Responsibilities

### Current Responsibilities
1. **Data Synchronization**: Validate that LIDAR and IMU data can be synchronized by timestamp
2. **Data Availability**: Verify that required sensor data is available
3. **Basic Fusion**: Create a simple fused representation without complex algorithms
4. **Status Reporting**: Publish fusion statistics and system status

### Current Limitations (Stub/Validation-Only)
1. **No Complex Algorithms**: No Kalman filters or advanced mathematical fusion techniques
2. **No Nervous System Integration**: Not integrated with the robot's nervous system
3. **Simple Validation-Level Fusion**: Basic timestamp-based synchronization only
4. **No Advanced Mapping**: No complex environmental mapping
5. **No Real-time Optimization**: No performance optimization for real-time applications

## Topics

### Subscriptions
- `/perception/lidar_3d/scan_processed` - Processed LIDAR data
- `/perception/imu_main/data_processed` - Processed IMU data

### Publications
- `/perception/fused_state` - Fused sensor state (JSON format)
- `/perception/fusion_status` - Fusion system status (JSON format)

## Future Enhancements

This foundation is designed to be extended with:
- Advanced fusion algorithms (Kalman filters, particle filters)
- Complex environmental mapping
- Integration with the nervous system
- Real-time performance optimization
- Additional sensor types (cameras, ultrasonic, etc.)