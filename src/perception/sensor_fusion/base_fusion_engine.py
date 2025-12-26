"""
Base FusionEngine interface for sensor fusion operations.

This defines the interface for sensor fusion without complex algorithms yet.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
import numpy as np

from ..common.data_types import SensorData, FusedData


class FusionEngine(ABC):
    """
    Abstract base class for sensor fusion engines.

    This interface defines the basic operations for fusing sensor data
    without implementing complex fusion algorithms.
    """

    @abstractmethod
    def add_sensor_data(self, sensor_id: str, sensor_data: SensorData):
        """
        Add sensor data to the fusion engine.

        Args:
            sensor_id: Unique identifier for the sensor
            sensor_data: Sensor data to add
        """
        pass

    @abstractmethod
    def get_synchronized_data(self, timestamp: float, tolerance: float = 0.1) -> Dict[str, SensorData]:
        """
        Get sensor data synchronized to a specific timestamp.

        Args:
            timestamp: Target timestamp for synchronization
            tolerance: Time tolerance for synchronization

        Returns:
            Dictionary mapping sensor_id to synchronized sensor data
        """
        pass

    @abstractmethod
    def fuse_data(self, sensor_data_dict: Dict[str, SensorData]) -> Optional[FusedData]:
        """
        Fuse synchronized sensor data into a unified representation.

        Args:
            sensor_data_dict: Dictionary mapping sensor_id to sensor data

        Returns:
            FusedData object or None if fusion fails
        """
        pass

    @abstractmethod
    def validate_fusion_input(self, sensor_data_dict: Dict[str, SensorData]) -> bool:
        """
        Validate that the input data is suitable for fusion.

        Args:
            sensor_data_dict: Dictionary mapping sensor_id to sensor data

        Returns:
            True if input is valid, False otherwise
        """
        pass

    @abstractmethod
    def get_fusion_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about fusion operations.

        Returns:
            Dictionary containing fusion statistics
        """
        pass


class SimpleTimestampFusionEngine(FusionEngine):
    """
    Simple timestamp-based fusion engine for basic sensor fusion.

    This implementation focuses on validating synchronization and data availability
    without complex fusion algorithms.
    """

    def __init__(self):
        """Initialize the simple fusion engine."""
        self.sensor_buffers: Dict[str, List[SensorData]] = {}
        self.buffer_max_size = 100

        # Fusion statistics
        self.stats = {
            'total_fusion_attempts': 0,
            'successful_fusions': 0,
            'failed_fusions': 0,
            'synchronization_failures': 0,
            'data_availability_failures': 0
        }

    def add_sensor_data(self, sensor_id: str, sensor_data: SensorData):
        """
        Add sensor data to the fusion buffer.

        Args:
            sensor_id: Unique identifier for the sensor
            sensor_data: Sensor data to add
        """
        if sensor_id not in self.sensor_buffers:
            self.sensor_buffers[sensor_id] = []

        # Add to buffer
        self.sensor_buffers[sensor_id].append(sensor_data)

        # Limit buffer size
        if len(self.sensor_buffers[sensor_id]) > self.buffer_max_size:
            self.sensor_buffers[sensor_id].pop(0)

    def get_synchronized_data(self, timestamp: float, tolerance: float = 0.1) -> Dict[str, SensorData]:
        """
        Get sensor data synchronized to a specific timestamp.

        Args:
            timestamp: Target timestamp for synchronization
            tolerance: Time tolerance for synchronization

        Returns:
            Dictionary mapping sensor_id to synchronized sensor data
        """
        synchronized = {}

        for sensor_id, buffer in self.sensor_buffers.items():
            # Find closest data point within tolerance
            closest_data = None
            min_time_diff = float('inf')

            for data in buffer:
                time_diff = abs(data.timestamp - timestamp)
                if time_diff < min_time_diff and time_diff <= tolerance:
                    min_time_diff = time_diff
                    closest_data = data

            if closest_data is not None:
                synchronized[sensor_id] = closest_data

        return synchronized

    def fuse_data(self, sensor_data_dict: Dict[str, SensorData]) -> Optional[FusedData]:
        """
        Fuse synchronized sensor data into a unified representation.

        Args:
            sensor_data_dict: Dictionary mapping sensor_id to sensor data

        Returns:
            FusedData object or None if fusion fails
        """
        self.stats['total_fusion_attempts'] += 1

        # Validate input
        if not self.validate_fusion_input(sensor_data_dict):
            self.stats['failed_fusions'] += 1
            return None

        # Check for required sensors (LIDAR + IMU)
        has_lidar = any(data.sensor_type.value == 'lidar' for data in sensor_data_dict.values())
        has_imu = any(data.sensor_type.value == 'imu' for data in sensor_data_dict.values())

        if not (has_lidar and has_imu):
            self.stats['data_availability_failures'] += 1
            self.stats['failed_fusions'] += 1
            return None

        # Extract timestamps for synchronization verification
        timestamps = [data.timestamp for data in sensor_data_dict.values()]
        timestamp_diff = max(timestamps) - min(timestamps) if timestamps else 0

        # Verify synchronization (timestamps should be close)
        sync_tolerance = 0.1  # 100ms tolerance
        if timestamp_diff > sync_tolerance:
            self.stats['synchronization_failures'] += 1
            self.stats['failed_fusions'] += 1
            return None

        # Create a simple fused data representation
        # This is a validation-level implementation, not a complex algorithm
        fused_payload = {
            'sensor_ids': list(sensor_data_dict.keys()),
            'sensor_types': [data.sensor_type.value for data in sensor_data_dict.values()],
            'timestamps': timestamps,
            'synchronization_verified': True,
            'data_available': True,
            'fusion_method': 'timestamp_based_validation'
        }

        # Calculate a basic confidence score based on number of sensors
        confidence_score = min(1.0, len(sensor_data_dict) * 0.5)

        # Create fused data object
        fused_data = FusedData(
            id=f"fused_{int(timestamps[0] * 1e9)}",  # Use first timestamp as base
            timestamp=timestamps[0],
            fused_payload=fused_payload,
            source_sensor_ids=list(sensor_data_dict.keys()),
            confidence_score=confidence_score,
            coordinate_frame='base_link'  # Default coordinate frame
        )

        self.stats['successful_fusions'] += 1
        return fused_data

    def validate_fusion_input(self, sensor_data_dict: Dict[str, SensorData]) -> bool:
        """
        Validate that the input data is suitable for fusion.

        Args:
            sensor_data_dict: Dictionary mapping sensor_id to sensor data

        Returns:
            True if input is valid, False otherwise
        """
        if not sensor_data_dict:
            return False

        # Check that all data has valid timestamps
        for sensor_id, data in sensor_data_dict.items():
            if data.timestamp is None or data.timestamp <= 0:
                return False

        return True

    def get_fusion_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about fusion operations.

        Returns:
            Dictionary containing fusion statistics
        """
        stats = self.stats.copy()

        # Add buffer statistics
        stats['buffer_sizes'] = {
            sensor_id: len(buffer)
            for sensor_id, buffer in self.sensor_buffers.items()
        }

        # Add success rate
        if stats['total_fusion_attempts'] > 0:
            stats['success_rate'] = stats['successful_fusions'] / stats['total_fusion_attempts']
        else:
            stats['success_rate'] = 0.0

        return stats

    def clear_buffers(self):
        """Clear all sensor buffers."""
        self.sensor_buffers.clear()

    def reset_statistics(self):
        """Reset fusion statistics."""
        self.stats = {
            'total_fusion_attempts': 0,
            'successful_fusions': 0,
            'failed_fusions': 0,
            'synchronization_failures': 0,
            'data_availability_failures': 0
        }