"""
Sensor Data entity handlers for the AI-Robot Brain
Provides processing and management functions for sensor data entities
"""
from typing import Dict, Any, List, Optional
import numpy as np
from datetime import datetime
import json
import struct
from .perception_models import SensorData, SensorType, Coordinate, Orientation


class SensorDataHandler:
    """
    Handler for Sensor Data entities with processing and management capabilities
    """

    def __init__(self):
        self._sensor_buffer = {}
        self._buffer_size = 100  # Maximum number of sensor readings to keep in buffer

    def create_sensor_data(
        self,
        sensor_type: SensorType,
        data_payload: bytes,
        frame_id: str,
        position: Coordinate,
        orientation: Orientation,
        timestamp: Optional[datetime] = None
    ) -> SensorData:
        """
        Create a new SensorData entity with validation

        Args:
            sensor_type: Type of sensor
            data_payload: Raw sensor data as bytes
            frame_id: Coordinate frame identifier
            position: Position of the sensor
            orientation: Orientation of the sensor
            timestamp: Timestamp of the sensor reading (defaults to now)

        Returns:
            SensorData entity
        """
        if timestamp is None:
            timestamp = datetime.now()

        sensor_data = SensorData(
            id=None,  # Will be auto-generated
            sensor_type=sensor_type,
            timestamp=timestamp,
            data_payload=data_payload,
            frame_id=frame_id,
            sensor_position=position,
            sensor_orientation=orientation
        )

        # Add to buffer for this sensor type
        if sensor_type.value not in self._sensor_buffer:
            self._sensor_buffer[sensor_type.value] = []

        self._sensor_buffer[sensor_type.value].append(sensor_data)

        # Maintain buffer size limit
        if len(self._sensor_buffer[sensor_type.value]) > self._buffer_size:
            self._sensor_buffer[sensor_type.value] = self._sensor_buffer[sensor_type.value][-self._buffer_size:]

        return sensor_data

    def process_rgb_data(self, sensor_data: SensorData) -> Dict[str, Any]:
        """
        Process RGB camera sensor data

        Args:
            sensor_data: SensorData entity with RGB camera data

        Returns:
            Dictionary with processed RGB data information
        """
        if sensor_data.sensor_type != SensorType.RGB_CAMERA:
            raise ValueError(f"Expected RGB camera data, got {sensor_data.sensor_type}")

        # For now, we'll just return basic information about the RGB data
        # In a real implementation, this would decode the image and extract features
        width, height, channels = struct.unpack('III', sensor_data.data_payload[:12])

        return {
            'width': width,
            'height': height,
            'channels': channels,
            'timestamp': sensor_data.timestamp,
            'frame_id': sensor_data.frame_id
        }

    def process_depth_data(self, sensor_data: SensorData) -> Dict[str, Any]:
        """
        Process depth camera sensor data

        Args:
            sensor_data: SensorData entity with depth camera data

        Returns:
            Dictionary with processed depth data information
        """
        if sensor_data.sensor_type != SensorType.DEPTH_CAMERA:
            raise ValueError(f"Expected depth camera data, got {sensor_data.sensor_type}")

        # For now, we'll just return basic information about the depth data
        # In a real implementation, this would decode the depth map
        width, height = struct.unpack('II', sensor_data.data_payload[:8])

        return {
            'width': width,
            'height': height,
            'min_depth': 0.1,  # Placeholder
            'max_depth': 10.0,  # Placeholder
            'timestamp': sensor_data.timestamp,
            'frame_id': sensor_data.frame_id
        }

    def process_lidar_data(self, sensor_data: SensorData) -> Dict[str, Any]:
        """
        Process LIDAR sensor data

        Args:
            sensor_data: SensorData entity with LIDAR data

        Returns:
            Dictionary with processed LIDAR data information
        """
        if sensor_data.sensor_type != SensorType.LIDAR:
            raise ValueError(f"Expected LIDAR data, got {sensor_data.sensor_type}")

        # For now, we'll just return basic information about the LIDAR data
        # In a real implementation, this would decode the point cloud
        num_points = struct.unpack('I', sensor_data.data_payload[:4])[0]

        return {
            'num_points': num_points,
            'timestamp': sensor_data.timestamp,
            'frame_id': sensor_data.frame_id
        }

    def process_imu_data(self, sensor_data: SensorData) -> Dict[str, Any]:
        """
        Process IMU sensor data

        Args:
            sensor_data: SensorData entity with IMU data

        Returns:
            Dictionary with processed IMU data information
        """
        if sensor_data.sensor_type != SensorType.IMU:
            raise ValueError(f"Expected IMU data, got {sensor_data.sensor_type}")

        # Unpack IMU data: accelerometer (x, y, z), gyroscope (x, y, z), magnetometer (x, y, z)
        ax, ay, az, gx, gy, gz, mx, my, mz = struct.unpack('fffffffff', sensor_data.data_payload[:36])

        return {
            'acceleration': {'x': ax, 'y': ay, 'z': az},
            'angular_velocity': {'x': gx, 'y': gy, 'z': gz},
            'magnetic_field': {'x': mx, 'y': my, 'z': mz},
            'timestamp': sensor_data.timestamp,
            'frame_id': sensor_data.frame_id
        }

    def process_pose_data(self, sensor_data: SensorData) -> Dict[str, Any]:
        """
        Process pose sensor data

        Args:
            sensor_data: SensorData entity with pose data

        Returns:
            Dictionary with processed pose data information
        """
        if sensor_data.sensor_type != SensorType.POSE:
            raise ValueError(f"Expected pose data, got {sensor_data.sensor_type}")

        # Unpack pose data: position (x, y, z) and orientation (x, y, z, w as quaternion)
        px, py, pz, ox, oy, oz, ow = struct.unpack('ddddddd', sensor_data.data_payload[:56])

        return {
            'position': {'x': px, 'y': py, 'z': pz},
            'orientation': {'x': ox, 'y': oy, 'z': oz, 'w': ow},
            'timestamp': sensor_data.timestamp,
            'frame_id': sensor_data.frame_id
        }

    def process_sensor_data(self, sensor_data: SensorData) -> Dict[str, Any]:
        """
        Process sensor data based on its type

        Args:
            sensor_data: SensorData entity to process

        Returns:
            Dictionary with processed sensor data information
        """
        processor_map = {
            SensorType.RGB_CAMERA: self.process_rgb_data,
            SensorType.DEPTH_CAMERA: self.process_depth_data,
            SensorType.LIDAR: self.process_lidar_data,
            SensorType.IMU: self.process_imu_data,
            SensorType.POSE: self.process_pose_data,
        }

        processor = processor_map.get(sensor_data.sensor_type)
        if processor:
            return processor(sensor_data)
        else:
            raise ValueError(f"Unsupported sensor type: {sensor_data.sensor_type}")

    def get_latest_sensor_data(self, sensor_type: SensorType) -> Optional[SensorData]:
        """
        Get the most recent sensor data for a specific sensor type

        Args:
            sensor_type: Type of sensor to get data for

        Returns:
            Latest SensorData entity or None if no data exists
        """
        if sensor_type.value in self._sensor_buffer and self._sensor_buffer[sensor_type.value]:
            return self._sensor_buffer[sensor_type.value][-1]
        return None

    def get_sensor_data_history(self, sensor_type: SensorType, count: int = 10) -> List[SensorData]:
        """
        Get recent sensor data for a specific sensor type

        Args:
            sensor_type: Type of sensor to get data for
            count: Number of recent readings to return

        Returns:
            List of SensorData entities
        """
        if sensor_type.value in self._sensor_buffer:
            return self._sensor_buffer[sensor_type.value][-count:]
        return []

    def serialize_sensor_data(self, sensor_data: SensorData) -> bytes:
        """
        Serialize SensorData entity to bytes for storage or transmission

        Args:
            sensor_data: SensorData entity to serialize

        Returns:
            Serialized bytes
        """
        data_dict = {
            'id': sensor_data.id,
            'sensor_type': sensor_data.sensor_type.value,
            'timestamp': sensor_data.timestamp.isoformat(),
            'data_payload': sensor_data.data_payload.hex(),  # Convert bytes to hex string
            'frame_id': sensor_data.frame_id,
            'sensor_position': {
                'x': sensor_data.sensor_position.x,
                'y': sensor_data.sensor_position.y,
                'z': sensor_data.sensor_position.z
            },
            'sensor_orientation': {
                'x': sensor_data.sensor_orientation.x,
                'y': sensor_data.sensor_orientation.y,
                'z': sensor_data.sensor_orientation.z,
                'w': sensor_data.sensor_orientation.w
            }
        }

        return json.dumps(data_dict).encode('utf-8')

    def deserialize_sensor_data(self, data: bytes) -> SensorData:
        """
        Deserialize bytes to SensorData entity

        Args:
            data: Serialized SensorData as bytes

        Returns:
            SensorData entity
        """
        data_dict = json.loads(data.decode('utf-8'))

        return SensorData(
            id=data_dict['id'],
            sensor_type=SensorType(data_dict['sensor_type']),
            timestamp=datetime.fromisoformat(data_dict['timestamp']),
            data_payload=bytes.fromhex(data_dict['data_payload']),
            frame_id=data_dict['frame_id'],
            sensor_position=Coordinate(**data_dict['sensor_position']),
            sensor_orientation=Orientation(**data_dict['sensor_orientation'])
        )

    def validate_sensor_data(self, sensor_data: SensorData) -> bool:
        """
        Validate SensorData entity according to the data model constraints

        Args:
            sensor_data: SensorData entity to validate

        Returns:
            True if valid, False otherwise
        """
        try:
            # Check if frame_id follows ROS naming conventions (basic check)
            if not sensor_data.frame_id or not isinstance(sensor_data.frame_id, str):
                return False

            # Check if timestamp is valid
            if not isinstance(sensor_data.timestamp, datetime):
                return False

            # Check if sensor position and orientation are valid
            pos = sensor_data.sensor_position
            if not all(isinstance(v, (int, float)) for v in [pos.x, pos.y, pos.z]):
                return False

            orient = sensor_data.sensor_orientation
            if not all(isinstance(v, (int, float)) for v in [orient.x, orient.y, orient.z, orient.w]):
                return False

            return True
        except:
            return False


# Global sensor data handler instance
_sensor_data_handler = None


def get_sensor_data_handler() -> SensorDataHandler:
    """
    Get the global sensor data handler instance

    Returns:
        SensorDataHandler instance
    """
    global _sensor_data_handler
    if _sensor_data_handler is None:
        _sensor_data_handler = SensorDataHandler()
    return _sensor_data_handler