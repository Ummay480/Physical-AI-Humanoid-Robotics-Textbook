"""
Sensor configuration handler for managing sensor configurations and calibration.
"""
import json
import yaml
from typing import Dict, Any, Optional, List
from pathlib import Path
import numpy as np
from .data_types import SensorConfig, SensorType, SensorConfigState
from .utils import get_current_timestamp


class SensorConfigHandler:
    """
    Handles sensor configuration management including loading, saving, and calibration.
    """

    def __init__(self, config_dir: str = "config"):
        """
        Initialize the sensor configuration handler.

        Args:
            config_dir: Directory where sensor configuration files are stored
        """
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        self.sensor_configs: Dict[str, SensorConfig] = {}

    def load_sensor_config(self, sensor_id: str) -> Optional[SensorConfig]:
        """
        Load a sensor configuration from file.

        Args:
            sensor_id: ID of the sensor to load configuration for

        Returns:
            SensorConfig: Loaded sensor configuration or None if not found
        """
        config_file = self.config_dir / f"{sensor_id}.yaml"

        if not config_file.exists():
            return None

        try:
            with open(config_file, 'r') as f:
                config_data = yaml.safe_load(f)

            # Create SensorConfig object from loaded data
            config = SensorConfig(
                sensor_id=config_data['sensor_id'],
                sensor_type=SensorType(config_data['sensor_type']),
                topic=config_data.get('topic', f'/{config_data["sensor_id"]}/data'),
                calibration_file=config_data.get('calibration_file', f'calibration/{config_data["sensor_id"]}.yaml'),
                enabled=config_data.get('enabled', True),
                processing_frequency=config_data.get('processing_frequency', 10.0),
                parameters=config_data.get('parameters'),
                state=SensorConfigState(config_data.get('state', 'UNCONFIGURED'))
            )

            self.sensor_configs[sensor_id] = config
            return config

        except Exception as e:
            print(f"Error loading sensor config for {sensor_id}: {e}")
            return None

    def save_sensor_config(self, sensor_config: SensorConfig) -> bool:
        """
        Save a sensor configuration to file.

        Args:
            sensor_config: Sensor configuration to save

        Returns:
            bool: True if saved successfully, False otherwise
        """
        try:
            config_data = {
                'sensor_id': sensor_config.sensor_id,
                'sensor_type': sensor_config.sensor_type.value,
                'topic': sensor_config.topic,
                'calibration_file': sensor_config.calibration_file,
                'enabled': sensor_config.enabled,
                'processing_frequency': sensor_config.processing_frequency,
                'parameters': sensor_config.parameters,
                'state': sensor_config.state.value
            }

            config_file = self.config_dir / f"{sensor_config.sensor_id}.yaml"
            with open(config_file, 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False)

            self.sensor_configs[sensor_config.sensor_id] = sensor_config
            return True

        except Exception as e:
            print(f"Error saving sensor config for {sensor_config.sensor_id}: {e}")
            return False

    def get_sensor_config(self, sensor_id: str) -> Optional[SensorConfig]:
        """
        Get a sensor configuration from memory or load from file.

        Args:
            sensor_id: ID of the sensor to get configuration for

        Returns:
            SensorConfig: Sensor configuration or None if not found
        """
        if sensor_id in self.sensor_configs:
            return self.sensor_configs[sensor_id]

        return self.load_sensor_config(sensor_id)

    def update_sensor_config(self, sensor_config: SensorConfig) -> bool:
        """
        Update an existing sensor configuration.

        Args:
            sensor_config: Updated sensor configuration

        Returns:
            bool: True if updated successfully, False otherwise
        """
        # Validate the sensor configuration
        if not self.validate_sensor_config(sensor_config):
            return False

        # Save to file
        if not self.save_sensor_config(sensor_config):
            return False

        # Update in memory
        self.sensor_configs[sensor_config.sensor_id] = sensor_config
        return True

    def validate_sensor_config(self, sensor_config: SensorConfig) -> bool:
        """
        Validate a sensor configuration.

        Args:
            sensor_config: Sensor configuration to validate

        Returns:
            bool: True if valid, False otherwise
        """
        # Check required fields
        if not sensor_config.sensor_id:
            print("Sensor config validation failed: sensor_id is required")
            return False

        if sensor_config.sensor_type not in SensorType:
            print(f"Sensor config validation failed: invalid sensor type {sensor_config.sensor_type}")
            return False

        if not sensor_config.topic:
            print("Sensor config validation failed: topic is required")
            return False

        if not sensor_config.calibration_file:
            print("Sensor config validation failed: calibration_file is required")
            return False

        return True

    def create_default_config(self, sensor_id: str, sensor_type: SensorType) -> SensorConfig:
        """
        Create a default sensor configuration.

        Args:
            sensor_id: ID of the sensor
            sensor_type: Type of the sensor

        Returns:
            SensorConfig: Default sensor configuration
        """
        default_config = SensorConfig(
            sensor_id=sensor_id,
            sensor_type=sensor_type,
            topic=f'/{sensor_id}/data',
            calibration_file=f'calibration/{sensor_id}.yaml',
            enabled=True,
            processing_frequency=10.0,
            parameters={},
            state=SensorConfigState.UNCONFIGURED
        )

        return default_config

    def list_sensor_configs(self) -> List[str]:
        """
        List all available sensor configuration IDs.

        Returns:
            List[str]: List of sensor IDs with available configurations
        """
        config_files = list(self.config_dir.glob("*.yaml"))
        sensor_ids = [f.stem for f in config_files]
        return sensor_ids

    def set_calibration_data(self, sensor_id: str, calibration_file_path: str) -> bool:
        """
        Update the calibration file path for a sensor.

        Args:
            sensor_id: ID of the sensor
            calibration_file_path: Path to the calibration file

        Returns:
            bool: True if updated successfully, False otherwise
        """
        config = self.get_sensor_config(sensor_id)
        if not config:
            # Create a default config if one doesn't exist
            config = self.create_default_config(sensor_id, SensorType.OTHER)

        config.calibration_file = calibration_file_path
        config.state = SensorConfigState.CALIBRATED

        return self.update_sensor_config(config)

    def get_calibration_file_path(self, sensor_id: str) -> Optional[str]:
        """
        Get the calibration file path for a sensor.

        Args:
            sensor_id: ID of the sensor

        Returns:
            str: Path to calibration file or None if not found
        """
        config = self.get_sensor_config(sensor_id)
        if not config:
            return None

        return config.calibration_file

    def get_default_operational_params(self, sensor_type: SensorType) -> Dict[str, Any]:
        """
        Get default operational parameters for a sensor type.

        Args:
            sensor_type: Type of sensor

        Returns:
            Dict[str, Any]: Default operational parameters
        """
        defaults = {
            SensorType.CAMERA: {
                "frequency": 30.0,  # Hz
                "resolution": [640, 480],
                "format": "bgr8"
            },
            SensorType.LIDAR: {
                "frequency": 10.0,  # Hz
                "range_min": 0.1,   # meters
                "range_max": 30.0,  # meters
                "angle_min": -2.356,  # radians (-135 degrees)
                "angle_max": 2.356,   # radians (135 degrees)
                "angle_increment": 0.0043,  # radians
            },
            SensorType.IMU: {
                "frequency": 100.0,  # Hz
                "linear_acceleration_stddev": 0.017,
                "angular_velocity_stddev": 0.001,
                "orientation_stddev": 6.66e-05
            },
            SensorType.ULTRASONIC: {
                "frequency": 20.0,  # Hz
                "range_min": 0.02,  # meters
                "range_max": 4.0,   # meters
                "cone_angle": 15.0,  # degrees
            },
            SensorType.GPS: {
                "frequency": 1.0,  # Hz
                "horizontal_accuracy": 2.0,  # meters
                "vertical_accuracy": 4.0,    # meters
            },
            SensorType.SONAR: {
                "frequency": 10.0,  # Hz
                "range_min": 0.02,  # meters
                "range_max": 4.0,   # meters
            },
            SensorType.FISHEYE_CAMERA: {
                "frequency": 30.0,  # Hz
                "resolution": [1280, 720],
                "format": "bgr8",
                "fov": 180.0,  # degrees
            },
            SensorType.THERMAL_CAMERA: {
                "frequency": 30.0,  # Hz
                "resolution": [640, 512],
                "format": "mono16",
                "temperature_range": [-10, 400],  # Celsius
            }
        }
        return defaults.get(sensor_type, {})

    def synchronize_sensor_configs(self, reference_config: SensorConfig) -> bool:
        """
        Synchronize operational parameters across all sensor configs based on a reference.

        Args:
            reference_config: Reference configuration to sync with

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            for sensor_id, config in self.sensor_configs.items():
                if config.sensor_type == reference_config.sensor_type:
                    # Update parameters from reference
                    if config.parameters is None:
                        config.parameters = {}
                    if reference_config.parameters:
                        config.parameters.update(reference_config.parameters)
                    self.update_sensor_config(config)
            return True
        except Exception as e:
            print(f"Error synchronizing sensor configs: {e}")
            return False

    def validate_coordinate_frame(self, frame_id: str) -> bool:
        """
        Validate that a coordinate frame follows ROS naming conventions.

        Args:
            frame_id: Coordinate frame identifier

        Returns:
            bool: True if frame ID is valid, False otherwise
        """
        # Basic validation: non-empty, no spaces, starts with letter or underscore
        if not frame_id or ' ' in frame_id:
            return False

        # Should start with letter or underscore
        first_char = frame_id[0]
        if not (first_char.isalpha() or first_char == '_'):
            return False

        # All characters should be alphanumeric, underscore, or hyphen
        for char in frame_id:
            if not (char.isalnum() or char in ['_', '-']):
                return False

        return True