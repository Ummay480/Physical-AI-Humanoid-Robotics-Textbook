"""
Configuration management system for AI-Robot Brain
Provides centralized configuration handling for all modules
"""
import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path


class ConfigManager:
    """
    Centralized configuration management for the AI-Robot Brain system
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the configuration manager

        Args:
            config_path: Path to the main configuration file
        """
        self.config_path = config_path or self._get_default_config_path()
        self.config = self._load_config()

    def _get_default_config_path(self) -> str:
        """Get the default configuration file path"""
        # Look for config in common/config directory first
        default_paths = [
            "src/common/config/config.yaml",
            "src/common/config/config.yml",
            "config/config.yaml",
            "config.yaml",
            "config.yml"
        ]

        for path in default_paths:
            if os.path.exists(path):
                return path

        # If no config file exists, return the standard location
        return "src/common/config/config.yaml"

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if not os.path.exists(self.config_path):
            # Create a default configuration
            self._create_default_config()

        with open(self.config_path, 'r') as file:
            return yaml.safe_load(file) or {}

    def _create_default_config(self):
        """Create a default configuration file"""
        default_config = {
            "isaac_sim": {
                "simulation": {
                    "rendering": {
                        "resolution": [1920, 1080],
                        "framerate": 60,
                        "physics_update_rate": 60
                    },
                    "domain_randomization": {
                        "enabled": True,
                        "lighting_conditions": ["day", "night", "dusk"],
                        "material_properties": {
                            "friction_range": [0.1, 0.9],
                            "restitution_range": [0.0, 0.5]
                        }
                    }
                }
            },
            "isaac_ros": {
                "perception": {
                    "object_detection": {
                        "model_path": "models/object_detection.pt",
                        "confidence_threshold": 0.7,
                        "nms_threshold": 0.5,
                        "input_size": [640, 640]
                    },
                    "gpu_acceleration": {
                        "enabled": True,
                        "device": "cuda:0"
                    }
                },
                "vslam": {
                    "mapping": {
                        "resolution": 0.05,
                        "range": 10.0,
                        "update_rate": 30.0
                    },
                    "localization": {
                        "accuracy_threshold": 0.01,  # 1cm
                        "update_rate": 30.0
                    }
                }
            },
            "nav2_bipedal": {
                "path_planning": {
                    "algorithm": "nav2_navfn_planner/NavfnPlanner",
                    "planning_rate": 1.0,
                    "max_iterations": 1000,
                    "timeout": 1.0  # seconds
                },
                "bipedal_constraints": {
                    "max_step_height": 0.15,  # 15cm
                    "max_step_width": 0.4,   # 40cm
                    "max_slope_angle": 15.0, # degrees
                    "foot_separation": 0.3   # 30cm
                }
            },
            "common": {
                "logging": {
                    "level": "INFO",
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                    "file_path": "logs/ai_robot_brain.log"
                },
                "performance": {
                    "target_fps": 30,
                    "benchmark_enabled": True
                }
            }
        }

        # Ensure directory exists
        config_dir = os.path.dirname(self.config_path)
        os.makedirs(config_dir, exist_ok=True)

        # Write the default configuration
        with open(self.config_path, 'w') as file:
            yaml.dump(default_config, file, default_flow_style=False)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value using dot notation

        Args:
            key: Configuration key using dot notation (e.g., "isaac_ros.perception.confidence_threshold")
            default: Default value if key doesn't exist

        Returns:
            Configuration value or default
        """
        keys = key.split('.')
        value = self.config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def set(self, key: str, value: Any):
        """
        Set a configuration value using dot notation

        Args:
            key: Configuration key using dot notation
            value: Value to set
        """
        keys = key.split('.')
        config_ref = self.config

        # Navigate to the parent of the target key
        for k in keys[:-1]:
            if k not in config_ref or not isinstance(config_ref[k], dict):
                config_ref[k] = {}
            config_ref = config_ref[k]

        # Set the final value
        config_ref[keys[-1]] = value

    def update_config(self, new_config: Dict[str, Any]):
        """
        Update the configuration with new values

        Args:
            new_config: Dictionary with new configuration values
        """
        def merge_dict(base_dict: Dict, update_dict: Dict):
            for key, value in update_dict.items():
                if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                    merge_dict(base_dict[key], value)
                else:
                    base_dict[key] = value

        merge_dict(self.config, new_config)

        # Save the updated configuration
        self.save_config()

    def save_config(self):
        """Save the current configuration to file"""
        with open(self.config_path, 'w') as file:
            yaml.dump(self.config, file, default_flow_style=False)

    def reload_config(self):
        """Reload configuration from file"""
        self.config = self._load_config()


# Global configuration manager instance
_config_manager = None


def get_config_manager() -> ConfigManager:
    """
    Get the global configuration manager instance

    Returns:
        ConfigManager instance
    """
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


def get_config(key: str, default: Any = None) -> Any:
    """
    Get a configuration value using the global configuration manager

    Args:
        key: Configuration key using dot notation
        default: Default value if key doesn't exist

    Returns:
        Configuration value or default
    """
    return get_config_manager().get(key, default)


def set_config(key: str, value: Any):
    """
    Set a configuration value using the global configuration manager

    Args:
        key: Configuration key using dot notation
        value: Value to set
    """
    get_config_manager().set(key, value)