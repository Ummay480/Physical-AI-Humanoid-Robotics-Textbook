"""
Isaac Sim environment setup for perception training
Provides configuration and utilities for setting up simulation environments
"""
from typing import Dict, Any, List, Optional
import json
from pathlib import Path
from dataclasses import dataclass, asdict

from ...common.models.perception_models import SimulationEnvironment, SensorType


@dataclass
class IsaacSimConfig:
    """
    Configuration for Isaac Sim environment
    """
    rendering: Dict[str, Any]
    physics: Dict[str, Any]
    domain_randomization: Dict[str, Any]
    sensors: List[Dict[str, Any]]
    objects: List[Dict[str, Any]]


class IsaacSimEnvironmentManager:
    """
    Manager for Isaac Sim environments used in perception training
    """

    def __init__(self, base_path: str = "src/isaac_sim/simulation_envs"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def create_basic_environment(self, name: str, description: str = "") -> SimulationEnvironment:
        """
        Create a basic simulation environment for perception training

        Args:
            name: Name of the environment
            description: Description of the environment

        Returns:
            SimulationEnvironment entity
        """
        # Define basic physics properties
        physics_properties = {
            "gravity": -9.81,
            "friction": 0.5,
            "restitution": 0.1
        }

        # Define lighting conditions for domain randomization
        lighting_conditions = ["day", "night", "dusk", "dawn", "overcast"]

        # Define basic sensor configurations
        sensors = [
            {
                "name": "rgb_camera",
                "type": SensorType.RGB_CAMERA.value,
                "position": [0.0, 0.0, 0.5],
                "orientation": [0.0, 0.0, 0.0, 1.0],
                "resolution": [640, 480],
                "fov": 90.0
            },
            {
                "name": "depth_camera",
                "type": SensorType.DEPTH_CAMERA.value,
                "position": [0.0, 0.0, 0.5],
                "orientation": [0.0, 0.0, 0.0, 1.0],
                "resolution": [640, 480],
                "fov": 90.0
            }
        ]

        # Define basic objects for perception training
        objects = [
            {
                "name": "cube_1",
                "type": "cube",
                "position": [2.0, 0.0, 0.5],
                "size": [0.5, 0.5, 0.5],
                "color": [1.0, 0.0, 0.0],  # Red
                "material": "plastic"
            },
            {
                "name": "sphere_1",
                "type": "sphere",
                "position": [0.0, 2.0, 0.5],
                "radius": 0.3,
                "color": [0.0, 1.0, 0.0],  # Green
                "material": "metal"
            },
            {
                "name": "cylinder_1",
                "type": "cylinder",
                "position": [-2.0, 0.0, 0.5],
                "radius": 0.2,
                "height": 0.8,
                "color": [0.0, 0.0, 1.0],  # Blue
                "material": "wood"
            }
        ]

        # Define domain randomization parameters
        domain_randomization_params = {
            "lighting": {
                "enabled": True,
                "intensity_range": [0.5, 2.0],
                "color_temperature_range": [3000, 8000]
            },
            "materials": {
                "enabled": True,
                "friction_range": [0.1, 0.9],
                "restitution_range": [0.0, 0.5],
                "color_variation": True
            },
            "objects": {
                "enabled": True,
                "position_jitter": 0.1,
                "size_variation": 0.1
            }
        }

        # Create the SimulationEnvironment entity
        sim_env = SimulationEnvironment(
            id=None,  # Will be auto-generated
            name=name,
            description=description,
            physics_properties=physics_properties,
            lighting_conditions=lighting_conditions,
            objects=objects,
            sensors=sensors,
            domain_randomization_params=domain_randomization_params
        )

        # Save the environment configuration
        self.save_environment_config(sim_env)

        return sim_env

    def save_environment_config(self, sim_env: SimulationEnvironment):
        """
        Save the simulation environment configuration to a file

        Args:
            sim_env: SimulationEnvironment entity to save
        """
        env_dir = self.base_path / sim_env.name
        env_dir.mkdir(exist_ok=True)

        config_path = env_dir / "config.json"
        config_data = {
            "name": sim_env.name,
            "description": sim_env.description,
            "physics_properties": sim_env.physics_properties,
            "lighting_conditions": sim_env.lighting_conditions,
            "objects": sim_env.objects,
            "sensors": sim_env.sensors,
            "domain_randomization_params": sim_env.domain_randomization_params
        }

        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2)

    def load_environment_config(self, name: str) -> Optional[SimulationEnvironment]:
        """
        Load a simulation environment configuration from a file

        Args:
            name: Name of the environment to load

        Returns:
            SimulationEnvironment entity or None if not found
        """
        config_path = self.base_path / name / "config.json"

        if not config_path.exists():
            return None

        with open(config_path, 'r') as f:
            config_data = json.load(f)

        return SimulationEnvironment(
            id=None,  # Will be auto-generated
            name=config_data["name"],
            description=config_data["description"],
            physics_properties=config_data["physics_properties"],
            lighting_conditions=config_data["lighting_conditions"],
            objects=config_data["objects"],
            sensors=config_data["sensors"],
            domain_randomization_params=config_data["domain_randomization_params"]
        )

    def create_isaac_sim_config(self, sim_env: SimulationEnvironment) -> IsaacSimConfig:
        """
        Create an Isaac Sim configuration from a SimulationEnvironment

        Args:
            sim_env: SimulationEnvironment entity

        Returns:
            IsaacSimConfig configuration
        """
        rendering_config = {
            "resolution": [1920, 1080],
            "framerate": 60,
            "physics_update_rate": 60,
            "max_render_secs": 300  # 5 minutes max render time
        }

        physics_config = sim_env.physics_properties.copy()
        physics_config["solver_type"] = "TGS"  # Time-stepping Gauss-Seidel solver
        physics_config["use_gpu"] = True

        domain_rand_config = sim_env.domain_randomization_params.copy()

        return IsaacSimConfig(
            rendering=rendering_config,
            physics=physics_config,
            domain_randomization=domain_rand_config,
            sensors=sim_env.sensors,
            objects=sim_env.objects
        )

    def save_isaac_sim_config(self, config: IsaacSimConfig, name: str):
        """
        Save Isaac Sim configuration to a file

        Args:
            config: IsaacSimConfig to save
            name: Name of the environment
        """
        env_dir = self.base_path / name
        env_dir.mkdir(exist_ok=True)

        config_path = env_dir / "isaac_sim_config.json"
        config_dict = asdict(config)

        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)

    def create_perception_training_environment(self, name: str) -> SimulationEnvironment:
        """
        Create a specialized environment for perception training with multiple object types

        Args:
            name: Name of the environment

        Returns:
            SimulationEnvironment entity
        """
        # Start with a basic environment
        sim_env = self.create_basic_environment(
            name=name,
            description="Specialized environment for perception training with domain randomization"
        )

        # Add more diverse objects for training
        diverse_objects = [
            {
                "name": "training_cube_red",
                "type": "cube",
                "position": [1.5, 1.0, 0.5],
                "size": [0.3, 0.3, 0.3],
                "color": [0.8, 0.2, 0.2],  # Red
                "material": "plastic",
                "category": "training_object"
            },
            {
                "name": "training_sphere_green",
                "type": "sphere",
                "position": [-1.0, 1.5, 0.5],
                "radius": 0.25,
                "color": [0.2, 0.8, 0.2],  # Green
                "material": "metal",
                "category": "training_object"
            },
            {
                "name": "training_cylinder_blue",
                "type": "cylinder",
                "position": [0.0, -1.5, 0.5],
                "radius": 0.2,
                "height": 0.6,
                "color": [0.2, 0.2, 0.8],  # Blue
                "material": "wood",
                "category": "training_object"
            },
            {
                "name": "training_cone_yellow",
                "type": "cone",
                "position": [2.0, -1.0, 0.5],
                "radius": 0.25,
                "height": 0.7,
                "color": [0.8, 0.8, 0.2],  # Yellow
                "material": "plastic",
                "category": "training_object"
            },
            {
                "name": "training_capsule_purple",
                "type": "capsule",
                "position": [-2.0, -1.0, 0.5],
                "radius": 0.15,
                "height": 0.5,
                "color": [0.8, 0.2, 0.8],  # Purple
                "material": "metal",
                "category": "training_object"
            }
        ]

        # Add the diverse objects to the environment
        sim_env.objects.extend(diverse_objects)

        # Update the domain randomization to include more variations
        sim_env.domain_randomization_params["objects"]["shape_variation"] = True
        sim_env.domain_randomization_params["objects"]["count_variation"] = {
            "min": 3,
            "max": 8
        }

        # Save the updated environment
        self.save_environment_config(sim_env)

        # Create and save the Isaac Sim configuration
        isaac_config = self.create_isaac_sim_config(sim_env)
        self.save_isaac_sim_config(isaac_config, name)

        return sim_env


# Global Isaac Sim environment manager instance
_isaac_sim_env_manager = None


def get_isaac_sim_env_manager() -> IsaacSimEnvironmentManager:
    """
    Get the global Isaac Sim environment manager instance

    Returns:
        IsaacSimEnvironmentManager instance
    """
    global _isaac_sim_env_manager
    if _isaac_sim_env_manager is None:
        _isaac_sim_env_manager = IsaacSimEnvironmentManager()
    return _isaac_sim_env_manager