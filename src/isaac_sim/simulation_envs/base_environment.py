"""
Base classes for Isaac Sim environments
Implements the BaseSimulationEnvironment abstract class as specified in T008
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List
from dataclasses import dataclass
from ...common.models.perception_models import SimulationEnvironment, Coordinate, SensorData


class BaseSimulationEnvironment(ABC):
    """Abstract base class for Isaac Sim environments."""

    @abstractmethod
    def setup(self) -> None:
        """Initialize the simulation environment."""
        pass

    @abstractmethod
    def reset(self) -> SimulationEnvironment:
        """Reset environment to initial state."""
        pass

    @abstractmethod
    def step(self, action: Any) -> SimulationEnvironment:
        """Execute one simulation step."""
        pass

    @abstractmethod
    def get_observations(self) -> List[SensorData]:
        """Get sensor observations from the environment."""
        pass

    @abstractmethod
    def get_environment_state(self) -> SimulationEnvironment:
        """Get the current state of the simulation environment."""
        pass

    @abstractmethod
    def configure_environment(self, config: Dict[str, Any]) -> None:
        """Configure the environment with the given configuration."""
        pass


@dataclass
class Action:
    """Represents an action that can be taken in the simulation environment."""
    position: Coordinate
    orientation: Coordinate  # Using Euler angles for orientation
    velocity: Coordinate
    angular_velocity: Coordinate
    timestamp: float