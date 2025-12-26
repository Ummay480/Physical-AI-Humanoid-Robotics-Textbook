"""
Data models for the Perception Pipeline in AI-Robot Brain
Based on the data model specification in specs/001-ai-robot-brain/data-model.md
"""
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
import uuid
from datetime import datetime


class SensorType(Enum):
    """Enumeration of supported sensor types"""
    RGB_CAMERA = "rgb_camera"
    DEPTH_CAMERA = "depth_camera"
    LIDAR = "lidar"
    IMU = "imu"
    GPS = "gps"
    POSE = "pose"


@dataclass
class Coordinate:
    """Represents a 3D coordinate"""
    x: float
    y: float
    z: float


@dataclass
class Orientation:
    """Represents orientation as quaternion"""
    x: float
    y: float
    z: float
    w: float


@dataclass
class SensorData:
    """
    Data model for raw or processed data from various sensors used in the perception system
    Corresponds to the Sensor Data entity in the data model
    """
    id: str
    sensor_type: SensorType
    timestamp: datetime
    data_payload: bytes
    frame_id: str
    sensor_position: Coordinate
    sensor_orientation: Orientation

    def __post_init__(self):
        """Validate the sensor data after initialization"""
        if not self.id:
            self.id = str(uuid.uuid4())
        if not self.frame_id:
            raise ValueError("frame_id must be provided and follow ROS naming conventions")


@dataclass
class PerceptionPipeline:
    """
    Data model for the processing chain that transforms sensor data into meaningful environmental understanding
    Corresponds to the Perception Pipeline entity in the data model
    """
    id: str
    name: str
    sensor_inputs: List[SensorType]
    processing_modules: List[str]  # List of processing module names
    output_format: str
    gpu_acceleration_enabled: bool
    performance_metrics: Dict[str, Any]

    def __post_init__(self):
        """Validate the perception pipeline after initialization"""
        if not self.id:
            self.id = str(uuid.uuid4())
        if not self.name or len(self.name) > 100:
            raise ValueError("name must be 1-100 characters")
        if not self.processing_modules:
            raise ValueError("processing_modules must contain at least one module")
        if self.gpu_acceleration_enabled is None:
            self.gpu_acceleration_enabled = False
        if self.performance_metrics is None:
            self.performance_metrics = {}


@dataclass
class SimulationEnvironment:
    """
    Data model for virtual representation of physical world with objects, lighting, and physics properties
    Corresponds to the Simulation Environment entity in the data model
    """
    id: str
    name: str
    description: str
    physics_properties: Dict[str, float]  # gravity, friction, etc.
    lighting_conditions: List[str]
    objects: List[Dict[str, Any]]  # List of objects in the environment
    sensors: List[Dict[str, Any]]  # List of sensor configurations
    domain_randomization_params: Dict[str, Any]

    def __post_init__(self):
        """Validate the simulation environment after initialization"""
        if not self.id:
            self.id = str(uuid.uuid4())
        if not self.name or len(self.name) > 100:
            raise ValueError("name must be 1-100 characters")
        if not self.physics_properties or 'gravity' not in self.physics_properties:
            raise ValueError("physics_properties must include gravity value")
        if not self.objects:
            raise ValueError("objects must contain at least one object")
        if not self.sensors:
            raise ValueError("sensors must contain at least one sensor configuration")


@dataclass
class BipedalConstraints:
    """Data model for kinematic constraints specific to bipedal locomotion"""
    max_step_height: float = 0.15  # 15cm
    max_step_width: float = 0.4    # 40cm
    max_slope_angle: float = 15.0  # degrees
    foot_separation: float = 0.3   # 30cm


@dataclass
class NavigationMap:
    """
    Data model for spatial representation of environment including obstacles and path constraints
    Corresponds to the Navigation Map entity in the data model
    """
    id: str
    map_type: str  # "2D", "3D", "occupancy_grid", "topological"
    resolution: float  # meters per cell
    origin: Coordinate
    static_obstacles: List[Dict[str, Any]]
    dynamic_obstacles: List[Dict[str, Any]]
    traversable_areas: List[Dict[str, Any]]
    bipedal_constraints: BipedalConstraints
    map_data: Optional[bytes] = None

    def __post_init__(self):
        """Validate the navigation map after initialization"""
        if not self.id:
            self.id = str(uuid.uuid4())
        valid_types = ["2D", "3D", "occupancy_grid", "topological"]
        if self.map_type not in valid_types:
            raise ValueError(f"map_type must be one of: {valid_types}")
        if self.resolution <= 0:
            raise ValueError("resolution must be positive number")
        if not self.bipedal_constraints:
            self.bipedal_constraints = BipedalConstraints()


@dataclass
class PathPlanner:
    """
    Data model for algorithmic system that computes optimal routes
    Corresponds to the Path Planner entity in the data model
    """
    id: str
    algorithm_type: str  # A*, Dijkstra, RRT, etc.
    start_position: Coordinate
    goal_position: Coordinate
    planned_path: List[Coordinate]
    path_constraints: Dict[str, Any]
    execution_status: str  # "pending", "executing", "completed", "failed"
    bipedal_kinematic_model: Dict[str, Any]

    def __post_init__(self):
        """Validate the path planner after initialization"""
        if not self.id:
            self.id = str(uuid.uuid4())
        valid_statuses = ["pending", "executing", "completed", "failed"]
        if self.execution_status not in valid_statuses:
            raise ValueError(f"execution_status must be one of: {valid_statuses}")
        if not self.bipedal_kinematic_model:
            raise ValueError("bipedal_kinematic_model must be provided for humanoid robots")


@dataclass
class PathTrajectory:
    """
    Data model for detailed trajectory containing waypoints and execution parameters
    Corresponds to the Path Trajectory entity in the data model
    """
    id: str
    waypoints: List[Coordinate]
    execution_time: float  # Estimated time in seconds
    safety_margin: float
    bipedal_specific_params: Dict[str, Any]
    trajectory_status: str  # "pending", "executing", "completed", "failed"

    def __post_init__(self):
        """Validate the path trajectory after initialization"""
        if not self.id:
            self.id = str(uuid.uuid4())
        if len(self.waypoints) < 2:
            raise ValueError("waypoints must contain at least 2 points")
        if self.execution_time <= 0:
            raise ValueError("execution_time must be positive")
        if self.safety_margin < 0:
            raise ValueError("safety_margin must be non-negative")
        valid_statuses = ["pending", "executing", "completed", "failed"]
        if self.trajectory_status not in valid_statuses:
            raise ValueError(f"trajectory_status must be one of: {valid_statuses}")


@dataclass
class DetectionResult:
    """
    Data model for object detection results
    """
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # (x, y, width, height)

    def __post_init__(self):
        """Validate the detection result after initialization"""
        if self.confidence < 0 or self.confidence > 1:
            raise ValueError("confidence must be between 0 and 1")
        if len(self.bbox) != 4:
            raise ValueError("bbox must contain 4 values (x, y, width, height)")


@dataclass
class LocomotionController:
    """
    Data model for system that executes planned trajectories with bipedal kinematics
    Corresponds to the Locomotion Controller entity in the data model
    """
    id: str
    control_algorithm: str
    current_pose: Dict[str, float]  # Position and orientation
    balance_parameters: Dict[str, float]
    gait_pattern: str
    control_frequency: float

    def __post_init__(self):
        """Validate the locomotion controller after initialization"""
        if not self.id:
            self.id = str(uuid.uuid4())
        if not self.control_algorithm:
            raise ValueError("control_algorithm must be valid for bipedal locomotion")
        if not self.current_pose:
            raise ValueError("current_pose must have valid position and orientation")
        if self.control_frequency <= 0:
            raise ValueError("control_frequency must be positive")