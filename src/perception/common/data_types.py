"""
Common data types for the perception system based on the data model.
"""
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from enum import Enum
import numpy as np


class SensorType(Enum):
    """Enumeration of sensor types supported by the system."""
    CAMERA = "camera"
    LIDAR = "lidar"
    IMU = "imu"
    ULTRASONIC = "ultrasonic"
    GPS = "gps"
    SONAR = "sonar"
    FISHEYE_CAMERA = "fisheye_camera"
    THERMAL_CAMERA = "thermal_camera"
    OTHER = "other"


class ObjectType(Enum):
    """Enumeration of detectable object types for humanoid robotics."""
    HUMAN = "human"
    CHAIR = "chair"
    TABLE = "table"
    DOOR = "door"
    STAIR = "stair"
    OBSTACLE = "obstacle"


class SensorDataState(Enum):
    """Enumeration of sensor data states."""
    ACQUIRING = "ACQUIRING"
    ACQUIRED = "ACQUIRED"
    PROCESSING = "PROCESSING"
    PROCESSED = "PROCESSED"
    FAILED = "FAILED"


class ProcessingState(Enum):
    """Enumeration of processing pipeline states."""
    QUEUED = "QUEUED"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    FUSED = "FUSED"


class SensorConfigState(Enum):
    """Enumeration of sensor configuration states."""
    UNCONFIGURED = "UNCONFIGURED"
    CALIBRATING = "CALIBRATING"
    CONFIGURED = "CONFIGURED"
    CALIBRATED = "CALIBRATED"
    ERROR = "ERROR"


@dataclass
class SensorConfig:
    """
    Represents configuration parameters for each sensor type
    Fields:
    - sensor_id: Unique identifier for the sensor
    - sensor_type: Type of sensor (camera, lidar, imu, etc.)
    - topic: ROS topic name for the sensor
    - calibration_file: Path to calibration file
    - enabled: Whether the sensor is enabled
    - processing_frequency: Frequency at which the sensor data is processed
    - parameters: Additional sensor-specific parameters
    """
    sensor_id: str
    sensor_type: SensorType
    topic: str
    calibration_file: str
    enabled: bool = True
    processing_frequency: float = 10.0  # Hz
    parameters: Optional[Dict[str, Any]] = None
    state: SensorConfigState = SensorConfigState.UNCONFIGURED


@dataclass
class SensorData:
    """
    Represents raw data from a specific sensor type
    Fields:
    - id: Unique identifier for the sensor data instance
    - sensor_type: Type of sensor (camera, lidar, imu, etc.)
    - timestamp: Acquisition time with nanosecond precision
    - data: Raw sensor data payload (varies by sensor type)
    - frame_id: Coordinate frame identifier for the sensor
    - metadata: Additional sensor-specific metadata
    """
    id: str
    sensor_type: SensorType
    timestamp: float  # Unix timestamp in seconds
    data: Any  # Raw sensor data payload
    frame_id: str
    metadata: Dict[str, Any]
    state: SensorDataState = SensorDataState.ACQUIRED


@dataclass
class UltrasonicData:
    """Ultrasonic sensor-specific data."""
    id: str
    sensor_type: SensorType
    timestamp: float  # Unix timestamp in seconds
    distance: float  # in meters
    confidence: float = 1.0
    frame_id: str = ""
    metadata: Optional[Dict[str, Any]] = None
    state: SensorDataState = SensorDataState.ACQUIRED


@dataclass
class ProcessedData:
    """
    Represents sensor data after initial processing
    Fields:
    - id: Unique identifier for the processed data instance
    - source_data_id: Reference to the original SensorData
    - timestamp: Processing completion time
    - processed_payload: Processed data (detected objects, features, etc.)
    - confidence_score: Confidence level of the processing results
    - processing_method: Algorithm or method used for processing
    """
    id: str
    source_data_id: str
    timestamp: float  # Unix timestamp in seconds
    processed_payload: Any
    confidence_score: float  # Between 0.0 and 1.0
    processing_method: str
    state: ProcessingState = ProcessingState.COMPLETED


@dataclass
class DetectedObject:
    """
    Represents an object detected by the computer vision system
    Fields:
    - id: Unique identifier for the detected object
    - object_type: Type of object (human, furniture, door, etc.)
    - position: 3D position relative to robot
    - bounding_box: 2D/3D bounding box coordinates
    - confidence_score: Detection confidence level
    - timestamp: Time of detection
    """
    id: str
    object_type: str  # human, furniture, door, etc.
    position_x: float
    position_y: float
    position_z: float
    bounding_box: List[float]  # [x, y, width, height] or 3D bbox
    confidence_score: float  # Between 0.0 and 1.0
    timestamp: float  # Unix timestamp in seconds


@dataclass
class FusedData:
    """
    Represents combined information from multiple sensors
    Fields:
    - id: Unique identifier for the fused data instance
    - timestamp: Fusion completion time
    - fused_payload: Combined sensor information
    - source_sensor_ids: List of source sensor data IDs used in fusion
    - confidence_score: Overall confidence of the fused data
    - coordinate_frame: Reference frame of the fused data
    """
    id: str
    timestamp: float  # Unix timestamp in seconds
    fused_payload: Any
    source_sensor_ids: List[str]
    confidence_score: float  # Between 0.0 and 1.0
    coordinate_frame: str


@dataclass
class Position:
    """Position in 3D space."""
    x: float
    y: float
    z: float


@dataclass
class Orientation:
    """Orientation using quaternion representation."""
    x: float
    y: float
    z: float
    w: float


@dataclass
class Pose:
    """Pose combining position and orientation."""
    position: Position
    orientation: Orientation


@dataclass
class BoundingBox:
    """Bounding box for object detection."""
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min

    @property
    def center_x(self) -> float:
        return (self.x_min + self.x_max) / 2.0

    @property
    def center_y(self) -> float:
        return (self.y_min + self.y_max) / 2.0


@dataclass
class DetectedObject:
    """Information about a detected object."""
    id: str
    object_type: ObjectType
    confidence: float
    bounding_box: BoundingBox
    position_3d: Optional[Position] = None
    pose_3d: Optional[Pose] = None


@dataclass
class SensorFusionData:
    """Fused data from multiple sensors."""
    timestamp: float
    fused_objects: List[DetectedObject]
    environmental_map: Optional[Dict[str, Any]] = None
    confidence_scores: Optional[Dict[str, float]] = None  # Confidence per sensor
    sensor_data: Optional[Dict[str, SensorData]] = None  # Original sensor data


@dataclass
class EnvironmentalMap:
    """
    Represents the 3D environmental map
    Fields:
    - id: Unique identifier for the map
    - map_data: 3D occupancy grid or point cloud data
    - resolution: Spatial resolution of the map
    - origin_frame: Reference frame for the map
    - update_timestamp: Last update time
    - coverage_area: Bounding box of mapped area
    """
    id: str
    map_data: np.ndarray  # 3D occupancy grid or point cloud
    resolution: float  # In meters
    origin_frame: str
    update_timestamp: float  # Unix timestamp in seconds
    coverage_area: List[float]  # [min_x, min_y, min_z, max_x, max_y, max_z]
    objects: List[DetectedObject] = None
    origin: Position = None  # Adding origin position