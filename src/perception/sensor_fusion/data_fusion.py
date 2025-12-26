"""
Data fusion algorithms for combining multiple sensor inputs.

Provides algorithms for fusing camera, LIDAR, and IMU data to create
a coherent understanding of the environment.
"""
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import threading

from .kalman_filter import MultiSensorKalmanFilter, KalmanState
from ..common.data_types import SensorData, SensorType, FusedData, DetectedObject
from ..common.utils import get_current_timestamp


@dataclass
class FusionConfig:
    """Configuration for sensor fusion."""
    # Weight for each sensor type (0.0 - 1.0)
    sensor_weights: Dict[str, float]

    # Time synchronization tolerance (seconds)
    sync_tolerance: float = 0.1

    # Minimum sensors required for fusion
    min_sensors: int = 2

    # Enable temporal filtering
    temporal_filtering: bool = True

    # Uncertainty threshold for acceptance
    uncertainty_threshold: float = 1.0


class SensorDataFusion:
    """
    Manages fusion of data from multiple sensors.

    Combines camera, LIDAR, and IMU data using weighted averaging,
    Kalman filtering, and temporal consistency checks.
    """

    def __init__(self, config: FusionConfig):
        """
        Initialize sensor data fusion.

        Args:
            config: Fusion configuration
        """
        self.config = config

        # Buffer for recent sensor data
        self.sensor_buffers: Dict[str, List[SensorData]] = {}
        self.buffer_max_size = 100

        # Kalman filters for state estimation
        self.kalman_filters: Dict[str, MultiSensorKalmanFilter] = {}

        # Thread lock for buffer access
        self._lock = threading.Lock()

        # Fusion statistics
        self.stats = {
            'total_fusions': 0,
            'successful_fusions': 0,
            'failed_fusions': 0,
            'avg_uncertainty': 0.0
        }

    def add_sensor_data(self, sensor_id: str, sensor_data: SensorData):
        """
        Add sensor data to the fusion buffer.

        Args:
            sensor_id: ID of the sensor
            sensor_data: Sensor data to add
        """
        with self._lock:
            if sensor_id not in self.sensor_buffers:
                self.sensor_buffers[sensor_id] = []

            # Add to buffer
            self.sensor_buffers[sensor_id].append(sensor_data)

            # Limit buffer size
            if len(self.sensor_buffers[sensor_id]) > self.buffer_max_size:
                self.sensor_buffers[sensor_id].pop(0)

    def get_synchronized_data(self, timestamp: float,
                             tolerance: Optional[float] = None) -> Dict[str, SensorData]:
        """
        Get sensor data synchronized to a specific timestamp.

        Args:
            timestamp: Target timestamp
            tolerance: Time tolerance (uses config if None)

        Returns:
            Dictionary of sensor_id -> synchronized sensor data
        """
        if tolerance is None:
            tolerance = self.config.sync_tolerance

        synchronized = {}

        with self._lock:
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

    def fuse_position_estimates(self, positions: Dict[str, np.ndarray],
                                confidences: Dict[str, float]) -> Tuple[np.ndarray, float]:
        """
        Fuse multiple position estimates using weighted averaging.

        Args:
            positions: Dictionary of sensor_id -> position estimate
            confidences: Dictionary of sensor_id -> confidence score

        Returns:
            Tuple of (fused_position, fused_confidence)
        """
        if not positions:
            return np.zeros(3), 0.0

        # Normalize confidences to weights
        total_confidence = sum(confidences.values())
        if total_confidence == 0:
            # Use equal weights if no confidence info
            weights = {k: 1.0 / len(positions) for k in positions.keys()}
        else:
            weights = {k: v / total_confidence for k, v in confidences.items()}

        # Weighted average of positions
        fused_position = np.zeros(3)
        for sensor_id, position in positions.items():
            weight = weights[sensor_id]
            # Ensure position is 3D
            if len(position) < 3:
                position = np.append(position, np.zeros(3 - len(position)))
            fused_position += weight * position[:3]

        # Fused confidence (weighted average)
        fused_confidence = sum(confidences.values()) / len(confidences)

        return fused_position, fused_confidence

    def fuse_detected_objects(self, detections_list: List[List[DetectedObject]],
                             iou_threshold: float = 0.5) -> List[DetectedObject]:
        """
        Fuse object detections from multiple sensors.

        Args:
            detections_list: List of detection lists from different sensors
            iou_threshold: IoU threshold for matching objects

        Returns:
            Fused list of detected objects
        """
        if not detections_list:
            return []

        # Flatten all detections
        all_detections = []
        for detections in detections_list:
            all_detections.extend(detections)

        if not all_detections:
            return []

        # Group similar detections (simple clustering based on position)
        fused_detections = []
        used = set()

        for i, det1 in enumerate(all_detections):
            if i in used:
                continue

            # Find all detections matching this one
            cluster = [det1]
            used.add(i)

            for j, det2 in enumerate(all_detections):
                if j <= i or j in used:
                    continue

                # Check if detections are similar (same type, close position)
                if det1.object_type == det2.object_type:
                    distance = np.linalg.norm(det1.position - det2.position)
                    if distance < 0.5:  # Within 0.5 meters
                        cluster.append(det2)
                        used.add(j)

            # Fuse cluster into single detection
            if cluster:
                fused_det = self._fuse_detection_cluster(cluster)
                fused_detections.append(fused_det)

        return fused_detections

    def _fuse_detection_cluster(self, cluster: List[DetectedObject]) -> DetectedObject:
        """
        Fuse a cluster of similar detections into one.

        Args:
            cluster: List of similar detected objects

        Returns:
            Fused detected object
        """
        # Average positions weighted by confidence
        total_confidence = sum(det.confidence_score for det in cluster)
        weights = np.array([det.confidence_score / total_confidence for det in cluster])

        # Weighted average position
        positions = np.array([det.position for det in cluster])
        fused_position = np.average(positions, axis=0, weights=weights)

        # Average bounding box
        bboxes = np.array([det.bounding_box for det in cluster])
        fused_bbox = np.average(bboxes, axis=0, weights=weights)

        # Maximum confidence
        fused_confidence = max(det.confidence_score for det in cluster)

        # Use most recent timestamp
        fused_timestamp = max(det.timestamp for det in cluster)

        # Create fused detection
        fused_detection = DetectedObject(
            object_type=cluster[0].object_type,
            position=fused_position,
            bounding_box=fused_bbox,
            confidence_score=fused_confidence,
            timestamp=fused_timestamp,
            id=f"fused_{cluster[0].id}"
        )

        return fused_detection

    def apply_kalman_filtering(self, sensor_id: str, measurement: np.ndarray,
                              timestamp: float) -> np.ndarray:
        """
        Apply Kalman filtering to sensor measurement.

        Args:
            sensor_id: ID of the sensor
            measurement: Measurement vector
            timestamp: Measurement timestamp

        Returns:
            Filtered state estimate
        """
        # Create filter if doesn't exist
        if sensor_id not in self.kalman_filters:
            dim = len(measurement)
            kf = MultiSensorKalmanFilter(dim)

            # Add sensor with identity measurement matrix
            H = np.eye(dim)
            R = np.eye(dim) * 0.1  # Measurement noise
            kf.add_sensor(sensor_id, H, R)

            self.kalman_filters[sensor_id] = kf

        kf = self.kalman_filters[sensor_id]

        # Update filter
        state = kf.update_from_sensor(sensor_id, measurement, timestamp)

        return state.x

    def reduce_uncertainty(self, fused_data: FusedData,
                          sensor_data_list: List[SensorData]) -> float:
        """
        Calculate uncertainty reduction from sensor fusion.

        Args:
            fused_data: Fused data result
            sensor_data_list: List of input sensor data

        Returns:
            Uncertainty reduction factor (0.0 - 1.0)
        """
        if not sensor_data_list:
            return 0.0

        # More sensors = lower uncertainty
        num_sensors = len(sensor_data_list)
        base_reduction = min(0.9, 0.3 + 0.2 * num_sensors)

        # Confidence-based adjustment
        avg_confidence = fused_data.confidence_score
        confidence_factor = avg_confidence

        # Combined uncertainty reduction
        uncertainty_reduction = base_reduction * confidence_factor

        return float(uncertainty_reduction)

    def create_fused_data(self, sensor_data_dict: Dict[str, SensorData],
                         coordinate_frame: str = 'base_link') -> Optional[FusedData]:
        """
        Create fused data from synchronized sensor inputs.

        Args:
            sensor_data_dict: Dictionary of sensor_id -> sensor data
            coordinate_frame: Target coordinate frame

        Returns:
            Fused data or None if fusion failed
        """
        if len(sensor_data_dict) < self.config.min_sensors:
            self.stats['failed_fusions'] += 1
            return None

        self.stats['total_fusions'] += 1

        # Extract positions and confidences
        positions = {}
        confidences = {}

        for sensor_id, data in sensor_data_dict.items():
            # For now, use simple data extraction
            # This should be customized based on data type
            weight = self.config.sensor_weights.get(sensor_id, 1.0)
            confidences[sensor_id] = weight

            # Extract position from data (simplified)
            if hasattr(data, 'position'):
                positions[sensor_id] = data.position
            elif isinstance(data.data, np.ndarray) and len(data.data) >= 3:
                positions[sensor_id] = data.data[:3]

        if not positions:
            self.stats['failed_fusions'] += 1
            return None

        # Fuse positions
        fused_position, fused_confidence = self.fuse_position_estimates(
            positions, confidences
        )

        # Create fused data
        fused_data = FusedData(
            timestamp=get_current_timestamp(),
            fused_payload={'position': fused_position.tolist()},
            source_sensor_ids=list(sensor_data_dict.keys()),
            confidence_score=fused_confidence,
            coordinate_frame=coordinate_frame,
            id=f"fused_{int(get_current_timestamp() * 1e9)}"
        )

        # Calculate uncertainty reduction
        uncertainty_reduction = self.reduce_uncertainty(
            fused_data, list(sensor_data_dict.values())
        )

        self.stats['successful_fusions'] += 1
        self.stats['avg_uncertainty'] = (
            (self.stats['avg_uncertainty'] * (self.stats['successful_fusions'] - 1) +
             (1.0 - uncertainty_reduction)) / self.stats['successful_fusions']
        )

        return fused_data

    def get_fusion_statistics(self) -> Dict:
        """
        Get fusion statistics.

        Returns:
            Dictionary with fusion statistics
        """
        with self._lock:
            stats = self.stats.copy()

            # Add buffer statistics
            stats['buffer_sizes'] = {
                sensor_id: len(buffer)
                for sensor_id, buffer in self.sensor_buffers.items()
            }

            # Add success rate
            if stats['total_fusions'] > 0:
                stats['success_rate'] = (
                    stats['successful_fusions'] / stats['total_fusions']
                )
            else:
                stats['success_rate'] = 0.0

            return stats

    def clear_buffers(self):
        """Clear all sensor buffers."""
        with self._lock:
            self.sensor_buffers.clear()

    def reset_statistics(self):
        """Reset fusion statistics."""
        self.stats = {
            'total_fusions': 0,
            'successful_fusions': 0,
            'failed_fusions': 0,
            'avg_uncertainty': 0.0
        }


class TemporalFusion:
    """
    Temporal fusion for smoothing sensor data over time.
    """

    def __init__(self, window_size: int = 10, alpha: float = 0.3):
        """
        Initialize temporal fusion.

        Args:
            window_size: Size of temporal window
            alpha: Exponential smoothing factor (0.0 - 1.0)
        """
        self.window_size = window_size
        self.alpha = alpha

        # Temporal buffers for each data type
        self.temporal_buffers: Dict[str, List] = {}

    def add_data(self, data_id: str, data: np.ndarray):
        """
        Add data to temporal buffer.

        Args:
            data_id: Identifier for data stream
            data: Data to add
        """
        if data_id not in self.temporal_buffers:
            self.temporal_buffers[data_id] = []

        self.temporal_buffers[data_id].append(data)

        # Limit buffer size
        if len(self.temporal_buffers[data_id]) > self.window_size:
            self.temporal_buffers[data_id].pop(0)

    def get_smoothed_data(self, data_id: str,
                         method: str = 'exponential') -> Optional[np.ndarray]:
        """
        Get temporally smoothed data.

        Args:
            data_id: Identifier for data stream
            method: Smoothing method ('exponential', 'moving_average')

        Returns:
            Smoothed data or None if insufficient data
        """
        if data_id not in self.temporal_buffers:
            return None

        buffer = self.temporal_buffers[data_id]
        if not buffer:
            return None

        if method == 'exponential':
            # Exponential moving average
            smoothed = buffer[0].copy()
            for data in buffer[1:]:
                smoothed = self.alpha * data + (1 - self.alpha) * smoothed
            return smoothed

        elif method == 'moving_average':
            # Simple moving average
            return np.mean(buffer, axis=0)

        return buffer[-1]  # Return latest if unknown method
