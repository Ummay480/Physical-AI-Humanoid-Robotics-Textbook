"""
3D environmental mapping implementation.

Provides occupancy grid mapping and point cloud processing for building
a coherent 3D representation of the robot's environment.
"""
import numpy as np
from typing import List, Tuple, Optional, Dict
import threading
from dataclasses import dataclass

from ..common.data_types import EnvironmentalMap, DetectedObject
from ..common.utils import get_current_timestamp


@dataclass
class MapConfig:
    """Configuration for environmental mapping."""
    # Map resolution in meters
    resolution: float = 0.05

    # Map size (width, height, depth) in meters
    map_size: Tuple[float, float, float] = (20.0, 20.0, 5.0)

    # Origin position (x, y, z) in meters
    origin: Tuple[float, float, float] = (-10.0, -10.0, 0.0)

    # Maximum range for sensor integration (meters)
    max_range: float = 10.0

    # Occupancy thresholds
    occupied_threshold: float = 0.7
    free_threshold: float = 0.3

    # Update frequency (Hz)
    update_frequency: float = 5.0


class OccupancyGrid3D:
    """
    3D occupancy grid for environmental mapping.

    Uses a voxel grid to represent occupied, free, and unknown space.
    """

    def __init__(self, config: MapConfig):
        """
        Initialize 3D occupancy grid.

        Args:
            config: Map configuration
        """
        self.config = config

        # Calculate grid dimensions
        self.grid_size = (
            int(config.map_size[0] / config.resolution),
            int(config.map_size[1] / config.resolution),
            int(config.map_size[2] / config.resolution)
        )

        # Initialize grid (0.5 = unknown, 0 = free, 1 = occupied)
        self.grid = np.ones(self.grid_size) * 0.5

        # Occupancy log-odds representation for probabilistic updates
        self.log_odds = np.zeros(self.grid_size)

        # Thread lock for grid access
        self._lock = threading.Lock()

        # Statistics
        self.last_update_time = 0.0
        self.update_count = 0

    def world_to_grid(self, world_pos: np.ndarray) -> Tuple[int, int, int]:
        """
        Convert world coordinates to grid indices.

        Args:
            world_pos: Position in world frame [x, y, z]

        Returns:
            Grid indices (ix, iy, iz)
        """
        # Translate to grid frame
        grid_pos = (world_pos - np.array(self.config.origin)) / self.config.resolution

        # Round to nearest grid cell
        ix = int(np.round(grid_pos[0]))
        iy = int(np.round(grid_pos[1]))
        iz = int(np.round(grid_pos[2]))

        return ix, iy, iz

    def grid_to_world(self, grid_idx: Tuple[int, int, int]) -> np.ndarray:
        """
        Convert grid indices to world coordinates.

        Args:
            grid_idx: Grid indices (ix, iy, iz)

        Returns:
            Position in world frame [x, y, z]
        """
        ix, iy, iz = grid_idx
        world_pos = (np.array([ix, iy, iz]) * self.config.resolution +
                    np.array(self.config.origin))
        return world_pos

    def is_valid_index(self, idx: Tuple[int, int, int]) -> bool:
        """
        Check if grid index is valid.

        Args:
            idx: Grid indices (ix, iy, iz)

        Returns:
            True if valid, False otherwise
        """
        ix, iy, iz = idx
        return (0 <= ix < self.grid_size[0] and
                0 <= iy < self.grid_size[1] and
                0 <= iz < self.grid_size[2])

    def update_from_point_cloud(self, points: np.ndarray,
                                sensor_origin: np.ndarray):
        """
        Update occupancy grid from point cloud data.

        Args:
            points: Point cloud (Nx3 array)
            sensor_origin: Sensor position in world frame
        """
        with self._lock:
            for point in points:
                # Skip points beyond max range
                distance = np.linalg.norm(point - sensor_origin)
                if distance > self.config.max_range:
                    continue

                # Mark endpoint as occupied
                end_idx = self.world_to_grid(point)
                if self.is_valid_index(end_idx):
                    self._update_cell(end_idx, occupied=True)

                # Ray trace from sensor to point (mark as free)
                self._ray_trace(sensor_origin, point)

            self.last_update_time = get_current_timestamp()
            self.update_count += 1

    def _ray_trace(self, start: np.ndarray, end: np.ndarray):
        """
        Ray trace from start to end, marking cells as free.

        Args:
            start: Start position in world frame
            end: End position in world frame
        """
        # Bresenham-like 3D line algorithm
        start_idx = self.world_to_grid(start)
        end_idx = self.world_to_grid(end)

        # Get ray direction
        ray = np.array(end_idx) - np.array(start_idx)
        length = int(np.linalg.norm(ray))

        if length == 0:
            return

        # Normalize
        ray = ray / length

        # Trace ray
        for i in range(length - 1):  # Don't mark endpoint
            t = i / length
            current_idx = tuple((np.array(start_idx) + t * ray * length).astype(int))

            if self.is_valid_index(current_idx):
                self._update_cell(current_idx, occupied=False)

    def _update_cell(self, idx: Tuple[int, int, int], occupied: bool):
        """
        Update occupancy of a cell using log-odds.

        Args:
            idx: Grid index
            occupied: True if cell is occupied, False if free
        """
        # Log-odds update
        log_odds_update = 0.4 if occupied else -0.2

        self.log_odds[idx] += log_odds_update

        # Clamp log-odds
        self.log_odds[idx] = np.clip(self.log_odds[idx], -5.0, 5.0)

        # Convert to probability
        self.grid[idx] = 1.0 / (1.0 + np.exp(-self.log_odds[idx]))

    def update_from_detections(self, detections: List[DetectedObject]):
        """
        Update map from detected objects.

        Args:
            detections: List of detected objects
        """
        with self._lock:
            for detection in detections:
                # Get object position
                position = detection.position

                # Estimate object size (simplified)
                bbox = detection.bounding_box
                object_size = np.array([bbox[2], bbox[3], 1.5]) * self.config.resolution

                # Mark voxels as occupied in object region
                self._mark_object_region(position, object_size)

    def _mark_object_region(self, center: np.ndarray, size: np.ndarray):
        """
        Mark a region around detected object as occupied.

        Args:
            center: Center position of object
            size: Size of object (width, height, depth)
        """
        # Calculate voxel range
        half_size = size / 2.0
        min_pos = center - half_size
        max_pos = center + half_size

        min_idx = self.world_to_grid(min_pos)
        max_idx = self.world_to_grid(max_pos)

        # Mark all voxels in range
        for ix in range(max(0, min_idx[0]), min(self.grid_size[0], max_idx[0] + 1)):
            for iy in range(max(0, min_idx[1]), min(self.grid_size[1], max_idx[1] + 1)):
                for iz in range(max(0, min_idx[2]), min(self.grid_size[2], max_idx[2] + 1)):
                    self._update_cell((ix, iy, iz), occupied=True)

    def get_occupied_cells(self, threshold: Optional[float] = None) -> np.ndarray:
        """
        Get positions of occupied cells.

        Args:
            threshold: Occupancy threshold (uses config if None)

        Returns:
            Nx3 array of occupied cell positions
        """
        if threshold is None:
            threshold = self.config.occupied_threshold

        with self._lock:
            # Find occupied cells
            occupied_mask = self.grid > threshold
            indices = np.argwhere(occupied_mask)

            # Convert to world coordinates
            positions = np.array([
                self.grid_to_world(tuple(idx))
                for idx in indices
            ])

            return positions

    def get_free_cells(self, threshold: Optional[float] = None) -> np.ndarray:
        """
        Get positions of free cells.

        Args:
            threshold: Free threshold (uses config if None)

        Returns:
            Nx3 array of free cell positions
        """
        if threshold is None:
            threshold = self.config.free_threshold

        with self._lock:
            # Find free cells
            free_mask = self.grid < threshold
            indices = np.argwhere(free_mask)

            # Convert to world coordinates
            positions = np.array([
                self.grid_to_world(tuple(idx))
                for idx in indices
            ])

            return positions

    def get_map_data(self) -> EnvironmentalMap:
        """
        Get environmental map data.

        Returns:
            EnvironmentalMap object
        """
        with self._lock:
            # Calculate coverage area
            coverage_area = np.array([
                self.config.origin[0],
                self.config.origin[1],
                self.config.origin[2],
                self.config.origin[0] + self.config.map_size[0],
                self.config.origin[1] + self.config.map_size[1],
                self.config.origin[2] + self.config.map_size[2]
            ])

            env_map = EnvironmentalMap(
                map_data=self.grid.copy(),
                resolution=self.config.resolution,
                origin_frame='base_link',
                update_timestamp=self.last_update_time,
                coverage_area=coverage_area,
                id=f"map_{int(get_current_timestamp() * 1e9)}"
            )

            return env_map

    def clear_map(self):
        """Clear the occupancy grid."""
        with self._lock:
            self.grid = np.ones(self.grid_size) * 0.5
            self.log_odds = np.zeros(self.grid_size)
            self.update_count = 0

    def get_statistics(self) -> Dict:
        """
        Get mapping statistics.

        Returns:
            Dictionary with statistics
        """
        with self._lock:
            occupied_count = np.sum(self.grid > self.config.occupied_threshold)
            free_count = np.sum(self.grid < self.config.free_threshold)
            unknown_count = self.grid.size - occupied_count - free_count

            return {
                'grid_size': self.grid_size,
                'resolution': self.config.resolution,
                'occupied_cells': int(occupied_count),
                'free_cells': int(free_count),
                'unknown_cells': int(unknown_count),
                'total_cells': int(self.grid.size),
                'update_count': self.update_count,
                'last_update_time': self.last_update_time
            }


class PointCloudProcessor:
    """
    Processes point cloud data for mapping and object detection.
    """

    def __init__(self):
        """Initialize point cloud processor."""
        pass

    def downsample(self, points: np.ndarray, voxel_size: float = 0.05) -> np.ndarray:
        """
        Downsample point cloud using voxel grid filter.

        Args:
            points: Input point cloud (Nx3)
            voxel_size: Voxel size for downsampling

        Returns:
            Downsampled point cloud
        """
        if len(points) == 0:
            return points

        # Voxelize points
        voxel_indices = np.floor(points / voxel_size).astype(int)

        # Find unique voxels
        unique_voxels, inverse_indices = np.unique(
            voxel_indices, axis=0, return_inverse=True
        )

        # Average points in each voxel
        downsampled = np.zeros((len(unique_voxels), 3))
        for i, voxel in enumerate(unique_voxels):
            mask = inverse_indices == i
            downsampled[i] = np.mean(points[mask], axis=0)

        return downsampled

    def remove_outliers(self, points: np.ndarray,
                       nb_neighbors: int = 20,
                       std_ratio: float = 2.0) -> np.ndarray:
        """
        Remove statistical outliers from point cloud.

        Args:
            points: Input point cloud (Nx3)
            nb_neighbors: Number of neighbors to consider
            std_ratio: Standard deviation ratio threshold

        Returns:
            Filtered point cloud
        """
        if len(points) < nb_neighbors:
            return points

        # For each point, compute distance to nearest neighbors
        from scipy.spatial import cKDTree

        tree = cKDTree(points)
        distances, _ = tree.query(points, k=nb_neighbors + 1)

        # Mean distance to neighbors (excluding self)
        mean_distances = np.mean(distances[:, 1:], axis=1)

        # Compute threshold
        global_mean = np.mean(mean_distances)
        global_std = np.std(mean_distances)
        threshold = global_mean + std_ratio * global_std

        # Filter outliers
        mask = mean_distances < threshold
        filtered_points = points[mask]

        return filtered_points

    def extract_ground_plane(self, points: np.ndarray,
                            distance_threshold: float = 0.05,
                            max_iterations: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract ground plane using RANSAC.

        Args:
            points: Input point cloud (Nx3)
            distance_threshold: Distance threshold for inliers
            max_iterations: Maximum RANSAC iterations

        Returns:
            Tuple of (ground_points, non_ground_points)
        """
        if len(points) < 3:
            return np.array([]), points

        best_inliers = None
        best_count = 0

        # RANSAC
        for _ in range(max_iterations):
            # Sample 3 random points
            sample_idx = np.random.choice(len(points), 3, replace=False)
            sample = points[sample_idx]

            # Fit plane
            v1 = sample[1] - sample[0]
            v2 = sample[2] - sample[0]
            normal = np.cross(v1, v2)

            if np.linalg.norm(normal) == 0:
                continue

            normal = normal / np.linalg.norm(normal)

            # Compute distances to plane
            distances = np.abs(np.dot(points - sample[0], normal))

            # Count inliers
            inliers = distances < distance_threshold
            count = np.sum(inliers)

            if count > best_count:
                best_count = count
                best_inliers = inliers

        if best_inliers is None:
            return np.array([]), points

        # Split points
        ground_points = points[best_inliers]
        non_ground_points = points[~best_inliers]

        return ground_points, non_ground_points

    def cluster_points(self, points: np.ndarray,
                      eps: float = 0.1,
                      min_points: int = 10) -> List[np.ndarray]:
        """
        Cluster points using DBSCAN.

        Args:
            points: Input point cloud (Nx3)
            eps: Maximum distance between points in cluster
            min_points: Minimum points for a cluster

        Returns:
            List of point clusters
        """
        if len(points) == 0:
            return []

        from scipy.spatial import cKDTree

        tree = cKDTree(points)
        clusters = []
        visited = np.zeros(len(points), dtype=bool)
        noise = np.zeros(len(points), dtype=bool)

        for i in range(len(points)):
            if visited[i]:
                continue

            visited[i] = True
            neighbors = tree.query_ball_point(points[i], eps)

            if len(neighbors) < min_points:
                noise[i] = True
            else:
                # Start new cluster
                cluster = [i]
                neighbors = set(neighbors) - {i}

                while neighbors:
                    j = neighbors.pop()
                    if not visited[j]:
                        visited[j] = True
                        new_neighbors = tree.query_ball_point(points[j], eps)

                        if len(new_neighbors) >= min_points:
                            neighbors.update(set(new_neighbors) - set(cluster))

                    if j not in cluster:
                        cluster.append(j)

                clusters.append(points[cluster])

        return clusters
