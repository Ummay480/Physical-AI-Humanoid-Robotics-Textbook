"""
Kalman filter implementation for sensor fusion.

Provides Extended Kalman Filter (EKF) and standard Kalman Filter for fusing
sensor data from multiple sources to reduce uncertainty.
"""
import numpy as np
from typing import Optional, Callable, Tuple
from dataclasses import dataclass


@dataclass
class KalmanState:
    """
    Represents the state of a Kalman filter.
    """
    # State estimate
    x: np.ndarray

    # State covariance matrix
    P: np.ndarray

    # Timestamp
    timestamp: float

    # State dimension
    @property
    def dim(self) -> int:
        return len(self.x)


class KalmanFilter:
    """
    Standard Kalman Filter for linear systems.

    Assumes linear state transition and measurement models:
    x_k = F * x_{k-1} + B * u_k + w_k  (state transition)
    z_k = H * x_k + v_k                 (measurement)

    where w_k ~ N(0, Q) and v_k ~ N(0, R)
    """

    def __init__(self, dim_x: int, dim_z: int):
        """
        Initialize Kalman filter.

        Args:
            dim_x: Dimension of state vector
            dim_z: Dimension of measurement vector
        """
        self.dim_x = dim_x
        self.dim_z = dim_z

        # State vector
        self.x = np.zeros(dim_x)

        # State covariance matrix
        self.P = np.eye(dim_x)

        # State transition matrix
        self.F = np.eye(dim_x)

        # Control input matrix
        self.B = None

        # Measurement function
        self.H = np.zeros((dim_z, dim_x))

        # Process noise covariance
        self.Q = np.eye(dim_x) * 0.01

        # Measurement noise covariance
        self.R = np.eye(dim_z) * 0.1

        # Identity matrix
        self._I = np.eye(dim_x)

        # Last update timestamp
        self.timestamp = 0.0

    def predict(self, u: Optional[np.ndarray] = None, F: Optional[np.ndarray] = None,
                Q: Optional[np.ndarray] = None) -> KalmanState:
        """
        Predict step of Kalman filter.

        Args:
            u: Control input (optional)
            F: State transition matrix (if None, uses self.F)
            Q: Process noise covariance (if None, uses self.Q)

        Returns:
            Predicted state
        """
        if F is not None:
            self.F = F

        if Q is not None:
            self.Q = Q

        # Predict state
        self.x = self.F @ self.x

        # Add control input if provided
        if u is not None and self.B is not None:
            self.x += self.B @ u

        # Predict covariance
        self.P = self.F @ self.P @ self.F.T + self.Q

        return KalmanState(self.x.copy(), self.P.copy(), self.timestamp)

    def update(self, z: np.ndarray, H: Optional[np.ndarray] = None,
               R: Optional[np.ndarray] = None) -> KalmanState:
        """
        Update step of Kalman filter.

        Args:
            z: Measurement vector
            H: Measurement matrix (if None, uses self.H)
            R: Measurement noise covariance (if None, uses self.R)

        Returns:
            Updated state
        """
        if H is not None:
            self.H = H

        if R is not None:
            self.R = R

        # Innovation (measurement residual)
        y = z - self.H @ self.x

        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # Update state
        self.x = self.x + K @ y

        # Update covariance
        self.P = (self._I - K @ self.H) @ self.P

        return KalmanState(self.x.copy(), self.P.copy(), self.timestamp)

    def get_state(self) -> KalmanState:
        """
        Get current state.

        Returns:
            Current Kalman state
        """
        return KalmanState(self.x.copy(), self.P.copy(), self.timestamp)

    def set_state(self, x: np.ndarray, P: Optional[np.ndarray] = None):
        """
        Set the filter state.

        Args:
            x: State vector
            P: Covariance matrix (if None, keeps current)
        """
        self.x = x.copy()
        if P is not None:
            self.P = P.copy()

    def reset(self):
        """Reset filter to initial state."""
        self.x = np.zeros(self.dim_x)
        self.P = np.eye(self.dim_x)


class ExtendedKalmanFilter:
    """
    Extended Kalman Filter for nonlinear systems.

    Uses linearization of nonlinear functions via Jacobians.
    """

    def __init__(self, dim_x: int, dim_z: int,
                 f: Callable, h: Callable,
                 F_jacobian: Callable, H_jacobian: Callable):
        """
        Initialize Extended Kalman Filter.

        Args:
            dim_x: Dimension of state vector
            dim_z: Dimension of measurement vector
            f: State transition function f(x, u) -> x'
            h: Measurement function h(x) -> z
            F_jacobian: Jacobian of f with respect to x
            H_jacobian: Jacobian of h with respect to x
        """
        self.dim_x = dim_x
        self.dim_z = dim_z

        # Nonlinear functions
        self.f = f
        self.h = h

        # Jacobian functions
        self.F_jacobian = F_jacobian
        self.H_jacobian = H_jacobian

        # State vector
        self.x = np.zeros(dim_x)

        # State covariance matrix
        self.P = np.eye(dim_x)

        # Process noise covariance
        self.Q = np.eye(dim_x) * 0.01

        # Measurement noise covariance
        self.R = np.eye(dim_z) * 0.1

        # Identity matrix
        self._I = np.eye(dim_x)

        # Last update timestamp
        self.timestamp = 0.0

    def predict(self, u: Optional[np.ndarray] = None,
                dt: float = 1.0) -> KalmanState:
        """
        Predict step using nonlinear state transition.

        Args:
            u: Control input (optional)
            dt: Time step

        Returns:
            Predicted state
        """
        # Compute Jacobian at current state
        F = self.F_jacobian(self.x, u, dt)

        # Predict state using nonlinear function
        self.x = self.f(self.x, u, dt)

        # Predict covariance using linearization
        self.P = F @ self.P @ F.T + self.Q

        return KalmanState(self.x.copy(), self.P.copy(), self.timestamp)

    def update(self, z: np.ndarray) -> KalmanState:
        """
        Update step using nonlinear measurement function.

        Args:
            z: Measurement vector

        Returns:
            Updated state
        """
        # Compute Jacobian at current state
        H = self.H_jacobian(self.x)

        # Predicted measurement
        z_pred = self.h(self.x)

        # Innovation
        y = z - z_pred

        # Innovation covariance
        S = H @ self.P @ H.T + self.R

        # Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)

        # Update state
        self.x = self.x + K @ y

        # Update covariance
        self.P = (self._I - K @ H) @ self.P

        return KalmanState(self.x.copy(), self.P.copy(), self.timestamp)

    def get_state(self) -> KalmanState:
        """Get current state."""
        return KalmanState(self.x.copy(), self.P.copy(), self.timestamp)

    def set_state(self, x: np.ndarray, P: Optional[np.ndarray] = None):
        """Set the filter state."""
        self.x = x.copy()
        if P is not None:
            self.P = P.copy()

    def reset(self):
        """Reset filter to initial state."""
        self.x = np.zeros(self.dim_x)
        self.P = np.eye(self.dim_x)


class MultiSensorKalmanFilter:
    """
    Kalman filter for fusing data from multiple sensors.

    Supports asynchronous sensor updates at different rates.
    """

    def __init__(self, dim_x: int):
        """
        Initialize multi-sensor Kalman filter.

        Args:
            dim_x: Dimension of state vector
        """
        self.dim_x = dim_x

        # Use standard Kalman filter as base
        self.kf = KalmanFilter(dim_x, dim_x)

        # Sensor configurations: {sensor_id: {'H': H_matrix, 'R': R_matrix}}
        self.sensors = {}

        # Last update times for each sensor
        self.last_update = {}

    def add_sensor(self, sensor_id: str, H: np.ndarray, R: np.ndarray):
        """
        Add a sensor to the fusion system.

        Args:
            sensor_id: Unique identifier for the sensor
            H: Measurement matrix for this sensor
            R: Measurement noise covariance for this sensor
        """
        self.sensors[sensor_id] = {
            'H': H,
            'R': R,
            'dim_z': H.shape[0]
        }
        self.last_update[sensor_id] = 0.0

    def predict(self, dt: float) -> KalmanState:
        """
        Predict state forward in time.

        Args:
            dt: Time step

        Returns:
            Predicted state
        """
        # Update state transition matrix for time step
        # For constant velocity model: x_k = x_{k-1} + v_{k-1} * dt
        # This is application-specific and should be customized

        self.kf.timestamp += dt
        return self.kf.predict()

    def update_from_sensor(self, sensor_id: str, z: np.ndarray,
                          timestamp: float) -> KalmanState:
        """
        Update state using measurement from a specific sensor.

        Args:
            sensor_id: ID of the sensor providing the measurement
            z: Measurement vector
            timestamp: Timestamp of measurement

        Returns:
            Updated state
        """
        if sensor_id not in self.sensors:
            raise ValueError(f"Unknown sensor: {sensor_id}")

        sensor_config = self.sensors[sensor_id]

        # Predict to measurement time if needed
        if timestamp > self.kf.timestamp:
            dt = timestamp - self.kf.timestamp
            self.predict(dt)

        # Update with sensor measurement
        state = self.kf.update(z, H=sensor_config['H'], R=sensor_config['R'])

        self.last_update[sensor_id] = timestamp
        self.kf.timestamp = timestamp

        return state

    def get_state(self) -> KalmanState:
        """Get current fused state."""
        return self.kf.get_state()

    def set_state(self, x: np.ndarray, P: Optional[np.ndarray] = None):
        """Set the filter state."""
        self.kf.set_state(x, P)

    def get_uncertainty(self) -> float:
        """
        Get overall state uncertainty (trace of covariance matrix).

        Returns:
            Scalar uncertainty measure
        """
        return float(np.trace(self.kf.P))


def create_position_velocity_filter(dim: int = 3,
                                    process_noise: float = 0.01,
                                    measurement_noise: float = 0.1) -> KalmanFilter:
    """
    Create a Kalman filter for position and velocity tracking.

    State: [x, y, z, vx, vy, vz] for 3D
    Measurement: [x, y, z]

    Args:
        dim: Spatial dimension (2 or 3)
        process_noise: Process noise standard deviation
        measurement_noise: Measurement noise standard deviation

    Returns:
        Configured Kalman filter
    """
    dim_x = dim * 2  # position + velocity
    dim_z = dim      # position measurements

    kf = KalmanFilter(dim_x, dim_z)

    # State transition matrix (constant velocity model)
    # x_k = x_{k-1} + v_{k-1} * dt
    # v_k = v_{k-1}
    dt = 1.0  # Will be updated during predict
    F = np.eye(dim_x)
    for i in range(dim):
        F[i, dim + i] = dt
    kf.F = F

    # Measurement matrix (observe position only)
    H = np.zeros((dim_z, dim_x))
    H[:dim, :dim] = np.eye(dim)
    kf.H = H

    # Process noise covariance
    kf.Q = np.eye(dim_x) * process_noise**2

    # Measurement noise covariance
    kf.R = np.eye(dim_z) * measurement_noise**2

    return kf


def create_imu_fusion_filter(process_noise: float = 0.001,
                             measurement_noise: float = 0.01) -> KalmanFilter:
    """
    Create a Kalman filter for IMU sensor fusion.

    State: [orientation (quaternion), angular_velocity, linear_acceleration]

    Args:
        process_noise: Process noise standard deviation
        measurement_noise: Measurement noise standard deviation

    Returns:
        Configured Kalman filter
    """
    # State: 4 (quat) + 3 (gyro) + 3 (accel) = 10
    dim_x = 10
    dim_z = 10

    kf = KalmanFilter(dim_x, dim_z)

    # Simple model (identity transition for now)
    kf.F = np.eye(dim_x)
    kf.H = np.eye(dim_z)

    # IMU has low process noise
    kf.Q = np.eye(dim_x) * process_noise**2

    # Measurement noise
    kf.R = np.eye(dim_z) * measurement_noise**2

    return kf
