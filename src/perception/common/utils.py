"""
Utility functions for the perception system.
"""
import time
from typing import List, Dict, Any, Optional


def get_current_timestamp() -> float:
    """
    Get the current timestamp in seconds (Unix timestamp).
    """
    return time.time()


def timestamp_to_ros_time(timestamp: float):
    """
    Convert a Unix timestamp to ROS Time message format.

    Args:
        timestamp: Unix timestamp in seconds (with fractional part for nanoseconds)

    Returns:
        builtin_interfaces.msg.Time: ROS Time message
    """
    from builtin_interfaces.msg import Time

    seconds = int(timestamp)
    nanoseconds = int((timestamp - seconds) * 1e9)

    time_msg = Time()
    time_msg.sec = seconds
    time_msg.nanosec = nanoseconds

    return time_msg


def ros_time_to_timestamp(ros_time: Dict[str, int]) -> float:
    """
    Convert a ROS Time message format to Unix timestamp.

    Args:
        ros_time: Dictionary with 'sec' and 'nanosec' keys

    Returns:
        float: Unix timestamp in seconds
    """
    return ros_time['sec'] + ros_time['nanosec'] / 1e9


def get_ros_time() -> float:
    """
    Get the current time as a Unix timestamp.

    Returns:
        float: Current time as Unix timestamp
    """
    return time.time()


def calculate_time_diff(start_time: float, end_time: float) -> float:
    """
    Calculate the time difference in seconds.

    Args:
        start_time: Start timestamp
        end_time: End timestamp

    Returns:
        float: Time difference in seconds
    """
    return end_time - start_time


def is_timestamp_valid(timestamp: float, max_future_offset: float = 1.0) -> bool:
    """
    Check if a timestamp is valid (not in the future beyond acceptable offset).

    Args:
        timestamp: Unix timestamp to validate
        max_future_offset: Maximum allowed offset in seconds (default 1 second)

    Returns:
        bool: True if timestamp is valid, False otherwise
    """
    current_time = get_current_timestamp()
    return timestamp <= (current_time + max_future_offset)


def normalize_confidence_score(score: float) -> float:
    """
    Normalize a confidence score to be between 0.0 and 1.0.

    Args:
        score: Raw confidence score

    Returns:
        float: Normalized confidence score between 0.0 and 1.0
    """
    return max(0.0, min(1.0, score))


def validate_coordinate_frame(frame_id: str) -> bool:
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


def validate_3d_position(position: List[float]) -> bool:
    """
    Validate that a position is a valid 3D coordinate.

    Args:
        position: List of [x, y, z] coordinates

    Returns:
        bool: True if position is valid, False otherwise
    """
    if len(position) != 3:
        return False

    # Check if all coordinates are finite numbers
    for coord in position:
        if not isinstance(coord, (int, float)) or not (abs(coord) < float('inf')):
            return False

    return True


def create_bounding_box(x: float, y: float, width: float, height: float) -> List[float]:
    """
    Create a 2D bounding box.

    Args:
        x: X coordinate of top-left corner
        y: Y coordinate of top-left corner
        width: Width of the bounding box
        height: Height of the bounding box

    Returns:
        List[float]: Bounding box as [x, y, width, height]
    """
    return [x, y, width, height]


def calculate_bounding_box_center(bbox: List[float]) -> List[float]:
    """
    Calculate the center of a 2D bounding box.

    Args:
        bbox: Bounding box as [x, y, width, height]

    Returns:
        List[float]: Center coordinates [center_x, center_y]
    """
    x, y, width, height = bbox
    center_x = x + width / 2.0
    center_y = y + height / 2.0
    return [center_x, center_y]


def get_ros_current_time() -> Dict[str, int]:
    """
    Get the current time in ROS Time format.

    Returns:
        Dict[str, int]: Dictionary with 'sec' and 'nanosec' keys
    """
    current_timestamp = time.time()
    seconds = int(current_timestamp)
    nanoseconds = int((current_timestamp - seconds) * 1e9)

    return {
        'sec': seconds,
        'nanosec': nanoseconds
    }


def time_diff_seconds(time1: float, time2: float) -> float:
    """
    Calculate the time difference in seconds between two timestamps.

    Args:
        time1: First timestamp (Unix timestamp)
        time2: Second timestamp (Unix timestamp)

    Returns:
        float: Time difference in seconds (time2 - time1)
    """
    return time2 - time1


def is_timestamp_valid_extended(timestamp: float, max_age_seconds: float = 10.0) -> bool:
    """
    Check if a timestamp is valid (not too old).

    Args:
        timestamp: Unix timestamp to check
        max_age_seconds: Maximum age in seconds (default 10 seconds)

    Returns:
        bool: True if timestamp is valid, False otherwise
    """
    current_time = get_current_timestamp()
    return current_time - timestamp <= max_age_seconds


def synchronize_timestamps(timestamps: list, reference_time: float = None) -> list:
    """
    Synchronize a list of timestamps to a reference time.

    Args:
        timestamps: List of timestamps to synchronize
        reference_time: Reference time to synchronize to (if None, uses current time)

    Returns:
        list: Synchronized timestamps
    """
    if reference_time is None:
        reference_time = get_current_timestamp()

    # Calculate the offset from the first timestamp to the reference
    if timestamps:
        offset = reference_time - timestamps[0]
        return [ts + offset for ts in timestamps]
    else:
        return []


def format_timestamp(timestamp: float, format_string: str = "%Y-%m-%d %H:%M:%S.%f") -> str:
    """
    Format a timestamp as a string.

    Args:
        timestamp: Unix timestamp to format
        format_string: Format string (default ISO-like format)

    Returns:
        str: Formatted timestamp string
    """
    from datetime import datetime
    dt = datetime.fromtimestamp(timestamp)
    return dt.strftime(format_string)


def interpolate_timestamps(start_time: float, end_time: float, num_points: int) -> list:
    """
    Interpolate timestamps between start and end time.

    Args:
        start_time: Start timestamp
        end_time: End timestamp
        num_points: Number of points to interpolate

    Returns:
        list: Interpolated timestamps
    """
    if num_points <= 0:
        return []

    if num_points == 1:
        return [start_time]

    step = (end_time - start_time) / (num_points - 1)
    return [start_time + i * step for i in range(num_points)]


def timestamp_to_nanoseconds(timestamp: float) -> int:
    """
    Convert a Unix timestamp to nanoseconds.

    Args:
        timestamp: Unix timestamp in seconds

    Returns:
        int: Timestamp in nanoseconds
    """
    return int(timestamp * 1e9)


def nanoseconds_to_timestamp(nanoseconds: int) -> float:
    """
    Convert nanoseconds to a Unix timestamp.

    Args:
        nanoseconds: Timestamp in nanoseconds

    Returns:
        float: Unix timestamp in seconds
    """
    return float(nanoseconds) / 1e9