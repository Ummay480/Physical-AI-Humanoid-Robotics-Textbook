"""
QoS Profiles for ROS 2 Nervous System.

Defines Quality of Service profiles for different message types in the system orchestration layer.
These profiles ensure proper message delivery characteristics for each use case.
"""

from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy, QoSHistoryPolicy
from typing import Dict


# Orchestration Data QoS Profile
# For reliable delivery of orchestrated data between modules
ORCHESTRATION_DATA_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.VOLATILE,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=10
)

# System Health QoS Profile
# For best-effort delivery of system health status updates
SYSTEM_HEALTH_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    durability=QoSDurabilityPolicy.VOLATILE,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1
)

# Diagnostics QoS Profile
# For best-effort delivery of diagnostic information
DIAGNOSTICS_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    durability=QoSDurabilityPolicy.VOLATILE,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=5
)

# Navigation Input QoS Profile
# For reliable delivery of navigation commands (future Nav2 integration)
# TODO: Adjust when Nav2 integration begins
NAVIGATION_INPUT_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.VOLATILE,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=5
)

# Control Input QoS Profile
# For reliable, low-latency delivery of control commands (future Control module integration)
# TODO: Adjust when Control module integration begins
CONTROL_INPUT_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.VOLATILE,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1
)

# Perception Data QoS Profiles (for subscribing to Perception module)
# These match the QoS profiles defined in the Perception module

REAL_TIME_SENSOR_DATA_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    durability=QoSDurabilityPolicy.VOLATILE,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1
)

PROCESSED_DATA_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    durability=QoSDurabilityPolicy.VOLATILE,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=10
)

MAP_DATA_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.VOLATILE,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=10
)

IMU_SENSOR_DATA_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.VOLATILE,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1
)

STATUS_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    durability=QoSDurabilityPolicy.VOLATILE,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1
)


def get_qos_profiles() -> Dict[str, QoSProfile]:
    """
    Get dictionary of all QoS profiles.

    Returns:
        Dictionary mapping profile names to QoSProfile objects
    """
    return {
        # System orchestration profiles
        'orchestration_data': ORCHESTRATION_DATA_QOS,
        'system_health': SYSTEM_HEALTH_QOS,
        'diagnostics': DIAGNOSTICS_QOS,
        'navigation_input': NAVIGATION_INPUT_QOS,
        'control_input': CONTROL_INPUT_QOS,

        # Perception module profiles (for subscriptions)
        'real_time_sensor_data': REAL_TIME_SENSOR_DATA_QOS,
        'processed_data': PROCESSED_DATA_QOS,
        'map_data': MAP_DATA_QOS,
        'imu_sensor_data': IMU_SENSOR_DATA_QOS,
        'status': STATUS_QOS,
    }


def get_qos_profile(profile_name: str) -> QoSProfile:
    """
    Get a specific QoS profile by name.

    Args:
        profile_name: Name of the profile

    Returns:
        QoSProfile object

    Raises:
        KeyError: If profile name is not found
    """
    profiles = get_qos_profiles()

    if profile_name not in profiles:
        raise KeyError(
            f"Unknown QoS profile: {profile_name}. "
            f"Available profiles: {list(profiles.keys())}"
        )

    return profiles[profile_name]


def validate_qos_compatibility(publisher_qos: QoSProfile, subscriber_qos: QoSProfile) -> bool:
    """
    Check if publisher and subscriber QoS profiles are compatible.

    Args:
        publisher_qos: Publisher's QoS profile
        subscriber_qos: Subscriber's QoS profile

    Returns:
        True if compatible, False otherwise
    """
    # Reliability compatibility:
    # RELIABLE publisher can work with BEST_EFFORT or RELIABLE subscriber
    # BEST_EFFORT publisher only works with BEST_EFFORT subscriber
    if (publisher_qos.reliability == QoSReliabilityPolicy.BEST_EFFORT and
        subscriber_qos.reliability == QoSReliabilityPolicy.RELIABLE):
        return False

    # Durability compatibility:
    # TRANSIENT_LOCAL publisher can work with VOLATILE or TRANSIENT_LOCAL subscriber
    # VOLATILE publisher only works with VOLATILE subscriber
    if (publisher_qos.durability == QoSDurabilityPolicy.VOLATILE and
        subscriber_qos.durability == QoSDurabilityPolicy.TRANSIENT_LOCAL):
        return False

    return True
