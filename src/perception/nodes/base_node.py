"""
Base node structure with QoS settings for the perception system.
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from rclpy.duration import Duration
from typing import Optional, Dict, Any

# QoS Profile constants based on the API contract
REAL_TIME_SENSOR_DATA_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
    deadline=Duration(seconds=0, nanoseconds=20000000)  # 20ms deadline
)

PROCESSED_DATA_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
    deadline=Duration(seconds=0, nanoseconds=50000000)  # 50ms deadline for detection
)

MAP_DATA_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
    deadline=Duration(seconds=0, nanoseconds=100000000)  # 100ms deadline for mapping
)

IMU_SENSOR_DATA_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
    deadline=Duration(seconds=0, nanoseconds=20000000)  # 20ms deadline
)

STATUS_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
    deadline=Duration(seconds=1)  # 1 second deadline
)


class BasePerceptionNode(Node):
    """
    Base class for all perception system nodes with standardized QoS settings.
    """

    def __init__(self, node_name: str, **kwargs):
        """
        Initialize the base perception node.

        Args:
            node_name: Name of the node
            **kwargs: Additional arguments for the node
        """
        super().__init__(node_name, **kwargs)

        # Store QoS profiles for reuse
        self.qos_profiles = {
            'real_time_sensor_data': REAL_TIME_SENSOR_DATA_QOS,
            'processed_data': PROCESSED_DATA_QOS,
            'map_data': MAP_DATA_QOS,
            'imu_sensor_data': IMU_SENSOR_DATA_QOS,
            'status': STATUS_QOS
        }

        # Initialize common parameters
        self._declare_common_parameters()

        # Log initialization
        self.get_logger().info(f"Initialized {node_name} with base perception functionality")

    def _declare_common_parameters(self):
        """
        Declare common parameters used across perception nodes.
        """
        # Sensor configuration parameters
        self.declare_parameter('sensors.camera.frequency', 30.0)
        self.declare_parameter('sensors.lidar.frequency', 10.0)
        self.declare_parameter('sensors.imu.frequency', 100.0)

        # Calibration parameters
        self.declare_parameter('calibration.auto_enabled', True)
        self.declare_parameter('calibration.manual_override', False)

        # Object detection parameters
        self.declare_parameter('object_detection.confidence_threshold', 0.5)
        self.declare_parameter('object_detection.model_path', 'models/yolo_humanoid_objects.pt')
        self.declare_parameter('object_detection.target_classes',
                              ['human', 'chair', 'table', 'door', 'stair', 'obstacle'])

        # Mapping parameters
        self.declare_parameter('mapping.resolution', 0.05)
        self.declare_parameter('mapping.max_range', 10.0)
        self.declare_parameter('mapping.update_frequency', 5.0)

        # Performance parameters
        self.declare_parameter('performance.real_time_enabled', True)
        self.declare_parameter('performance.thread_count', 4)

        # Security parameters
        self.declare_parameter('security.encryption_enabled', False)
        self.declare_parameter('security.encryption_password', '')

        self.get_logger().info("Declared common perception parameters")

    def get_qos_profile(self, profile_name: str) -> QoSProfile:
        """
        Get a QoS profile by name.

        Args:
            profile_name: Name of the QoS profile to retrieve

        Returns:
            QoSProfile: The requested QoS profile
        """
        if profile_name not in self.qos_profiles:
            self.get_logger().warning(f"Unknown QoS profile: {profile_name}, using default")
            return self.qos_profiles['real_time_sensor_data']

        return self.qos_profiles[profile_name]

    def create_publisher_with_profile(self, msg_type, topic: str, profile_name: str):
        """
        Create a publisher with a specific QoS profile.

        Args:
            msg_type: Message type for the publisher
            topic: Topic name
            profile_name: Name of the QoS profile to use

        Returns:
            Publisher: The created publisher
        """
        qos_profile = self.get_qos_profile(profile_name)
        return self.create_publisher(msg_type, topic, qos_profile)

    def create_subscription_with_profile(self, msg_type, topic: str, callback, profile_name: str):
        """
        Create a subscription with a specific QoS profile.

        Args:
            msg_type: Message type for the subscription
            topic: Topic name
            callback: Callback function for the subscription
            profile_name: Name of the QoS profile to use

        Returns:
            Subscription: The created subscription
        """
        qos_profile = self.get_qos_profile(profile_name)
        return self.create_subscription(msg_type, topic, callback, qos_profile)

    def get_sensor_frequency(self, sensor_type: str) -> float:
        """
        Get the configured frequency for a specific sensor type.

        Args:
            sensor_type: Type of sensor ('camera', 'lidar', 'imu')

        Returns:
            float: Configured frequency in Hz
        """
        param_name = f'sensors.{sensor_type}.frequency'
        return self.get_parameter(param_name).value

    def get_security_config(self) -> Dict[str, Any]:
        """
        Get security configuration parameters.

        Returns:
            Dict containing security configuration
        """
        return {
            'encryption_enabled': self.get_parameter('security.encryption_enabled').value,
            'encryption_password': self.get_parameter('security.encryption_password').value
        }

    def get_detection_config(self) -> Dict[str, Any]:
        """
        Get object detection configuration parameters.

        Returns:
            Dict containing detection configuration
        """
        return {
            'confidence_threshold': self.get_parameter('object_detection.confidence_threshold').value,
            'model_path': self.get_parameter('object_detection.model_path').value,
            'target_classes': self.get_parameter('object_detection.target_classes').value
        }

    def get_mapping_config(self) -> Dict[str, Any]:
        """
        Get mapping configuration parameters.

        Returns:
            Dict containing mapping configuration
        """
        return {
            'resolution': self.get_parameter('mapping.resolution').value,
            'max_range': self.get_parameter('mapping.max_range').value,
            'update_frequency': self.get_parameter('mapping.update_frequency').value
        }

    def get_performance_config(self) -> Dict[str, Any]:
        """
        Get performance configuration parameters.

        Returns:
            Dict containing performance configuration
        """
        return {
            'real_time_enabled': self.get_parameter('performance.real_time_enabled').value,
            'thread_count': self.get_parameter('performance.thread_count').value
        }

    def is_calibration_auto_enabled(self) -> bool:
        """
        Check if automatic calibration is enabled.

        Returns:
            bool: True if auto calibration is enabled, False otherwise
        """
        return self.get_parameter('calibration.auto_enabled').value

    def is_manual_calibration_override(self) -> bool:
        """
        Check if manual calibration override is enabled.

        Returns:
            bool: True if manual override is enabled, False otherwise
        """
        return self.get_parameter('calibration.manual_override').value


class PerceptionNodeManager:
    """
    Manager class for handling multiple perception nodes.
    """

    def __init__(self):
        self.nodes = {}
        self.active = False

    def add_node(self, node_name: str, node: BasePerceptionNode):
        """
        Add a perception node to the manager.

        Args:
            node_name: Name of the node
            node: BasePerceptionNode instance
        """
        self.nodes[node_name] = node

    def start_all_nodes(self):
        """
        Start all managed perception nodes.
        """
        self.active = True
        for node_name, node in self.nodes.items():
            node.get_logger().info(f"Starting perception node: {node_name}")

    def stop_all_nodes(self):
        """
        Stop all managed perception nodes.
        """
        self.active = False
        for node_name, node in self.nodes.items():
            node.get_logger().info(f"Stopping perception node: {node_name}")

    def get_node(self, node_name: str) -> Optional[BasePerceptionNode]:
        """
        Get a perception node by name.

        Args:
            node_name: Name of the node to retrieve

        Returns:
            BasePerceptionNode or None if not found
        """
        return self.nodes.get(node_name)

    def get_all_nodes(self) -> Dict[str, BasePerceptionNode]:
        """
        Get all managed perception nodes.

        Returns:
            Dict of node names to node instances
        """
        return self.nodes.copy()


def create_default_qos_profiles():
    """
    Create and return the default QoS profiles for the perception system.

    Returns:
        Dict of QoS profile names to QoSProfile objects
    """
    return {
        'real_time_sensor_data': REAL_TIME_SENSOR_DATA_QOS,
        'processed_data': PROCESSED_DATA_QOS,
        'map_data': MAP_DATA_QOS,
        'imu_sensor_data': IMU_SENSOR_DATA_QOS,
        'status': STATUS_QOS
    }


# Example usage
if __name__ == '__main__':
    rclpy.init()

    # Create a sample perception node
    node = BasePerceptionNode('sample_perception_node')

    # Get various QoS profiles
    sensor_qos = node.get_qos_profile('real_time_sensor_data')
    map_qos = node.get_qos_profile('map_data')

    print(f"Sensor QoS reliability: {sensor_qos.reliability}")
    print(f"Map QoS reliability: {map_qos.reliability}")

    # Get configuration
    detection_config = node.get_detection_config()
    print(f"Detection config: {detection_config}")

    node.destroy_node()
    rclpy.shutdown()