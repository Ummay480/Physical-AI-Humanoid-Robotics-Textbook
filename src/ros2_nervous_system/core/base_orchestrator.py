"""
Base Orchestrator Node for ROS 2 Nervous System.

Provides common functionality for all orchestration nodes including QoS profile management,
parameter declarations, and utility methods.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from typing import Dict, Any, Optional
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from ros2_nervous_system.interfaces.qos_profiles import get_qos_profiles, get_qos_profile


class BaseOrchestratorNode(Node):
    """
    Base class for all orchestration nodes in the Nervous System.

    Provides:
    - QoS profile management
    - Common parameter declarations
    - Utility methods for subscriptions/publications
    - Configuration access methods
    """

    def __init__(self, node_name: str):
        """
        Initialize base orchestrator node.

        Args:
            node_name: Name of the node
        """
        super().__init__(node_name)

        # Setup QoS profiles
        self.qos_profiles = get_qos_profiles()

        # Declare common parameters
        self._declare_common_parameters()

        self.get_logger().info(f"{node_name} initialized")

    def _declare_common_parameters(self):
        """Declare common ROS 2 parameters for orchestration."""

        # Orchestration parameters
        self.declare_parameter('orchestration.update_frequency', 50.0)
        self.declare_parameter('orchestration.health_check_frequency', 2.0)
        self.declare_parameter('orchestration.diagnostics_frequency', 1.0)

        # Timing budgets (in milliseconds)
        self.declare_parameter('timing.latency_budgets.perception_to_nervous', 5.0)
        self.declare_parameter('timing.latency_budgets.nervous_to_navigation', 10.0)
        self.declare_parameter('timing.latency_budgets.nervous_to_control', 5.0)
        self.declare_parameter('timing.latency_budgets.end_to_end', 50.0)

        # Frequency requirements
        self.declare_parameter('timing.frequency_requirements.perception_map.target', 5.0)
        self.declare_parameter('timing.frequency_requirements.perception_map.tolerance', 1.0)
        self.declare_parameter('timing.frequency_requirements.perception_camera.target', 30.0)
        self.declare_parameter('timing.frequency_requirements.perception_camera.tolerance', 5.0)
        self.declare_parameter('timing.frequency_requirements.perception_lidar.target', 10.0)
        self.declare_parameter('timing.frequency_requirements.perception_lidar.tolerance', 2.0)
        self.declare_parameter('timing.frequency_requirements.perception_imu.target', 100.0)
        self.declare_parameter('timing.frequency_requirements.perception_imu.tolerance', 10.0)

        # Health monitoring
        self.declare_parameter('health.heartbeat_timeout', 1.0)
        self.declare_parameter('health.stall_detection_threshold', 2.0)
        self.declare_parameter('health.subsystems', ['perception', 'navigation', 'control'])

        # Data quality
        self.declare_parameter('data_quality.min_confidence_threshold', 0.5)
        self.declare_parameter('data_quality.max_age_threshold', 0.5)

        # Topic configuration
        self.declare_parameter('topics.perception_objects', '/perception/objects_detected')
        self.declare_parameter('topics.perception_map', '/perception/environmental_map')
        self.declare_parameter('topics.perception_fused_data', '/perception/fused_data')
        self.declare_parameter('topics.perception_health', '/perception/system_status')
        self.declare_parameter('topics.perception_camera', '/perception/camera_front/image_processed')
        self.declare_parameter('topics.perception_lidar', '/perception/lidar_3d/scan_processed')
        self.declare_parameter('topics.perception_imu', '/perception/imu_main/data_processed')

        # Output topics
        self.declare_parameter('topics.perception_state', '/nervous_system/perception_state')
        self.declare_parameter('topics.navigation_input', '/nervous_system/navigation_input')
        self.declare_parameter('topics.control_input', '/nervous_system/control_input')
        self.declare_parameter('topics.system_health', '/nervous_system/system_health')
        self.declare_parameter('topics.diagnostics', '/nervous_system/diagnostics')

    # QoS Profile Management

    def get_qos_profile(self, profile_name: str) -> QoSProfile:
        """
        Get QoS profile by name.

        Args:
            profile_name: Name of the profile

        Returns:
            QoSProfile object

        Raises:
            KeyError: If profile name not found
        """
        if profile_name not in self.qos_profiles:
            self.get_logger().error(f"Unknown QoS profile: {profile_name}")
            raise KeyError(f"Unknown QoS profile: {profile_name}")

        return self.qos_profiles[profile_name]

    def create_publisher_with_profile(self, msg_type, topic: str, profile_name: str):
        """
        Create publisher with named QoS profile.

        Args:
            msg_type: Message type class
            topic: Topic name
            profile_name: QoS profile name

        Returns:
            Publisher object
        """
        qos_profile = self.get_qos_profile(profile_name)
        publisher = self.create_publisher(msg_type, topic, qos_profile)
        self.get_logger().debug(f"Created publisher on {topic} with {profile_name} QoS")
        return publisher

    def create_subscription_with_profile(self, msg_type, topic: str, callback, profile_name: str):
        """
        Create subscription with named QoS profile.

        Args:
            msg_type: Message type class
            topic: Topic name
            callback: Callback function
            profile_name: QoS profile name

        Returns:
            Subscription object
        """
        qos_profile = self.get_qos_profile(profile_name)
        subscription = self.create_subscription(msg_type, topic, callback, qos_profile)
        self.get_logger().debug(f"Created subscription to {topic} with {profile_name} QoS")
        return subscription

    # Parameter Access Methods

    def get_orchestration_config(self) -> Dict[str, float]:
        """
        Get orchestration configuration parameters.

        Returns:
            Dictionary with orchestration config
        """
        return {
            'update_frequency': self.get_parameter('orchestration.update_frequency').value,
            'health_check_frequency': self.get_parameter('orchestration.health_check_frequency').value,
            'diagnostics_frequency': self.get_parameter('orchestration.diagnostics_frequency').value,
        }

    def get_latency_budgets(self) -> Dict[str, float]:
        """
        Get latency budget configuration.

        Returns:
            Dictionary with latency budgets in milliseconds
        """
        return {
            'perception_to_nervous': self.get_parameter('timing.latency_budgets.perception_to_nervous').value,
            'nervous_to_navigation': self.get_parameter('timing.latency_budgets.nervous_to_navigation').value,
            'nervous_to_control': self.get_parameter('timing.latency_budgets.nervous_to_control').value,
            'end_to_end': self.get_parameter('timing.latency_budgets.end_to_end').value,
        }

    def get_frequency_requirements(self) -> Dict[str, Dict[str, float]]:
        """
        Get frequency requirement configuration.

        Returns:
            Dictionary with frequency requirements (target and tolerance)
        """
        return {
            'perception_map': {
                'target': self.get_parameter('timing.frequency_requirements.perception_map.target').value,
                'tolerance': self.get_parameter('timing.frequency_requirements.perception_map.tolerance').value,
            },
            'perception_camera': {
                'target': self.get_parameter('timing.frequency_requirements.perception_camera.target').value,
                'tolerance': self.get_parameter('timing.frequency_requirements.perception_camera.tolerance').value,
            },
            'perception_lidar': {
                'target': self.get_parameter('timing.frequency_requirements.perception_lidar.target').value,
                'tolerance': self.get_parameter('timing.frequency_requirements.perception_lidar.tolerance').value,
            },
            'perception_imu': {
                'target': self.get_parameter('timing.frequency_requirements.perception_imu.target').value,
                'tolerance': self.get_parameter('timing.frequency_requirements.perception_imu.tolerance').value,
            },
        }

    def get_health_config(self) -> Dict[str, Any]:
        """
        Get health monitoring configuration.

        Returns:
            Dictionary with health config
        """
        return {
            'heartbeat_timeout': self.get_parameter('health.heartbeat_timeout').value,
            'stall_detection_threshold': self.get_parameter('health.stall_detection_threshold').value,
            'subsystems': self.get_parameter('health.subsystems').value,
        }

    def get_data_quality_config(self) -> Dict[str, float]:
        """
        Get data quality configuration.

        Returns:
            Dictionary with data quality thresholds
        """
        return {
            'min_confidence_threshold': self.get_parameter('data_quality.min_confidence_threshold').value,
            'max_age_threshold': self.get_parameter('data_quality.max_age_threshold').value,
        }

    def get_topic_config(self) -> Dict[str, str]:
        """
        Get topic configuration.

        Returns:
            Dictionary mapping topic names to actual topic paths
        """
        return {
            # Input topics (from Perception)
            'perception_objects': self.get_parameter('topics.perception_objects').value,
            'perception_map': self.get_parameter('topics.perception_map').value,
            'perception_fused_data': self.get_parameter('topics.perception_fused_data').value,
            'perception_health': self.get_parameter('topics.perception_health').value,
            'perception_camera': self.get_parameter('topics.perception_camera').value,
            'perception_lidar': self.get_parameter('topics.perception_lidar').value,
            'perception_imu': self.get_parameter('topics.perception_imu').value,

            # Output topics
            'perception_state': self.get_parameter('topics.perception_state').value,
            'navigation_input': self.get_parameter('topics.navigation_input').value,
            'control_input': self.get_parameter('topics.control_input').value,
            'system_health': self.get_parameter('topics.system_health').value,
            'diagnostics': self.get_parameter('topics.diagnostics').value,
        }

    def shutdown(self):
        """Gracefully shutdown the node."""
        self.get_logger().info(f"Shutting down {self.get_name()}")
        # Subclasses can override to add cleanup logic


def main(args=None):
    """Test main function for BaseOrchestratorNode."""
    rclpy.init(args=args)

    try:
        node = BaseOrchestratorNode('test_orchestrator_node')
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if rclpy.ok():
            node.shutdown()
            node.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    main()
