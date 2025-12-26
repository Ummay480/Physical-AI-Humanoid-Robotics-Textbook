"""
Launch file for Isaac Perception Node
Implements launch configuration for T018: Create Perception ROS 2 Node
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Generate launch description for Isaac Perception Node."""

    # Declare launch arguments
    confidence_threshold_arg = DeclareLaunchArgument(
        'confidence_threshold',
        default_value='0.5',
        description='Minimum confidence threshold for detections'
    )

    enable_tracking_arg = DeclareLaunchArgument(
        'enable_tracking',
        default_value='false',
        description='Enable object tracking across frames'
    )

    enable_depth_processing_arg = DeclareLaunchArgument(
        'enable_depth_processing',
        default_value='false',
        description='Enable depth processing for 3D position estimation'
    )

    publish_visualization_arg = DeclareLaunchArgument(
        'publish_visualization',
        default_value='true',
        description='Publish visualization images with bounding boxes'
    )

    # Get launch configurations
    confidence_threshold = LaunchConfiguration('confidence_threshold')
    enable_tracking = LaunchConfiguration('enable_tracking')
    enable_depth_processing = LaunchConfiguration('enable_depth_processing')
    publish_visualization = LaunchConfiguration('publish_visualization')

    # Create perception node
    perception_node = Node(
        package='isaac_ros',
        executable='perception_node',
        name='isaac_perception',
        parameters=[
            {
                'confidence_threshold': confidence_threshold,
                'enable_tracking': enable_tracking,
                'enable_depth_processing': enable_depth_processing,
                'publish_visualization': publish_visualization
            }
        ],
        remappings=[
            ('/isaac_sim/camera/rgb/image_raw', '/camera/rgb/image_raw'),
            ('/isaac_sim/camera/depth/image_raw', '/camera/depth/image_raw'),
            ('/isaac_perception/detections', '/perception/detections'),
            ('/isaac_perception/status', '/perception/status'),
            ('/isaac_perception/visualization', '/perception/visualization')
        ],
        output='screen'
    )

    # Return launch description
    return LaunchDescription([
        confidence_threshold_arg,
        enable_tracking_arg,
        enable_depth_processing_arg,
        publish_visualization_arg,
        perception_node
    ])