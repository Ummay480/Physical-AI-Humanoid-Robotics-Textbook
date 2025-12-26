# Launch file for Isaac ROS Visual SLAM
# This is a template launch file as mentioned in T002 task

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Generate launch description for Isaac ROS Visual SLAM."""

    # Declare launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time')

    # Declare launch argument
    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation clock if true'
    )

    # Placeholder nodes for Isaac ROS Visual SLAM
    # These would be replaced with actual Isaac ROS nodes after installation
    visual_slam_node = Node(
        package='isaac_ros_visual_slam',
        executable='visual_slam_node',
        name='visual_slam_node',
        parameters=[
            {'use_sim_time': use_sim_time},
            # Additional parameters would go here
        ],
        remappings=[
            ('/camera/color/image_rect', '/camera/color/image_raw'),
            ('/camera/depth/image_rect', '/camera/depth/image_raw'),
        ],
        output='screen'
    )

    return LaunchDescription([
        declare_use_sim_time,
        visual_slam_node,
    ])