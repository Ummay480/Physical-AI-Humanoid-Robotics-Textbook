import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.substitutions import Command
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # Get the package share directory
    pkg_share = get_package_share_directory('robot_description')

    # Define the URDF file path
    urdf_file = os.path.join(pkg_share, 'urdf', 'humanoid.urdf.xacro')

    # Get the URDF content using xacro
    robot_description = Command(['xacro ', urdf_file])

    # Launch configuration for use_sim_time
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')

    # Joint state publisher node (publishes joint states for visualization)
    joint_state_publisher_node = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        name='joint_state_publisher',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}]
    )

    # Robot state publisher node (publishes tf transforms)
    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[
            {'use_sim_time': use_sim_time},
            {'robot_description': robot_description}
        ]
    )

    # RViz2 node for visualization
    rviz_config_file = os.path.join(pkg_share, 'rviz', 'robot_view.rviz')

    # Create rviz directory and config if they don't exist
    rviz_dir = os.path.join(pkg_share, 'rviz')
    if not os.path.exists(rviz_dir):
        os.makedirs(rviz_dir)

    # Create a basic RViz config file if it doesn't exist
    rviz_config_path = os.path.join(rviz_dir, 'robot_view.rviz')
    if not os.path.exists(rviz_config_path):
        with open(rviz_config_path, 'w') as f:
            f.write("""Panels:
  - Class: rviz_common/Displays
    Name: Displays
  - Class: rviz_common/Views
    Name: Views
Visualization Manager:
  Displays:
    - Class: rviz_default_plugins/Grid
      Name: Grid
    - Class: rviz_default_plugins/RobotModel
      Name: RobotModel
      Description: Robot Model
      Enabled: true
      Topic:
        Depth: 5
        Durability Policy: Volatile
        History Policy: Keep Last
        Reliability Policy: Reliable
        Value: /robot_description
  Global Options:
    Fixed Frame: base_link
    Frame Rate: 30
  Name: root
  Views:
    Current:
      Class: rviz_default_plugins/Orbit
      Name: Current View
""")

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config_path],
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}]
    )

    # Return the launch description
    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation (Gazebo) clock if true'
        ),
        joint_state_publisher_node,
        robot_state_publisher_node,
        rviz_node
    ])