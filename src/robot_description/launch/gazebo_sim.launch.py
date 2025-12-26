import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, ExecuteProcess, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration, Command
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # Get package directories
    pkg_share = get_package_share_directory('robot_description')
    gazebo_ros_share = get_package_share_directory('gazebo_ros')

    # File paths
    urdf_file = os.path.join(pkg_share, 'urdf', 'humanoid.urdf.xacro')
    world_file = os.path.join(pkg_share, 'worlds', 'humanoid_world.world')
    controller_config = os.path.join(pkg_share, 'config', 'humanoid_controllers.yaml')

    # Launch configurations
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    world = LaunchConfiguration('world', default=world_file)

    # Process the URDF using xacro
    robot_description = Command(['xacro ', urdf_file])

    # Declare launch arguments
    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation (Gazebo) clock if true'
    )

    declare_world_arg = DeclareLaunchArgument(
        'world',
        default_value=world_file,
        description='Full path to world file to load'
    )

    # Gazebo server (gzserver)
    gzserver = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(gazebo_ros_share, 'launch', 'gzserver.launch.py')
        ),
        launch_arguments={'world': world, 'verbose': 'true'}.items()
    )

    # Gazebo client (gzclient)
    gzclient = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(gazebo_ros_share, 'launch', 'gzclient.launch.py')
        )
    )

    # Robot State Publisher node
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[
            {'use_sim_time': use_sim_time},
            {'robot_description': robot_description}
        ]
    )

    # Spawn the robot in Gazebo
    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        name='spawn_humanoid',
        output='screen',
        arguments=[
            '-topic', 'robot_description',
            '-entity', 'humanoid_robot',
            '-x', '0.0',
            '-y', '0.0',
            '-z', '1.5',  # Spawn at 1.5m height to see falling physics
            '-R', '0.0',
            '-P', '0.0',
            '-Y', '0.0'
        ],
        parameters=[{'use_sim_time': use_sim_time}]
    )

    # Joint State Broadcaster Spawner
    joint_state_broadcaster_spawner = Node(
        package='controller_manager',
        executable='spawner',
        name='joint_state_broadcaster_spawner',
        output='screen',
        arguments=[
            'joint_state_broadcaster',
            '--controller-manager', '/controller_manager'
        ],
        parameters=[{'use_sim_time': use_sim_time}]
    )

    # Position Controller Spawner
    position_controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        name='position_controller_spawner',
        output='screen',
        arguments=[
            'position_controller',
            '--controller-manager', '/controller_manager'
        ],
        parameters=[{'use_sim_time': use_sim_time}]
    )

    # Register event handler to spawn controllers after robot is spawned
    spawn_controllers_after_robot = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=spawn_entity,
            on_exit=[joint_state_broadcaster_spawner, position_controller_spawner],
        )
    )

    return LaunchDescription([
        declare_use_sim_time,
        declare_world_arg,
        gzserver,
        gzclient,
        robot_state_publisher,
        spawn_entity,
        spawn_controllers_after_robot
    ])
