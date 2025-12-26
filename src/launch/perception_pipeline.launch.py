"""
Launch file for the complete perception pipeline.

This launch file starts all perception system components:
- Sensor acquisition node (camera, LIDAR, IMU)
- Object detection node (YOLO-based)
- Sensor fusion node (mapping and fusion)
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, LogInfo
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node, PushRosNamespace
import os


def generate_launch_description():
    """Generate launch description for perception pipeline."""

    # Declare launch arguments
    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation time'
    )

    declare_enable_visualization = DeclareLaunchArgument(
        'enable_visualization',
        default_value='false',
        description='Enable visualization output'
    )

    declare_config_dir = DeclareLaunchArgument(
        'config_dir',
        default_value='config',
        description='Directory containing sensor configuration files'
    )

    declare_model_path = DeclareLaunchArgument(
        'model_path',
        default_value='models/yolov8n.pt',
        description='Path to YOLO model file'
    )

    declare_camera_topics = DeclareLaunchArgument(
        'camera_topics',
        default_value='[/camera_front/image_raw]',
        description='List of camera topics to subscribe to'
    )

    declare_lidar_topics = DeclareLaunchArgument(
        'lidar_topics',
        default_value='[/lidar_3d/scan]',
        description='List of LIDAR topics to subscribe to'
    )

    declare_imu_topics = DeclareLaunchArgument(
        'imu_topics',
        default_value='[/imu/data]',
        description='List of IMU topics to subscribe to'
    )

    # Get launch configurations
    use_sim_time = LaunchConfiguration('use_sim_time')
    enable_visualization = LaunchConfiguration('enable_visualization')
    config_dir = LaunchConfiguration('config_dir')
    model_path = LaunchConfiguration('model_path')
    camera_topics = LaunchConfiguration('camera_topics')
    lidar_topics = LaunchConfiguration('lidar_topics')
    imu_topics = LaunchConfiguration('imu_topics')

    # Sensor Acquisition Node
    sensor_acquisition_node = Node(
        package='perception',
        executable='sensor_acquisition_node.py',
        name='sensor_acquisition_node',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'config_dir': config_dir,
            'sensors_to_load': ['camera_front', 'lidar_3d', 'imu_main'],
            # Sensor frequencies
            'sensors.camera.frequency': 30.0,
            'sensors.lidar.frequency': 10.0,
            'sensors.imu.frequency': 100.0,
            # Performance
            'performance.real_time_enabled': True,
            'performance.thread_count': 4,
            # Security
            'security.encryption_enabled': False,
        }],
        remappings=[
            ('/camera_front/image_raw', '/camera_front/image_raw'),
            ('/lidar_3d/scan', '/lidar_3d/scan'),
            ('/imu/data', '/imu/data'),
        ]
    )

    # Object Detection Node
    object_detection_node = Node(
        package='perception',
        executable='object_detection_node.py',
        name='object_detection_node',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'object_detection.model_path': model_path,
            'object_detection.confidence_threshold': 0.5,
            'object_detection.target_classes': [
                'human', 'chair', 'table', 'door', 'stair', 'obstacle'
            ],
            # Camera parameters (should match calibration)
            'camera.fx': 525.0,
            'camera.fy': 525.0,
            'camera.cx': 319.5,
            'camera.cy': 239.5,
            # Preprocessing
            'preprocessing.enhance': True,
            'preprocessing.denoise': False,
            # Visualization
            'visualization.enabled': enable_visualization,
            # Camera topics to subscribe to
            'camera_topics': camera_topics,
        }],
        remappings=[
            ('/perception/camera_front/image_processed',
             '/perception/camera_front/image_processed'),
        ]
    )

    # Sensor Fusion Node
    sensor_fusion_node = Node(
        package='perception',
        executable='sensor_fusion_node.py',
        name='sensor_fusion_node',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            # Mapping configuration
            'mapping.resolution': 0.05,
            'mapping.max_range': 10.0,
            'mapping.update_frequency': 5.0,
            # Fusion configuration
            'fusion.sensor_weights': {
                'camera': 0.3,
                'lidar': 0.5,
                'imu': 0.2
            },
            'fusion.sync_tolerance': 0.1,
            'fusion.min_sensors': 2,
            # Topics
            'lidar_topics': lidar_topics,
            'imu_topics': imu_topics,
        }],
    )

    # Group all nodes under perception namespace
    perception_group = GroupAction([
        PushRosNamespace('perception'),
        sensor_acquisition_node,
        object_detection_node,
        sensor_fusion_node,
    ])

    # Log information
    log_info = LogInfo(
        msg='Starting Perception Pipeline with the following configuration:'
    )

    return LaunchDescription([
        # Declare arguments
        declare_use_sim_time,
        declare_enable_visualization,
        declare_config_dir,
        declare_model_path,
        declare_camera_topics,
        declare_lidar_topics,
        declare_imu_topics,

        # Log and launch
        log_info,
        perception_group,
    ])
