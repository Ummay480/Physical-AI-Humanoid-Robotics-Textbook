// @ts-check

/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  tutorialSidebar: [
    'intro/index',
    {
      type: 'category',
      label: 'Getting Started',
      items: ['getting-started/what-is-physical-ai'],
    },
    {
      type: 'category',
      label: 'Module 1: The Robotic Nervous System (ROS 2)',
      items: [
        'robotic-nervous-system/ros2-nodes-topics',
        'module-1-ros2-nervous-system/rclpy-ai-bridge',
        'module-1-ros2-nervous-system/humanoid-urdf'
      ],
    },
    {
      type: 'category',
      label: 'Module 2: The Digital Twin (Gazebo & Unity)',
      items: [
        'gazebo_physics_quickstart',
        'module-1-ros2-nervous-system/ros2-control'
      ],
    },
    {
      type: 'category',
      label: 'AI Control Systems',
      items: [
        'actuation_sensors_quickstart',
        'perception_quickstart'
      ],
    },
    {
      type: 'category',
      label: 'Applications',
      items: [
        'module-1-ros2-nervous-system/index'
      ],
    },
  ],
};

module.exports = sidebars;