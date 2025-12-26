---
sidebar_position: 1
title: ROS 2 Nodes, Topics, and Services
---

# ROS 2: The Robotic Nervous System

ROS 2 (Robot Operating System 2) serves as the "nervous system" for robotic platforms, providing the infrastructure for communication, coordination, and control. Understanding ROS 2 is fundamental to developing complex robotic systems.

## Core Concepts

### Nodes
Nodes are the fundamental building blocks of a ROS 2 system. Each node is a process that performs specific computation and can communicate with other nodes through topics, services, and actions.

```python
import rclpy
from rclpy.node import Node

class MinimalPublisher(Node):
    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher = self.create_publisher(String, 'topic', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = 'Hello World: %d' % self.i
        self.publisher.publish(msg)
        self.get_logger().info('Publishing: "%s"' % msg.data)
        self.i += 1
```

### Topics
Topics enable asynchronous communication between nodes using a publish-subscribe model. Publishers send messages to topics, and subscribers receive messages from topics.

### Services
Services provide synchronous request-response communication between nodes. A service client sends a request and waits for a response from a service server.

## Setting Up ROS 2 for Humanoid Robots

For humanoid robots, ROS 2 provides the communication backbone that connects:

- **Sensor nodes**: IMU, cameras, force/torque sensors
- **Control nodes**: Joint controllers, walking pattern generators
- **Perception nodes**: Object detection, SLAM, vision processing
- **Planning nodes**: Path planning, motion planning, task planning
- **Behavior nodes**: State machines, decision-making systems

## Best Practices

1. **Modularity**: Design nodes to have single, well-defined responsibilities
2. **Standard Interfaces**: Use standard message types when possible
3. **Configuration**: Externalize parameters for easy system customization
4. **Monitoring**: Implement proper logging and diagnostics
5. **Real-time Considerations**: Use appropriate QoS policies for timing-critical communications

## Integration with AI Systems

ROS 2 serves as the bridge between traditional robotic control and AI systems:

- AI perception nodes can publish processed data to topics
- AI planning nodes can receive sensor data and publish commands
- Machine learning models can be integrated as ROS 2 nodes
- Data collection for training can be facilitated through ROS 2 topics

This architecture allows for the development of sophisticated Physical AI systems where traditional robotics and artificial intelligence work in harmony.