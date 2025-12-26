# Real-time Constraints in ROS 2 Systems

## Introduction to Real-time Systems in Robotics

Real-time systems in robotics are those that must respond to inputs or complete tasks within strict time constraints. In robot control applications, meeting these time constraints is critical for safety, stability, and performance. Unlike general-purpose computing systems where delays might simply cause inconvenience, in robotics, missing real-time deadlines can result in unstable control, safety violations, or physical damage to the robot or its environment.

## Types of Real-time Constraints

### Hard Real-time Constraints
- Missing a deadline is considered a system failure
- Examples: Emergency stop responses, collision avoidance
- Typically require specialized real-time operating systems

### Soft Real-time Constraints
- Missing a deadline degrades performance but doesn't cause failure
- Examples: Control loop updates, sensor data processing
- Most ROS 2 robot control applications fall into this category

### Firm Real-time Constraints
- Results are useless if they arrive after the deadline
- Examples: Image processing for navigation, trajectory updates
- Results should not be used if late

## Real-time Requirements for Robot Control

### Control Loop Timing
Robot control systems typically require:
- **High-frequency control loops**: 100Hz-1kHz depending on the application
- **Predictable timing**: Consistent update intervals with minimal jitter
- **Low latency**: Minimal delay between sensor input and control output

### Typical Timing Requirements
- **Joint position control**: 100-500Hz (2-10ms update periods)
- **Balance control for humanoid robots**: 200-1000Hz (1-5ms update periods)
- **Sensor feedback processing**: \<20ms latency
- **Planning updates**: 1-20Hz depending on complexity

## Quality of Service (QoS) for Real-time Communication

ROS 2 provides Quality of Service (QoS) profiles to help meet real-time requirements:

### Reliability Policy
- **RELIABLE**: All messages are guaranteed to be delivered (may block)
- **BEST_EFFORT**: Messages may be dropped, but no blocking (better for real-time)

```python
from rclpy.qos import QoSProfile, ReliabilityPolicy

# For sensor data (best effort to avoid blocking)
sensor_qos = QoSProfile(
    depth=10,
    reliability=ReliabilityPolicy.BEST_EFFORT
)

# For critical control commands (reliable delivery)
control_qos = QoSProfile(
    depth=10,
    reliability=ReliabilityPolicy.RELIABLE
)
```

### Durability Policy
- **VOLATILE**: No message persistence (lower latency)
- **TRANSIENT_LOCAL**: Messages persisted for late-joining subscribers (higher latency)

### History Policy
- **KEEP_LAST**: Keep only the most recent messages
- **KEEP_ALL**: Keep all messages (use with caution due to memory usage)

## Implementing Real-time Capable Nodes

### Proper Timer Usage
Use ROS 2 timers for consistent control loop timing:

```python
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile

class RealTimeController(Node):
    def __init__(self):
        super().__init__('real_time_controller')

        # Set timer for 100Hz control loop (0.01 seconds)
        self.control_timer = self.create_timer(
            0.01,  # 10ms period = 100Hz
            self.control_loop
        )

        # Publishers and subscribers with appropriate QoS
        self.sensor_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.sensor_callback,
            QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        )

        self.command_pub = self.create_publisher(
            JointTrajectory,
            '/joint_trajectory_controller/joint_trajectory',
            QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE)
        )

        self.last_sensor_update = self.get_clock().now()

    def control_loop(self):
        """Main control loop - should execute at consistent intervals"""
        current_time = self.get_clock().now()
        dt = (current_time - self.last_sensor_update).nanoseconds / 1e9

        # Check for sensor timeout
        if dt > 0.1:  # 100ms timeout
            self.get_logger().warn('Sensor timeout detected')
            return

        # Apply control algorithm
        control_commands = self.compute_control_commands()

        # Publish commands
        self.publish_control_commands(control_commands)

        self.last_sensor_update = current_time
```

### Avoiding Blocking Operations
Real-time nodes should avoid operations that can block execution:

```python
# DON'T: File I/O in control loop
def control_loop(self):
    with open('data.txt', 'w') as f:  # BLOCKING!
        f.write(str(self.robot_state))

# DO: Use non-blocking alternatives or separate threads
def control_loop(self):
    # Process control logic without blocking I/O
    self.process_control_logic()

    # Schedule non-critical I/O for later
    self.get_logger().info('Control cycle completed')
```

## Real-time Performance Monitoring

### Timing Analysis
Monitor your system's timing performance:

```python
import time
from collections import deque

class TimingMonitor:
    def __init__(self, window_size=100):
        self.period_times = deque(maxlen=window_size)
        self.execution_times = deque(maxlen=window_size)

    def start_period(self):
        self.period_start = time.time()

    def end_period(self):
        if hasattr(self, 'period_start'):
            period_time = time.time() - self.period_start
            self.period_times.append(period_time)

    def start_execution(self):
        self.execution_start = time.time()

    def end_execution(self):
        if hasattr(self, 'execution_start'):
            execution_time = time.time() - self.execution_start
            self.execution_times.append(execution_time)

    def get_stats(self):
        if self.period_times:
            avg_period = sum(self.period_times) / len(self.period_times)
            min_period = min(self.period_times)
            max_period = max(self.period_times)

            avg_execution = sum(self.execution_times) / len(self.execution_times) if self.execution_times else 0

            return {
                'avg_period': avg_period,
                'min_period': min_period,
                'max_period': max_period,
                'avg_execution': avg_execution,
                'jitter': max_period - min_period
            }
        return {}
```

### Performance Metrics for Robot Control
Key metrics to monitor:

```python
class PerformanceMonitor(Node):
    def __init__(self):
        super().__init__('performance_monitor')

        # Create timer for performance logging
        self.perf_timer = self.create_timer(1.0, self.log_performance)  # Every second

        self.timing_monitor = TimingMonitor()
        self.message_counts = {}

    def log_performance(self):
        stats = self.timing_monitor.get_stats()

        if stats:
            self.get_logger().info(
                f'Performance: '
                f'Avg Period: {stats["avg_period"]*1000:.2f}ms, '
                f'Jitter: {stats["jitter"]*1000:.2f}ms, '
                f'Avg Execution: {stats["avg_execution"]*1000:.2f}ms'
            )

            # Check if timing requirements are met
            if stats['avg_period'] > 0.011:  # 11ms for 100Hz target
                self.get_logger().warn('Timing requirement violated')
```

## Real-time Scheduling in Linux

### CPU Affinity
Pin real-time threads to specific CPU cores:

```python
import os
import psutil

def set_cpu_affinity(core_id):
    """Set CPU affinity to reduce context switching"""
    p = psutil.Process(os.getpid())
    p.cpu_affinity([core_id])

def set_realtime_priority():
    """Set real-time priority (requires root privileges)"""
    import resource
    # Set to SCHED_FIFO with maximum priority
    os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(99))
```

### Memory Management
Prevent page faults that can cause timing delays:

```python
import mmap

def lock_memory():
    """Lock current memory pages to prevent swapping"""
    # This is a simplified example; in practice, use mlock or similar
    pass
```

## Real-time Considerations for Multi-node Systems

### Clock Synchronization
Ensure consistent timing across nodes:

```python
from rclpy.time import Time
from builtin_interfaces.msg import Time as TimeMsg

def get_synchronized_time(self):
    """Get time from ROS 2 clock"""
    return self.get_clock().now()

def compare_times(self, time1, time2):
    """Compare ROS 2 times properly"""
    return (time1.nanoseconds - time2.nanoseconds) / 1e9
```

### Message Timing Analysis
Monitor communication delays:

```python
class MessageTimingAnalyzer:
    def __init__(self, node):
        self.node = node
        self.sent_times = {}

    def on_message_sent(self, message_id):
        """Record time when message is sent"""
        self.sent_times[message_id] = self.node.get_clock().now()

    def on_message_received(self, message_id):
        """Calculate round-trip time when message is received"""
        if message_id in self.sent_times:
            sent_time = self.sent_times[message_id]
            received_time = self.node.get_clock().now()
            rtt = (received_time - sent_time).nanoseconds / 1e9
            self.node.get_logger().info(f'Message RTT: {rtt*1000:.2f}ms')
            del self.sent_times[message_id]
```

## Best Practices for Real-time ROS 2 Systems

### 1. Design for Deterministic Behavior
- Use fixed-size data structures when possible
- Avoid dynamic memory allocation in control loops
- Minimize branching in time-critical code

### 2. Proper Resource Management
- Monitor CPU usage and set appropriate limits
- Use dedicated threads for time-critical operations
- Consider using real-time Linux kernel for critical applications

### 3. Testing and Validation
- Test timing behavior under load
- Validate performance with realistic sensor data rates
- Monitor for timing violations during operation

### 4. Error Handling
- Implement graceful degradation when timing constraints are violated
- Include timeout mechanisms for critical operations
- Log timing violations for debugging

## Common Real-time Issues and Solutions

### Issue 1: Control Loop Jitter
**Problem**: Inconsistent timing in control loops
**Solution**: Use dedicated timers, minimize other operations in loop, consider real-time kernel

### Issue 2: Message Queue Overflow
**Problem**: Sensor data queues filling up, causing delays
**Solution**: Use appropriate QoS settings, process data immediately, implement data dropping policies

### Issue 3: Context Switching Overhead
**Problem**: OS scheduling interfering with timing
**Solution**: Use CPU affinity, real-time scheduling, minimize system calls in control loops

## Real-time Simulation Considerations

When testing real-time systems in simulation:

- Configure Gazebo/other simulators to match real-time requirements
- Use appropriate physics update rates
- Account for simulation timing differences when deploying to hardware

## Summary

Real-time constraints are fundamental to safe and effective robot control systems. While ROS 2 is not a hard real-time framework, it provides tools and patterns to achieve soft real-time performance suitable for most robot control applications. Key considerations include proper QoS configuration, consistent control loop timing, performance monitoring, and resource management. Understanding these concepts is essential for building robot control systems that perform reliably under timing constraints, especially for humanoid robots that require precise, coordinated motion control for balance and stability.