cmake_minimum_required(VERSION 3.8)
project(perception_msgs)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(octomap_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# Generate messages, services, and actions
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/SensorData.msg"
  "msg/ProcessedData.msg"
  "msg/DetectedObject.msg"
  "msg/EnvironmentalMap.msg"
  "srv/Calibration.srv"
  "srv/ObjectDetection.srv"
  "action/PerceptionTask.action"
  DEPENDENCIES builtin_interfaces geometry_msgs sensor_msgs std_msgs octomap_msgs
)

ament_package()