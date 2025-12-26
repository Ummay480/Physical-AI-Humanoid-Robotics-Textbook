# Represents an object detected by the computer vision system
string id                  # Unique identifier for the detected object
string object_type         # Type of object (human, furniture, door, etc.)
geometry_msgs/Point position  # 3D position relative to robot
geometry_msgs/Polygon bounding_box  # 2D/3D bounding box coordinates
float64 confidence_score   # Detection confidence level
builtin_interfaces/Time timestamp  # Time of detection