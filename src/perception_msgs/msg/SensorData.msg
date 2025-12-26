# Represents raw data from a specific sensor type
string sensor_type           # Type of sensor (camera, lidar, imu, etc.)
builtin_interfaces/Time timestamp  # Acquisition time
string frame_id             # Coordinate frame identifier
float64[] data              # Raw sensor data payload
string metadata             # Additional sensor-specific metadata