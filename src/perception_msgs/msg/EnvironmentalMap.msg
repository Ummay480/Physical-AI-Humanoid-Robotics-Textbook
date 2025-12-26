# Represents the 3D environmental map
string id                  # Unique identifier for the map
octomap_msgs/Octomap map_data  # 3D occupancy grid data
float64 resolution         # Spatial resolution of the map
string origin_frame        # Reference frame for the map
builtin_interfaces/Time update_timestamp  # Last update time
geometry_msgs/Polygon coverage_area  # Bounding box of mapped area