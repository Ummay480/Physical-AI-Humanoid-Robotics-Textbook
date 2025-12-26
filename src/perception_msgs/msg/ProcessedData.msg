# Represents sensor data after initial processing
string source_data_id       # Reference to the original SensorData
builtin_interfaces/Time timestamp  # Processing completion time
string processed_payload    # Processed data (detected objects, features, etc.)
float64 confidence_score    # Confidence level of the processing results
string processing_method    # Algorithm or method used for processing