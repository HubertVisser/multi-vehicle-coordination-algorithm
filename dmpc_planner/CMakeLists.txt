cmake_minimum_required(VERSION 3.8)
project(dmpc_planner)

# Use ament_python to manage Python-based packages
find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)  # Required to use rclpy (Python client library for ROS 2)

# Install Python modules
ament_python_install_package(${PROJECT_NAME})

# Install Python scripts (nodes)
install(PROGRAMS
  src/dmpc_planner/jackalsimulator_mpc.py
  DESTINATION lib/${PROJECT_NAME}
)

# Export dependencies
ament_export_dependencies(rclpy)

# Register package
ament_package()
