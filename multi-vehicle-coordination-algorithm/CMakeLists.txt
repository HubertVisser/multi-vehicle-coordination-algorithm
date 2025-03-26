cmake_minimum_required(VERSION 3.0.2)
project(multi-vehicle-coordination-algorithm)

find_package(catkin REQUIRED COMPONENTS
  rospy
  dynamic_reconfigure
)

catkin_python_setup()

catkin_package(
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(PROGRAMS 
  scripts/centralised-algorithm/main_centralised.py
  scripts/distributed-algorithm/main_distributed.py
  scripts/path_generator.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
