cmake_minimum_required(VERSION 3.0.2)
project(multi_vehicle_coordination_algorithm)

find_package(catkin REQUIRED COMPONENTS
  rospy
  dynamic_reconfigure
  message_generation
)

add_message_files(
  FILES
  LambdaArray.msg
  LambdaArrayList.msg
)


catkin_python_setup()

generate_messages(
  DEPENDENCIES std_msgs
)

catkin_package(
  CATKIN_DEPENDS rospy std_msgs message_runtime
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
