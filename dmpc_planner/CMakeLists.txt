cmake_minimum_required(VERSION 3.0.2)
project(mpc_planner_py)

find_package(catkin REQUIRED COMPONENTS
  rospy
  guidance_planner
)

catkin_python_setup()

catkin_package(
)

include_directories(

  # include
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(PROGRAMS
  scripts/jackalsimulator_mpc.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

catkin_install_python(PROGRAMS
  scripts/jackalsimulator_guidance_mpc.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
