cmake_minimum_required(VERSION 3.8)
project(mpc_planner_standalone)

add_compile_options(-std=c++17)

find_package(OpenMP)

if(OPENMP_FOUND)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
else()
  message(WARNING "Compiling without OPENMP")
endif()

find_package(catkin REQUIRED COMPONENTS
  mpc_planner_types
  mpc_planner_util
  message_generation
  ros_tools
  decomp_util
  tf
  costmap_2d
)

add_service_files(
  FILES
  decomp.srv
)

generate_messages(
  DEPENDENCIES
  std_msgs
)

catkin_package(
  CATKIN_DEPENDS mpc_planner_types mpc_planner_util message_runtime ros_tools decomp_util tf costmap_2d
  INCLUDE_DIRS include include/${PROJECT_NAME}
  LIBRARIES ${PROJECT_NAME}
)

# # BUILD ###
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

add_definitions(-DMPC_PLANNER_ROS)
add_definitions(-DDECOMP_OLD)

add_executable(decomp_node src/decomp_node.cpp)
add_dependencies(decomp_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(decomp_node ${catkin_LIBRARIES})


install(
  TARGETS decomp_node
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
  PATTERN ".svn" EXCLUDE
)