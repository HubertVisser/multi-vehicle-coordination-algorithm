#!/usr/bin/bash 

# Debug: Print environment variables
echo "ROS_PACKAGE_PATH: $ROS_PACKAGE_PATH"
echo "PYTHONPATH: $PYTHONPATH"
echo "PATH: $PATH"

# Set the PYTHONPATH
export PYTHONPATH=$HOME/ros_ws/src:$HOME/ros_ws/devel/lib/python3/dist-packages:/opt/ros/noetic/lib/python3/dist-packages

# Debug: Print PYTHONPATH after setting it
echo "PYTHONPATH after setting: $PYTHONPATH"

# Change to the script directory
cd $HOME/ros_ws/src/DMPC_planner/dmpc_planner/scripts

# Ensure PYTHONPATH is still set correctly after changing directory
export PYTHONPATH=$HOME/ros_ws/src:$HOME/ros_ws/devel/lib/python3/dist-packages:/opt/ros/noetic/lib/python3/dist-packages

# Debug: Print PYTHONPATH after changing directory
echo "PYTHONPATH after cd: $PYTHONPATH"

# Activate Poetry environment and run the Python script as a module
python3 jetracersimulator_mpc.py