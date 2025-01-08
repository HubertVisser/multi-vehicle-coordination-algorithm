#!/usr/bin/bash 

# Source the ROS setup file
source /opt/ros/<ros_distro>/setup.bash

# Source the workspace setup file
source $HOME/ros_ws/devel/setup.bash

# Debug: Print environment variables
echo "ROS_PACKAGE_PATH: $ROS_PACKAGE_PATH"
echo "PYTHONPATH: $PYTHONPATH"
echo "PATH: $PATH"

# Set the PYTHONPATH
export PYTHONPATH=$HOME/ros_ws/src:$PYTHONPATH

# Debug: Print PYTHONPATH after setting it
echo "PYTHONPATH after setting: $PYTHONPATH"

# Print Python version
python3 --version

# Check if rospy is available
python3 -c "import rospy; print('rospy is available')"

path="$HOME/ros_ws/src"
export PYTHONPATH=${path}

# Change to the script directory
cd $HOME/ros_ws/src/DMPC_planner/dmpc_planner/scripts

# Ensure PYTHONPATH is still set correctly after changing directory
export PYTHONPATH=$HOME/ros_ws/src:$PYTHONPATH

# Debug: Print PYTHONPATH after changing directory
echo "PYTHONPATH after cd: $PYTHONPATH"

# Check if rospy is available after changing directory
python3 -c "import rospy; print('rospy is available after cd')"

# Run the Python script
python3 jetracersimulator_mpc.py