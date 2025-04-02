#!/usr/bin/env python3
import rospy
import os
import subprocess

def main():
    rospy.init_node("launch_wrapper", anonymous=True)

    # Get the 'scheme' parameter from the ROS parameter server
    scheme = rospy.get_param("scheme", "default_scheme")  # Use default if missing

    # Construct the launch file path dynamically
    launch_file = f"/home/dock_user/ros_ws/src/multi-vehicle-coordination-algorithm/multi_vehicle_coordination_algorithm/launch/{scheme}_algorithm.launch"
    
    # Run the launch file using roslaunch
    subprocess.call(["roslaunch", launch_file])

if __name__ == "__main__":
    main()