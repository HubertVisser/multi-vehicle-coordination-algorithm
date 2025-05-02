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
    
    # Get the current environment and ensure ROS parameters are passed
    env = os.environ.copy()
    env["ROS_MASTER_URI"] = os.environ.get("ROS_MASTER_URI", "http://localhost:11311")
    env["ROS_PACKAGE_PATH"] = os.environ.get("ROS_PACKAGE_PATH", "")

    # Run the launch file using roslaunch with the updated environment
    subprocess.call(["roslaunch", launch_file], env=env)

if __name__ == "__main__":
    main()