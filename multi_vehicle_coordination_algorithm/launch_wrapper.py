#!/usr/bin/env python3
import rospy
import os
import subprocess
import signal

def main():
    rospy.init_node("launch_wrapper", anonymous=True)

    # Get the 'scheme' parameter from the ROS parameter server
    scheme = rospy.get_param("scheme", "default_scheme")  # Use default if missing
    launch_file = f"/home/dock_user/ros_ws/src/multi-vehicle-coordination-algorithm/multi_vehicle_coordination_algorithm/launch/{scheme}_algorithm.launch"
    debug = rospy.get_param("~debug", False) 

    # Run the launch file using roslaunch
    process = subprocess.Popen(["roslaunch", launch_file, f"debug:={debug}"])

    def shutdown_hook():
        rospy.loginfo("Shutting down Launch Wrapper...")
        process.send_signal(signal.SIGINT)  # Send SIGINT to the subprocess
        process.wait()  # Wait for the subprocess to terminate

    rospy.on_shutdown(shutdown_hook)
    rospy.spin()

if __name__ == "__main__":
    main()