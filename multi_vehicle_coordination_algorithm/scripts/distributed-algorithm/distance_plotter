#!/usr/bin/env python3
# filepath: /home/dock_user/ros_ws/src/multi-vehicle-coordination-algorithm/multi_vehicle_coordination_algorithm/scripts/distributed-algorithm/distance_plotter.py

import rospy
from nav_msgs.msg import Odometry
from plot_utils import plot_distance

class DistancePlotter:
    def __init__(self):
        self.robot_1_states = []
        self.robot_2_states = []

        # Subscribers for robot states
        rospy.Subscriber("/robot_1/state", Odometry, self.robot_1_callback)
        rospy.Subscriber("/robot_2/state", Odometry, self.robot_2_callback)

        self._settings = rospy.get_param("/multi_vehicle_coordination_algorithm/settings")
        self.length = self._settings["polytopic"]["length"]
        self.width = self._settings["polytopic"]["width"]

        rospy.Timer(rospy.Duration(1.0), self.plot_distances)

    def robot_1_callback(self, msg):
        self.robot_1_states.append(msg.pose.pose)

    def robot_2_callback(self, msg):
        self.robot_2_states.append(msg.pose.pose)

    def plot_distances(self, event):
        if len(self.robot_1_states) > 0 and len(self.robot_2_states) > 0:
            plot_distance(self.robot_1_states, self.robot_2_states, self.width, self.length, scheme="distributed")
            rospy.loginfo("Distance plot updated.")


if __name__ == "__main__":
    rospy.init_node("distance_plotter", anonymous=False)
    plotter = DistancePlotter()
    rospy.spin()