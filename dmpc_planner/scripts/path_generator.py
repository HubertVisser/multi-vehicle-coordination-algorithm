#!/usr/bin/env python3

import os, sys
import pathlib
path = pathlib.Path(__file__).parent.resolve()
sys.path.append(os.path.join(path))
sys.path.append(os.path.join(sys.path[0], "..", "..", "solver_generator"))

import numpy as np
import rospy
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
import matplotlib.pyplot as plt

from util.files import load_settings


def raw_track(choice, start_x, start_y):
    n_checkpoints = 10
    if choice == 't_junction' and len(start_x) == 2:

        # Straight segment from (-5, 1) to (-2, 1)
        checkpoints_x_straight1 = np.linspace(start_x[0], -2, 4)
        checkpoints_y_straight1 = np.ones(4) * start_y[0]
        
        # 90 degree turn with radius 3 from (-2, 1) to (0, -2)
        theta = np.linspace(0.5*np.pi,0, 8)
        checkpoints_x_turn = -2 + 3 * np.cos(theta)
        checkpoints_y_turn = -2 + 3 * np.sin(theta)
        
        # Straight segment from (0, -2) to (1, -5)
        checkpoints_x_straight2 = np.ones(4) * start_x[1]
        checkpoints_y_straight2 = np.linspace(-2, start_y[1], 4)
        
        # Concatenate the segments
        checkpoints_x_2 = np.concatenate((checkpoints_x_straight1[:-1], checkpoints_x_turn[:-1], checkpoints_x_straight2))
        checkpoints_y_2 = np.concatenate((checkpoints_y_straight1[:-1], checkpoints_y_turn[:-1], checkpoints_y_straight2))
        
        checkpoints_x_1 = np.linspace(start_x[0], 5, n_checkpoints)
        checkpoints_y_1 = np.zeros(n_checkpoints)

        return checkpoints_x_1, checkpoints_y_1, checkpoints_x_2, checkpoints_y_2
    
    elif choice == 'straight_line' or len(start_x) == 1:
        checkpoints_x_1 = np.linspace(start_x[0], 5, n_checkpoints)
        checkpoints_y_1 = np.zeros(n_checkpoints)
        
        return checkpoints_x_1, checkpoints_y_1

    elif choice == 'sinus' and len(start_x) == 1:
        # Straight line segment from x = -5 at y = 0
        checkpoints_x_straight = np.linspace(start_x[0], start_x[0]+3, 4)
        checkpoints_y_straight = np.zeros(4)
        
        # Sinusoidal segment to x = 20 completing one period
        checkpoints_x_sinus = np.linspace(start_x[0] + 3, start_x[0] + 11, n_checkpoints-4)
        checkpoints_y_sinus = 1 * np.sin(2 * np.pi * (checkpoints_x_sinus + 3) / 8 )  # One period with length 15
        
        # Concatenate the two segments
        checkpoints_x_1 = np.concatenate((checkpoints_x_straight[:-1], checkpoints_x_sinus))
        checkpoints_y_1 = np.concatenate((checkpoints_y_straight[:-1], checkpoints_y_sinus))

        return checkpoints_x_1, checkpoints_y_1

def generate_path_msg():
        settings = load_settings(package="dmpc_planner")
        track_choice = settings["track_choice"]
        num_robot = settings["number_of_robots"]
        start_x = []
        start_y = []
        paths = []
        
        if track_choice=='t_junction' and num_robot==2:
            start_x.extend(settings[f"robot_1"]["start_x"], settings[f"robot_2"]["start_x"])
            start_y.extend(settings[f"robot_1"]["start_y"], settings[f"robot_2"]["start_y"])
                
            checkpoints_x_1, checkpoints_y_1, checkpoints_x_2, checkpoints_y_2 = raw_track(track_choice, start_x, start_y)

            path_1 = Path()
            path_1.header.frame_id = "map"
            path_1.header.stamp = rospy.Time.now()

            for x, y in zip(checkpoints_x_1, checkpoints_y_1):
                pose = PoseStamped()
                pose.header.frame_id = "map"
                pose.header.stamp = rospy.Time.now()
                pose.pose.position.x = x
                pose.pose.position.y = y
                pose.pose.position.z = 0
                pose.pose.orientation.w = 1.0  # No rotation
                path_1.poses.append(pose)
            paths.append(path_1)

            path_2 = Path()
            path_2.header.frame_id = "map"
            path_2.header.stamp = rospy.Time.now()

            for x, y in zip(checkpoints_x_1, checkpoints_y_1):
                pose = PoseStamped()
                pose.header.frame_id = "map"
                pose.header.stamp = rospy.Time.now()
                pose.pose.position.x = x
                pose.pose.position.y = y
                pose.pose.position.z = 0
                pose.pose.orientation.w = 1.0  # No rotation
                path_2.poses.append(pose)
            paths.append(path_2)

        elif track_choice=='straight_line' or num_robot==1:
            start_x.append(settings[f"robot_1"]["start_x"])
            start_y.append(settings[f"robot_1"]["start_y"])
                
            checkpoints_x_1, checkpoints_y_1 = raw_track(track_choice, start_x, start_y)

            path_1 = Path()
            path_1.header.frame_id = "map"
            path_1.header.stamp = rospy.Time.now()

            for x, y in zip(checkpoints_x_1, checkpoints_y_1):
                pose = PoseStamped()
                pose.header.frame_id = "map"
                pose.header.stamp = rospy.Time.now()
                pose.pose.position.x = x
                pose.pose.position.y = y
                pose.pose.position.z = 0
                pose.pose.orientation.w = 1.0  # No rotation
                path_1.poses.append(pose)
            paths.append(path_1)
    
        return paths

def publish_path(event):
    paths = generate_path_msg()
    path_pub_1.publish(paths[0])
    if len(paths) > 1: path_pub_2.publish(paths[1])
    rospy.loginfo("Published paths")

if __name__ == '__main__':
    rospy.init_node('path_publisher')
    path_pub_1 = rospy.Publisher("roadmap/reference_1", Path, queue_size=1)
    path_pub_2 = rospy.Publisher("roadmap/reference_2", Path, queue_size=1)
    rospy.Timer(rospy.Duration(0.5), publish_path)  
    rospy.spin()

    # choice = 'straight_line'
    # start_x = [-5]
    # start_y = [0]
    # raw_track = raw_track(choice, start_x, start_y)
    # plt.plot(raw_track[0], raw_track[1])
    # # plt.plot(raw_track[2], raw_track[3])
    # plt.show()
