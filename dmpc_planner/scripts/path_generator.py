#!/usr/bin/env python3

import numpy as np
import rospy
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
import matplotlib.pyplot as plt

def raw_track(choice='sinus'):
    n_checkpoints = 25
    # x_shift_vicon_lab = -3
    # y_shift_vicon_lab = -2.2 #-2.7
    if choice == 'savoiardo':
    
        R = 2  # as a reference the max radius of curvature is  R = L/tan(delta) = 0.82
        theta_init2 = np.pi * -0.5
        theta_end2 = np.pi * 0.5
        theta_vec2 = np.linspace(theta_init2, theta_end2, n_checkpoints)
        theta_init4 = np.pi * 0.5
        theta_end4 = np.pi * 1.5
        theta_vec4 = np.linspace(theta_init4, theta_end4, n_checkpoints)
        Checkpoints_x1 = np.linspace(- 1.5 * R, 1.5 * R, n_checkpoints)
        Checkpoints_y1 = np.zeros(n_checkpoints) - R
        Checkpoints_x2 = 1.5 * R + R * np.cos(theta_vec2)
        Checkpoints_y2 = R * np.sin(theta_vec2)
        Checkpoints_x3 = np.linspace(1.5 * R, -1.5*R, n_checkpoints)
        Checkpoints_y3 = R * np.ones(n_checkpoints)
        Checkpoints_x4 = -1.5* R + R * np.cos(theta_vec4)
        Checkpoints_y4 = R * np.sin(theta_vec4)

        Checkpoints_x = [*Checkpoints_x2[0:n_checkpoints - 1],
                            *Checkpoints_x3[0:n_checkpoints - 1], *Checkpoints_x4[0:n_checkpoints - 1], *Checkpoints_x1[0:n_checkpoints]]
        Checkpoints_y = [*Checkpoints_y2[0:n_checkpoints - 1],
                            *Checkpoints_y3[0:n_checkpoints - 1], *Checkpoints_y4[0:n_checkpoints -1], *Checkpoints_y1[0:n_checkpoints]]
        
    elif choice == 'circle':

        n_checkpoints = 4 * n_checkpoints
        R = 3  # as a reference the max radius of curvature is  R = L/tan(delta) = 0.82
        theta_init = np.pi * -0.5
        theta_end = np.pi * 1.5
        theta_vec = np.linspace(theta_init, theta_end, n_checkpoints)
        Checkpoints_x = R * np.cos(theta_vec)
        Checkpoints_y = R * np.sin(theta_vec)
    
    elif choice == 'straight_line':
        Checkpoints_x = np.linspace(0, 100, n_checkpoints)
        Checkpoints_y = np.zeros(n_checkpoints)
    
    elif choice == 'sinus':
        # Straight line segment from x = -5 to x = -4 at y = 0
        Checkpoints_x_straight = np.linspace(-5, -2, 4)
        Checkpoints_y_straight = np.zeros(4)
        
        # Sinusoidal segment from x = -4 to x = 8 completing one period
        Checkpoints_x_sinus = np.linspace(-2, 12, n_checkpoints-4)
        Checkpoints_y_sinus = 1 * np.sin(2 * np.pi * (Checkpoints_x_sinus + 3) / 14)  # One period from -3 to 10
        
        # Concatenate the two segments
        Checkpoints_x = np.concatenate((Checkpoints_x_straight[:-1], Checkpoints_x_sinus))
        Checkpoints_y = np.concatenate((Checkpoints_y_straight[:-1], Checkpoints_y_sinus))

    return Checkpoints_x, Checkpoints_y

def generate_path_msg():
        # track_choice = settings["track_choice"]
        Checkpoints_x, Checkpoints_y = raw_track()

        path = Path()
        path.header.frame_id = "map"
        path.header.stamp = rospy.Time.now()

        for x, y in zip(Checkpoints_x, Checkpoints_y):
            pose = PoseStamped()
            pose.header.frame_id = "map"
            pose.header.stamp = rospy.Time.now()
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = 0
            pose.pose.orientation.w = 1.0  # No rotation
            path.poses.append(pose)

        return path

def publish_path(event):
    path = generate_path_msg()
    path_pub.publish(path)
    rospy.loginfo("Published path")

if __name__ == '__main__':
    rospy.init_node('path_publisher')
    path_pub = rospy.Publisher("roadmap/reference", Path, queue_size=1)
    rospy.Timer(rospy.Duration(0.5), publish_path)  
    rospy.spin()

    # raw_track = raw_track()
    # plt.plot(raw_track[0], raw_track[1])
    # plt.show()
