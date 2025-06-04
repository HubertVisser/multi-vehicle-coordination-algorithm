#!/usr/bin/env python3

import os, sys
import pathlib
path = pathlib.Path(__file__).parent.resolve()
sys.path.append(os.path.join(path))
sys.path.append(os.path.join(sys.path[-1], "..", "..", "solver_generator"))

import numpy as np
import rospy
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
import matplotlib.pyplot as plt

from util.files import load_settings

class PathGenerator:
    def __init__(self):
        self.settings = load_settings(package="multi_vehicle_coordination_algorithm")
        self.track_choice = self.settings["track_choice"]
        self.num_robot = self.settings["number_of_robots"]

        self.n_pts = 10
        self.robots = {}

        self.load_robot_info()
            
        rospy.Timer(rospy.Duration(0.5), self.publish_path)
        rospy.spin()

    def load_robot_info(self):
        for i in range(1, self.num_robot + 1):
            robot_info = {
                "start_x": self.settings[f"robot_{i}"]["start_x"],
                "start_y": self.settings[f"robot_{i}"]["start_y"],
                "publisher": rospy.Publisher(f"roadmap/reference_{i}", Path, queue_size=1)
            }
            self.robots[i] = robot_info

    def generate_t_junction_track(self):
        pts_x_strt1 = np.ones(4) * self.robots[2]["start_x"]
        pts_y_strt1 = np.linspace(self.robots[2]["start_y"], -2, 4)

        theta = np.linspace(0, 0.5 * np.pi, 8)
        pts_x_turn = -3 + 3 * np.cos(theta)
        pts_y_turn = -2 + 3 * np.sin(theta)

        pts_x_strt2 = np.linspace(-3, -5, 4)
        pts_y_strt2 = np.ones(4)

        pts_x_2 = np.concatenate((pts_x_strt1[:-1], pts_x_turn[:-1], pts_x_strt2))
        pts_y_2 = np.concatenate((pts_y_strt1[:-1], pts_y_turn[:-1], pts_y_strt2))

        pts_x_1, pts_y_1 = self.generate_strt_line_track_2_robot()

        return pts_x_1, pts_y_1, pts_x_2, pts_y_2


    def generate_strt_line_track_1_robot(self):
        pts_x_1 = np.linspace(self.start_x[0], 5, self.n_pts)
        pts_y_1 = np.ones(self.n_pts) * self.start_y[0]

        return pts_x_1, pts_y_1
    
    def generate_strt_line_track_2_robot(self):
        pts_x = np.linspace(-5, 5, self.n_pts)
        pts_y = np.zeros(self.n_pts)

        return pts_x, pts_y

    def generate_merging_track(self):
        pts_x_strt1 = np.linspace(self.start_x[1], -3, 4)
        pts_y_strt1 = np.ones(4) * self.start_y[1]

        # theta = np.linspace(0, 0.5 * np.pi, 8)
        # pts_x_turn = -3 + 3 * np.cos(theta)
        # pts_y_turn = -2 + 3 * np.sin(theta)

        pts_x_strt2 = np.linspace(-1, 6, 4)
        pts_y_strt2 = np.ones(4) * 0

        pts_x_2 = np.concatenate((pts_x_strt1, pts_x_strt2))
        pts_y_2 = np.concatenate((pts_y_strt1, pts_y_strt2))

        pts_x_1, pts_y_1 = self.generate_strt_line_track_1_robot()

        return pts_x_1, pts_y_1, pts_x_2, pts_y_2

    def generate_path_msg(self):
        if self.track_choice == 't_junction':
            assert self.num_robot == 2, "T-junction track only supports 2 robots"
            pts_x_1, pts_y_1, pts_x_2, pts_y_2 = self.generate_t_junction_track()
            self.robots[1]["path"] = self.create_path(pts_x_1, pts_y_1)
            self.robots[2]["path"] = self.create_path(pts_x_2, pts_y_2)

        elif self.track_choice == 'straight_line':
            pts_x, pts_y = self.generate_strt_line_track_2_robot()
            for i in range(1, self.num_robot + 1):
                self.robots[i]["path"] = self.create_path(pts_x, pts_y)

        elif self.track_choice == 'merging':
            assert self.num_robot == 2, "Merging track only supports 2 robots"
            pts_x_1, pts_y_1, pts_x_2, pts_y_2 = self.generate_merging_track()
            self.robots[1]["path"] = self.create_path(pts_x_1, pts_y_1)
            self.robots[2]["path"] = self.create_path(pts_x_2, pts_y_2)

        else:
            raise ValueError(f"Invalid track choice: {self.track_choice}")

    def create_path(self, pts_x, pts_y):
        path = Path()
        path.header.frame_id = "map"
        path.header.stamp = rospy.Time.now()

        for x, y in zip(pts_x, pts_y):
            pose = PoseStamped()
            pose.header.frame_id = "map"
            pose.header.stamp = rospy.Time.now()
            pose.pose.position.x = float(x)
            pose.pose.position.y = float(y)
            pose.pose.position.z = 0
            pose.pose.orientation.w = 1.0  # No rotation
            path.poses.append(pose)

        return path

    def publish_path(self, event):
        self.generate_path_msg()
        for i in range(1, self.num_robot + 1):
            pub = self.robots[i]['publisher']
            path = self.robots[i]['path']
            pub.publish(path)
        rospy.loginfo("Paths published")

if __name__ == '__main__':
    rospy.init_node('path_publisher')
    path_generator = PathGenerator()

    # choice = 't_junction'
    # start_x = [-4, 0]
    # start_y = [0, -4]
    # raw_track = raw_track(choice, start_x, start_y, positive_track=False)
    # plt.plot(raw_track[0], raw_track[1])
    # plt.plot(raw_track[2], raw_track[3])
    # plt.show()