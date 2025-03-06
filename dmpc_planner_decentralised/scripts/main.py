#!/usr/bin/env python3

import os, sys
import pathlib
path = pathlib.Path(__file__).parent.resolve()
sys.path.append(os.path.join(path))
sys.path.append(os.path.join(sys.path[0], "..", "..", "solver_generator"))
sys.path.append(os.path.join(sys.path[0], "..", "..", "mpc_planner_modules"))

import threading

import rospy
from nav_msgs.msg import Odometry, Path

import numpy as np
from copy import deepcopy
import math
import matplotlib.pyplot as plt

from util.files import load_settings
from util.convertion import quaternion_to_yaw
from util.logging import print_value 
from util.parameters import GlobalParameters
from timer import Timer

from ros_mpc_controller import ROSMPCPlanner
# from path_generator import generate_path_msg
import debugpy

# Wait for the debugger to attach
debugpy.listen(("localhost", 5678))
print("Waiting for debugger to attach...")
debugpy.wait_for_client()
print("Debugger attached.")

class ROSMPCCoordinator:
    def __init__(self):
        self._settings = load_settings(package="dmpc_planner_decentralised")
        self._N = self._settings["N"]
        self._integrator_step = self._settings["integrator_step"]
        self._number_of_robots = self._settings["number_of_robots"]

        self._robots = []
        for i in range(1, self._number_of_robots+1):
            robot = ROSMPCPlanner(i, self._settings)
            self._robots.append(robot)

        self._trajectory_counter = 0
        self._trajectory_lock = threading.Lock()
        self._trajectory_condition = threading.Condition(self._trajectory_lock)

        self.initialize_subscribers()

        self._timer = rospy.Timer(
            rospy.Duration(1.0 / self._settings["control_frequency"]), self.run
        )

    def initialize_subscribers(self):
        for i in range(1, self._number_of_robots + 1):
            rospy.Subscriber(f"trajectory_{i}", Path, self.trajectory_callback, callback_args=i)

    def trajectory_callback(self, msg, robot_idx):
        # Update the trajectory for the robot
        self._robots[robot_idx - 1].trajectory_callback(msg)

        with self._trajectory_lock:
            self._trajectory_counter += 1
            if self._trajectory_counter == self._number_of_robots:
                self._trajectory_condition.notify_all()

    def run(self, timer):
        # Run NMPC for each robot
        for robot in self._robots:
            if robot._spline_fitter._splines:
                robot.run_nmpc(timer)

        # Run CA for all robots after all trajectories are received
        self.run_ca_for_all_robots(timer)

    def run_ca_for_all_robots(self, timer):
        with self._trajectory_condition:
            while self._trajectory_counter < self._number_of_robots:
                self._trajectory_condition.wait()

            for robot in self._robots:
                robot.run_ca(timer)

            self._trajectory_counter = 0

    def plot_min_distance(self):
        
        min_distances = []
        for state1, state2 in zip(self._states_save_1, self._states_save_2):
            dist = np.linalg.norm(np.array([state1[0], state1[1]]) - np.array([state2[0], state2[1]]))
            min_distances.append(dist)

        plt.plot(min_distances, label='Min Distance')
            

        plt.xlabel('Time Steps')
        plt.ylabel('Minimum Distance')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__), 'plots', 'min_dist_plot.png'))  # Save the plot to a file
        plt.close()
        

        

if __name__ == "__main__":
    try:
        rospy.loginfo("Initializing MPC")
        rospy.init_node("dmpc_planner_coordinator", anonymous=False)
    
        #vehicle           #x y theta vx vy w
        initial_state_1 = [0., 3., 0, 0, 0, 0]
        initial_state_2 = [5.0, 0.0, 0.5 * np.pi, 0, 0, 0]

        coordinator = ROSMPCCoordinator()

        while not rospy.is_shutdown():
            rospy.spin()

        coordinator._robots[0].plot_states()
    except rospy.ROSInterruptException:
        rospy.logerr("ROS Interrupt Exception! Just shutting down to be safe.")
   
        
    
    
    