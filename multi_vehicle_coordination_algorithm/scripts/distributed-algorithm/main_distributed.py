#!/usr/bin/env python3

import os, sys
import pathlib
path = pathlib.Path(__file__).parent.resolve()
sys.path.append(os.path.join(path))
sys.path.append(os.path.join(sys.path[-1], "..", "..", "..", "solver_generator"))
sys.path.append(os.path.join(sys.path[-2], "..", "..", "..", "mpc_planner_modules"))
sys.path.append(os.path.join(sys.path[-3], ".."))

import threading

import rospy
from nav_msgs.msg import Odometry, Path
from concurrent.futures import ThreadPoolExecutor


import numpy as np
from copy import deepcopy
import math
import matplotlib.pyplot as plt

from util.files import load_settings
from util.convertion import quaternion_to_yaw
from util.logging import print_value, TimeTracker

from ros_mpc_controller import ROSMPCPlanner
from plot_utils import plot_distance, plot_trajectory, get_reference_from_path_msg
from timer import Timer


class ROSMPCCoordinator:
    def __init__(self):
        self._settings = load_settings(package="multi_vehicle_coordination_algorithm")
        self._N = self._settings["N"]
        self._integrator_step = self._settings["integrator_step"]
        self._number_of_robots = self._settings["number_of_robots"]
        self._iterations = self._settings['solver_settings']['iterations_distributed']

        self._robots = []
        for i in range(1, self._number_of_robots+1):
            robot = ROSMPCPlanner(i, self._settings)
            self._robots.append(robot)

        self._trajectory_counter = 0
        self._trajectory_lock = threading.Lock()
        self._trajectory_condition = threading.Condition(self._trajectory_lock)

        self.time_tracker = TimeTracker(f"Distributed - iterations: {self._iterations} vehicles: {self._number_of_robots}")
        self._timer = rospy.Timer(
            rospy.Duration(1.0 / self._settings["control_frequency"]), self.run
        )

    def run(self, timer):
        nmpc_ca_timer = Timer("NMPC-CA")

        # Create a ThreadPoolExecutor for parallel execution
        with ThreadPoolExecutor() as executor:
            for it in range(1, self._iterations + 1):
                # Run NMPC for each robot in parallel
                nmpc_futures = [
                    executor.submit(robot.run_nmpc, timer, it)
                    for robot in self._robots
                    if robot._spline_fitter._splines
                ]

                # Wait for all NMPC tasks to complete
                for future in nmpc_futures:
                    future.result()  # This will raise any exceptions if they occur

                # Run CA for each robot in parallel
                ca_futures = [
                    executor.submit(robot.run_ca, timer, it)
                    for robot in self._robots
                    if robot._spline_fitter._splines #and robot.all_neighbor_trajectories_received()
                ]

                # Wait for all CA tasks to complete
                for future in ca_futures:
                    future.result()  # This will raise any exceptions if they occur

        self.time_tracker.add(nmpc_ca_timer.stop())
        del nmpc_ca_timer
        _, _, calls = self.time_tracker.get_stats()
        print_value("calls", calls)
        

    # def run_ca_for_all_robots(self, timer):
    #     with self._trajectory_condition:
    #         while self._trajectory_counter < self._number_of_robots:
    #             self._trajectory_condition.wait()

    #         for robot in self._robots:
    #             if robot._spline_fitter._splines:
    #                 robot.run_ca(timer)

    #         self._trajectory_counter = 0

    def plot_distance(self):
        
        poses_1 = self._robots[0]._states_history
        poses_2 = self._robots[1]._states_history

        length = self._settings["polytopic"]["length"]
        width = self._settings["polytopic"]["width"]

        plot_distance(poses_1, poses_2, width, length, scheme=self._settings["scheme"])
    
    def plot_trajectory(self):

        poses_1 = self._robots[0]._states_history
        poses_2 = self._robots[1]._states_history
        reference_1 = get_reference_from_path_msg(self._robots[0]._path_msg)
        reference_2 = get_reference_from_path_msg(self._robots[1]._path_msg)

        plot_trajectory(np.array(poses_1), np.array(poses_2), reference_1, reference_2, track_choice=self._settings["track_choice"], scheme=self._settings["scheme"])

        
if __name__ == "__main__":
        
    rospy.loginfo("Initializing MPC")
    rospy.init_node("multi_vehicle_coordination_algorithm", anonymous=False)

    coordinator = ROSMPCCoordinator()

    while not rospy.is_shutdown():
        rospy.spin()

    for robot in coordinator._robots:
        robot.plot_states()
        robot.plot_duals()
        robot.log_tracking_error()
        robot.plot_slack()

    coordinator.plot_distance()
    coordinator.plot_trajectory()
    coordinator.time_tracker.print_stats()
    
    


 
   
        
    
    
    