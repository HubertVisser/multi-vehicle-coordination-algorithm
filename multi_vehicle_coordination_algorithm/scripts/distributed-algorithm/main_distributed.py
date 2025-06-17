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
        self._number_of_calls = self._settings['number_of_calls']

        self._robots = []
        for i in range(1, self._number_of_robots+1):
            robot = ROSMPCPlanner(i, self._settings)
            self._robots.append(robot)

            rospy.loginfo(f"Waiting for path {i} to be received and plotted...")
            robot._path_ready_event.wait()
            rospy.loginfo(f"Path {i} received and plotted. Starting controller.")

        self._trajectory_counter = 0
        self._trajectory_lock = threading.Lock()
        self._trajectory_condition = threading.Condition(self._trajectory_lock)

        self.time_tracker = TimeTracker(f"Distributed - iterations: {self._iterations} vehicles: {self._number_of_robots}")
        # self._timer = rospy.Timer(
        #     rospy.Duration(1.0 / self._settings["control_frequency"]), self.run
        # )

    def run(self, timer):

        max_attempts = 10
        attempts = 0
        while not all(robot._spline_fitter._splines for robot in self._robots) and attempts < max_attempts:
            rospy.loginfo("Waiting for all robots to fit their path splines... (attempt %d/%d)", attempts + 1, max_attempts)
            rospy.sleep(0.1)  # Sleep 100ms before checking again
            attempts += 1
        if not all(robot._spline_fitter._splines for robot in self._robots):
            rospy.logwarn("Not all robots have fitted their path splines after %d attempts. Shutting down.", max_attempts)
            rospy.signal_shutdown("Failed to fit all robot path splines.")
            return
            
        nmpc_ca_timer = Timer("NMPC-CA")

        # Create a ThreadPoolExecutor for parallel execution
        with ThreadPoolExecutor() as executor:
            for it in range(1, self._iterations + 1):
                # Run NMPC for each robot in parallel
                nmpc_futures = [
                    executor.submit(robot.run_nmpc, timer, it)
                    for robot in self._robots
                ]

                # Wait for all NMPC tasks to complete
                for future in nmpc_futures:
                    future.result()  # This will raise any exceptions if they occur

                #--- Wait for all robots to receive neighbour trajectories ---
                attempts = 0
                while not all(robot.all_neighbour_trajectories_received() for robot in self._robots) and attempts < max_attempts:
                    rospy.loginfo("Waiting for all robots to receive neighbour trajectories...(attempt %d/%d)", attempts + 1, max_attempts)
                    rospy.sleep(0.1)
                    attempts += 1
                if not all(robot._spline_fitter._splines for robot in self._robots):
                    rospy.logwarn("Not all trajectories are received after %d attempts. Shutting down.", max_attempts)
                    rospy.signal_shutdown("Failed to fit all robot path splines.")
                    return
                # -----------------------------------------------------------

                # Run CA for each robot in parallel
                ca_futures = [
                    executor.submit(robot.run_ca, timer, it)
                    for robot in self._robots
                ]

                # Wait for all CA tasks to complete
                for future in ca_futures:
                    future.result()  # This will raise any exceptions if they occur

        self.time_tracker.add(nmpc_ca_timer.stop())
        del nmpc_ca_timer

    def plot_distance(self):
        
        poses_1 = self._robots[0]._states_history
        poses_2 = self._robots[1]._states_history

        length = self._settings["polytopic"]["length"]
        width = self._settings["polytopic"]["width"]

        plot_distance(poses_1, poses_2, width, length, scheme=self._settings["scheme"])
    
    def plot_trajectory(self):

        poses_1 = self._robots[0]._states_history
        poses_2 = self._robots[1]._states_history
        reference_1 = self._robots[0].load_centralised_traj(1)
        reference_2 = self._robots[1].load_centralised_traj(2)

        plot_trajectory(np.array(poses_1), np.array(poses_2), reference_1, reference_2, track_choice=self._settings["track_choice"], scheme=self._settings["scheme"])

        
if __name__ == "__main__":
        
    rospy.loginfo("Initializing MPC")
    rospy.init_node("multi_vehicle_coordination_algorithm", anonymous=False)

    coordinator = ROSMPCCoordinator()
    rate = rospy.Rate(coordinator._settings['control_frequency'])

    # while not rospy.is_shutdown():
    #     rospy.spin()

    for _ in range(coordinator._number_of_calls):
        if rospy.is_shutdown():
            break
        coordinator.run(None)
        rate.sleep()

    coordinator.plot_trajectory()
    coordinator.time_tracker.print_stats()
    coordinator.plot_distance()

    for robot in coordinator._robots:
        robot.plot_states()
        robot.plot_duals()
        robot.evaluate_tracking_error()
        # robot.log_tracking_error()
        robot.plot_slack()
        robot.evaluate_total_cost()


    
    


 
   
        
    
    
    