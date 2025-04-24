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

import numpy as np
from copy import deepcopy
import math
import matplotlib.pyplot as plt

from util.files import load_settings
from util.convertion import quaternion_to_yaw
from util.logging import print_value, TimeTracker

from ros_mpc_controller import ROSMPCPlanner
from plot_utils import plot_distance
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
        for it in range(1, self._iterations+1):


            # Run NMPC for each robot
            for robot in self._robots:
                if robot._spline_fitter._splines:
                    robot.run_nmpc(timer, it)
                else:
                    rospy.logwarn("Splines have not been computed yet. Waiting for splines to be available.")
                    return

            for robot in self._robots:
                if robot._spline_fitter._splines:
                    robot.run_ca(timer, it)
            # Run CA for all robots after all trajectories are received
            # self.run_ca_for_all_robots(timer)
        
        self.time_tracker.add(nmpc_ca_timer.stop())
        del nmpc_ca_timer
        

    def run_ca_for_all_robots(self, timer):
        with self._trajectory_condition:
            while self._trajectory_counter < self._number_of_robots:
                self._trajectory_condition.wait()

            for robot in self._robots:
                if robot._spline_fitter._splines:
                    robot.run_ca(timer)

            self._trajectory_counter = 0

    def plot_distance(self):
        
        poses1 = self._robots[0]._states_save
        poses2 = self._robots[1]._states_save

        length = self._settings["polytopic"]["length"]
        width = self._settings["polytopic"]["width"]

        plot_distance(poses1, poses2, width, length, scheme=self._settings["scheme"])

        

if __name__ == "__main__":
        
    rospy.loginfo("Initializing MPC")
    rospy.init_node("multi_vehicle_coordination_algorithm", anonymous=False)

    coordinator = ROSMPCCoordinator()

    while not rospy.is_shutdown():
        rospy.spin()

    for robot in coordinator._robots:
        robot.plot_states()
        robot.plot_duals()

    coordinator.plot_distance()
    coordinator.time_tracker.print_stats()
    
    


 
   
        
    
    
    