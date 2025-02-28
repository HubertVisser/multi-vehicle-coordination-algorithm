#!/usr/bin/env python3

import os, sys
import pathlib
path = pathlib.Path(__file__).parent.resolve()
sys.path.append(os.path.join(path))
sys.path.append(os.path.join(sys.path[0], "..", "..", "solver_generator"))
sys.path.append(os.path.join(sys.path[0], "..", "..", "mpc_planner_modules"))

import threading

import rospy
from std_msgs.msg import Int32, Float32, Empty
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped, Pose, Twist
from sensor_msgs.msg import Joy
from robot_localization.srv import SetPose
import std_srvs.srv

from mpc_planner_msgs.msg import ObstacleArray
from mpc_planner_msgs.msg import WeightArray

import numpy as np
from copy import deepcopy
import math
import matplotlib.pyplot as plt

from util.files import load_settings
from util.realtime_parameters import RealTimeGlobalParameters
from util.convertion import quaternion_to_yaw
from util.logging import print_value 
from util.parameters import GlobalParameters
from timer import Timer
from spline import Spline, Spline2D

from contouring_spline import SplineFitter
# from static_constraints import StaticConstraints
from mpc_controller import MPCPlanner
from ros_visuals import ROSMarkerPublisher
from project_trajectory import project_trajectory_to_safety
from pyplot import plot_x_traj, plot_splines
# from path_generator import generate_path_msg


class ROSMPCCoordinator:
    def __init__(self):
        self._settings = load_settings(package="dmpc_planner_decentralised")
        self._N = self._settings["N"]
        self._integrator_step = self._settings["integrator_step"]
        self._number_of_robots = self._settings["number_of_robots"]

        self._global_params = GlobalParameters()
        self._global_params_realtime = RealTimeGlobalParameters(self._settings)

        self.define_global_params()
        self._global_params.save_map()

        

    
    def define_global_params(self):
        for i in range(1, self._number_of_robots+1):
            self._global_params.add(f"x_{i}")
            self._global_params.add(f"y_{i}")
            self._global_params.add(f"theta_{i}")

            for j in range(1, self._number_of_robots+1):
                if i != j:
                    self._global_params.add(f"lam_{i}_{j}_0")
                    self._global_params.add(f"lam_{i}_{j}_1")
                    self._global_params.add(f"lam_{i}_{j}_2")
                    self._global_params.add(f"lam_{i}_{j}_3")
            for j in range(i,self._number_of_robots+1):
                if i != j:
                    self._global_params.add(f"s_{i}_{j}_0")
                    self._global_params.add(f"s_{i}_{j}_1")

       
        
        

if __name__ == "__main__":
    # rospy.loginfo("Initializing MPC")
    # rospy.init_node("dmpc_planner", anonymous=False)

    mpc = ROSMPCCoordinator()

    # while not rospy.is_shutdown():
    #     rospy.spin()
        
    # # mpc.plot_outputs()
    # mpc.plot_states()
    # mpc.plot_duals()
    # mpc.plot_min_distance()
    # # mpc.plot_pred_traj()
    # mpc.print_stats()

    
    # try:
    #     rospy.init_node('Vehicles_Integrator_node' , anonymous=True)

    #     #vehicle 1         #x y theta vx vy w
    #     initial_state_1 = [0., 3., 0, 0, 0, 0]
    #     car_number_1 = 1
    #     actuator_dynamics = False
    #     vehicle_1_integrator = Forward_intergrate_vehicle(car_number_1, vehicle_model, initial_state_1,
    #                                                        dt_int,actuator_dynamics)
        
    #     #vehicle 2         #x y theta vx vy w
    #     initial_state_2 = [5.0, 0.0, 0.5 * np.pi, 0, 0, 0]
    #     # initial_state_2 = [, 0, 0, 0, 0, 0]
    #     car_number_2 = 2
    #     actuator_dynamics = False
    #     vehicle_2_integrator = Forward_intergrate_vehicle(car_number_2, vehicle_model, initial_state_2,
    #                                                        dt_int,actuator_dynamics)


    #     vehicles_list = [vehicle_1_integrator, vehicle_2_integrator]

    #     # forwards integrate
    #     #rate = rospy.Rate(1 / dt_int)
    #     while not rospy.is_shutdown():
    #         # forwards integrate all vehicles
    #         for i in range(len(vehicles_list)):
    #             # get time now
    #             rostime_begin_loop = rospy.get_rostime()
    #             vehicles_list[i].forward_integrate_1_timestep(rostime_begin_loop,dt_int)
            

    #             rostime_finished_loop = rospy.get_rostime()
    #             #evalaute time needed to do the loop and print
    #             time_for_loop = rostime_finished_loop - rostime_begin_loop
    #             #print('time for loop: ' + str(time_for_loop.to_sec()))

    #         # if you have extra time, wait until rate to keep the loop at the desired rate
    #         if time_for_loop.to_sec() < dt_int:
    #             # wait by the difference
    #             rospy.sleep(dt_int - time_for_loop.to_sec())


    # except rospy.ROSInterruptException:
    #     pass