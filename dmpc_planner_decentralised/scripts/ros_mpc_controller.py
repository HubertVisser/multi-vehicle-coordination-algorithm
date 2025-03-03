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
from util.realtime_parameters import RealTimeParameters
from util.convertion import quaternion_to_yaw, yaw_to_quaternion
from util.logging import print_value 
from timer import Timer
from spline import Spline, Spline2D

from contouring_spline import SplineFitter
# from static_constraints import StaticConstraints
from mpc_controller import MPCPlanner
from ros_visuals import ROSMarkerPublisher
from project_trajectory import project_trajectory_to_safety
from pyplot import plot_x_traj, plot_splines
# from path_generator import generate_path_msg


class ROSMPCPlanner:
    def __init__(self, idx, settings=None):
        self._settings = settings
        self._N = self._settings["N"]
        self._integrator_step = self._settings["integrator_step"]
        self._braking_acceleration = self._settings["braking_acceleration"]
        self._number_of_robots = self._settings["number_of_robots"]
        self._idx = idx

        self._verbose = self._settings["verbose"]
        self._debug_visuals = self._settings["debug_visuals"]
        self._dart_simulator = self._settings["dart_simulator"]

        self._planner = MPCPlanner(self._settings, idx)
        # self._planner.set_projection(lambda trajectory: self.project_to_safety(trajectory))

        self._spline_fitter = SplineFitter(self._settings)

        self._solver_settings_nmpc = load_settings("solver_settings_nmpc", package="mpc_planner_solver")
        self._solver_settings_ca = load_settings("solver_settings_ca", package="mpc_planner_solver")

        # Tied to the solver
        self._params_nmpc = RealTimeParameters(self._settings, parameter_map_name=f"parameter_map_nmpc_{idx}", package="mpc_planner_solver")  
        self._params_ca = RealTimeParameters(self._settings, parameter_map_name=f"parameter_map_ca_{idx}", package="mpc_planner_solver")  
        self._weights = self._settings["weights"]

        self._nx_nmpc = self._solver_settings_nmpc["nx"]
        self._nu_nmpc = self._solver_settings_nmpc["nu"]
        self._nvar_nmpc = self._solver_settings_nmpc["nvar"]
        self._nx_ca = self._solver_settings_ca["nx"]
        self._nu_ca = self._solver_settings_ca["nu"]
        self._nvar_ca = self._solver_settings_ca["nvar"]

        self._state = np.zeros((self._nx,))
        self._state[: 3] = [self._settings[f"robot_{self._idx}"]["start_x"], \
                            self._settings[f"robot_{self._idx}"]["start_y"], \
                            self._settings[f"robot_{self._idx}"]["start_theta"] * np.pi]

        self._visuals = ROSMarkerPublisher("mpc_visuals", 100)
        self._path_visual = ROSMarkerPublisher("reference_path", 10)
        self._debug_visuals_pub = ROSMarkerPublisher("mpc_planner_py/debug", 10)

        self._state_msg = None

        self._path_msg = None

        self._trajectory = None

        self._states_save = []
        self._outputs_save = []
        self._save_output = []
        self._save_s = np.array([])
        self._save_lam = np.array([])


        self._neighbour_pos = np.array([])

        self._enable_output = False
        self._mpc_feasible = True

        self._timer = rospy.Timer(
            rospy.Duration(1.0 / self._settings["control_frequency"]), self.run
        )

        self._callbacks_enabled = False
        self.initialize_publishers_and_subscribers()
        self._callbacks_enabled = True
        self._enable_output = True
        # self.start_environment()

    def initialize_publishers_and_subscribers(self):

        # Subscribers DART
        self._state_sub = rospy.Subscriber(f"vicon/jetracer{self._idx}", PoseStamped, self.state_pose_callback, queue_size=1)
        self._vy_sub = rospy.Subscriber(f"vy_{self._idx}", Float32, self.vy_pose_callback, queue_size=1)
        self._w_sub = rospy.Subscriber(f"omega_{self._idx}", Float32, self.w_pose_callback, queue_size=1)

        # Subscriber path generator
        self._path_sub = rospy.Subscriber(f"roadmap/reference_{self._idx}", Path, lambda msg: self.path_callback(msg), queue_size=1)

        # Trajectory publisher
        setattr(self, f"_traj_pub_{self._idx}") = rospy.Publisher(f"trajectory_{self._idx}", Path, queue_size=1)

        # Publishers
        self._th_pub = rospy.Publisher(f"throttle_{self._idx}", Float32, queue_size=1) # Throttle publisher
        self._st_pub = rospy.Publisher(f"steering_{self._idx}", Float32, queue_size=1) # Steering publisher

    def run_nmpc(self, timer):
        # Check if splines exist
        if not self._spline_fitter._splines:
            rospy.logwarn("Splines have not been computed yet. Waiting for splines to be available.")
            return
        
        timer = Timer("loop")
        
        self.set_nmpc_parameters()
        # self._params.print()
        # self._params.check_for_nan()

        mpc_timer = Timer("NMPC")
        output, self._mpc_feasible, self._trajectory = self._planner.solve_nmpc( 
            self._state, self._params_nmpc.get_solver_params()
        )
        del mpc_timer

        if self._verbose:
            time = timer.stop_and_print()

        if self._mpc_feasible:
            if self._dart_simulator == False:
                output_keys = [f"x_{self._idx}", f"y_{self._idx}", f"theta_{self._idx}", f"vx_{self._idx}", f"vy_{self._idx}", f"w_{self._idx}", f"s_{self._idx}"]
                self._state = [output[key] for key in output_keys]
                getattr(self, f'_states_save').append(deepcopy(self._state))
            
            getattr(self, f'_outputs_save').append([output[f"throttle_{self._idx}"], output[f"steering_{self._idx}"]])
            
            # self.plot_pred_traj() # slows down the simulation

        self.publish_throttle(output, self._mpc_feasible) if self._dart_simulator else None
        self.publish_steering(output, self._mpc_feasible) if self._dart_simulator else None
        self.publish_trajectory(self._trajectory) 
    
    def run_ca(self, timer):
        # Check if splines exist
        
        timer = Timer("loop")
        
        self.set_ca_parameters()
        # self._params.print()
        # self._params.check_for_nan()

        ca_timer = Timer("CA")
        output, self._mpc_feasible, self._ca_solution = self._planner.solve_ca(self._params_ca.get_solver_params())
        del ca_timer

        if self._verbose:
            time = timer.stop_and_print()

        if self._mpc_feasible:
            lam = np.array([])
            for i in range(1, self._number_of_robots+1):
                for j in range(1, self._number_of_robots+1):
                    if i != j and (j == self._idx or i == self._idx):
                        lam = np.concatenate((lam, np.array([output[f"lam_{i}_{j}_0"], 
                                                             output[f"lam_{i}_{j}_1"], 
                                                             output[f"lam_{i}_{j}_2"], 
                                                             output[f"lam_{i}_{j}_3"]])))
                for j in range(i, self._number_of_robots+1):
                    if i != j and (j == self._idx or i == self._idx):
                        self._save_s = np.vstack((self._save_s, np.array([output[f"s_{i}_{j}_0"], output[f"s_{i}_{j}_1"]]))) if self._save_s.size else np.array([output[f"s_{i}_{j}_0"], output[f"s_{i}_{j}_1"]])
            
            self._save_lam = np.vstack((self._save_lam, lam)) if self._save_lam.size else lam
            
            # self.plot_pred_traj() # slows down the simulation

    def set_nmpc_parameters(self):
        
        splines = None
        if self._path_msg is not None and self._spline_fitter.ready():
            splines = self._spline_fitter.get_active_splines(np.array([self._state[0], self._state[1]]))
            self._state[-1] = self._spline_fitter.find_closest_s(np.array([self._state[0], self._state[1]]))

        # Set parameters for all k
        for k in range(self._N + 1):

            # Tuning parameters
            for weight, value in self._weights.items():
                self._params_nmpc.set(k, weight, value)

            if splines is not None:
                for i in range(self._settings["contouring"]["num_segments"]):
                    self._params_nmpc.set(k, f"spline_x{i}_a_{self._idx}", splines[i]["a_x"])
                    self._params_nmpc.set(k, f"spline_x{i}_b_{self._idx}", splines[i]["b_x"])
                    self._params_nmpc.set(k, f"spline_x{i}_c_{self._idx}", splines[i]["c_x"])
                    self._params_nmpc.set(k, f"spline_x{i}_d_{self._idx}", splines[i]["d_x"])
                
                    self._params_nmpc.set(k, f"spline_y{i}_a_{self._idx}", splines[i]["a_y"])
                    self._params_nmpc.set(k, f"spline_y{i}_b_{self._idx}", splines[i]["b_y"])
                    self._params_nmpc.set(k, f"spline_y{i}_c_{self._idx}", splines[i]["c_y"])
                    self._params_nmpc.set(k, f"spline_y{i}_d_{self._idx}", splines[i]["d_y"])

                    self._params_nmpc.set(k, f"spline{i}_start_{self._idx}", splines[i]["s"])        

    def publish_throttle(self, output, exit_flag):
        for n in range(1, self._number_of_robots + 1):
            throttle = Float32()
            if not self._mpc_feasible or not self._enable_output:
                if not self._mpc_feasible:
                    rospy.logwarn_throttle(1, "Infeasible MPC. Braking!")
                    throttle.data = max(0.0, self._state[3 + (n-1) * self._nx_one_robot] - self._braking_acceleration * self._integrator_step,)
                else:
                    rospy.logwarn_throttle(1, "Output is disabled. Sending zero velocity!")
                    throttle.data = 0.0
            else:
                throttle.data = output[f"throttle_{n}"]
                rospy.loginfo_throttle(1000, "MPC is driving")
                getattr(self,f"_th_pub_{n}").publish(throttle)
    
    def publish_steering(self, output, exit_flag):
         for n in range(1, self._number_of_robots + 1):
            steering = Float32()
            if not self._mpc_feasible or not self._enable_output:
                steering.data = 0.0
            else:
                steering.data = output[f"steering_{n}"]
                getattr(self, f"_st_pub_{n}").publish(steering)

    def publish_robot_state(self):
        pose = PoseStamped()
        pose.header.stamp = rospy.Time.now()
        pose.header.frame_id = "map"

        pose.pose.position.x = self._state[0]
        pose.pose.position.y = self._state[1]
        pose.pose.position.z = self._state[3]
        self._ped_robot_state_pub.publish(pose)
    
    def publish_trajectory(self, trajectory):
        if trajectory is not None:
            traj_msg = Path()
            traj_msg.header.stamp = rospy.Time.now()
            traj_msg.header.frame_id = "map"
            for k in range(1, self._N):
                pose = PoseStamped()
                pose.header.stamp = rospy.Time.now()
                pose.header.frame_id = "map"
                if k != self._N - 1:
                    pose.pose.position.x = trajectory[k, 0]
                    pose.pose.position.y = trajectory[k, 1]
                    pose.pose.orientation = yaw_to_quaternion(trajectory[k, 2])
                elif k == self._N - 1:
                    pose.pose.position.x = trajectory[k, 0] + (trajectory[k, 0] - trajectory[k - 1, 0])
                    pose.pose.position.y = trajectory[k, 1] + (trajectory[k, 1] - trajectory[k - 1, 1])
                    pose.pose.orientation = yaw_to_quaternion(trajectory[k, 2] + (trajectory[k, 2] - trajectory[k - 1, 2]))
                traj_msg.poses.append(pose)
            
            getattr(self, f"_traj_pub_{self._idx}").publish(traj_msg)
    
    def visualize(self): #TODO change to one vehicle
        for n in range(1, self._number_of_robots+1):
            state_msg = getattr(self, f'_state_msg_{n}')
            splineFitter = getattr(self, f'_spline_fitter_{n}')
            if state_msg is not None:
                robot_pos = self._visuals.get_sphere()
                robot_pos.set_color(0)
                robot_pos.set_scale(0.3, 0.3, 0.3)

                pose = Pose()
                pose.position.x = float(self._state[0 + (n-1) * self._nx_one_robot])
                pose.position.y = float(self._state[1 + (n-1) * self._nx_one_robot])
                robot_pos.add_marker(pose)
            if self._save_s.size:
                line = self._visuals.get_line()
                line.set_scale(0.05)
                line.set_color(n*7, alpha=1.0)
                ego_pos = np.array([self._state[0 + (n-1)*self._nx_one_robot], self._state[1 + (n-1)*self._nx_one_robot]])

                for j in range(n, self._number_of_robots+1):
                    if j != n:
                        s = self._save_s[-1,:]
                        #setattr(self, f'neighbour_pos_{j}', np.array([]))
                        neighbour_pos = np.array([self._state[0 + (j-1)*self._nx_one_robot], self._state[1 + (j-1)*self._nx_one_robot]])
                        midpoint = (ego_pos + neighbour_pos) / 2
                
                        # Calculate the direction vector of the line (perpendicular to the normal vector)
                        direction_vector = np.array([-s[1], s[0]]) 
                        assert np.dot(direction_vector, s) == 0
                        line_length = 100
                        line_start = midpoint - (line_length / 2) * direction_vector
                        line_end = midpoint + (line_length / 2) * direction_vector
                        pose_a = Pose()
                        pose_a.position.x = float(line_start[0])
                        pose_a.position.y = float(line_start[1])
                        pose_b = Pose()
                        pose_b.position.x = float(line_end[0])
                        pose_b.position.y = float(line_end[1])
                        line.add_line_from_poses(pose_a, pose_b)

            if self._debug_visuals:
                if splineFitter._closest_s is not None:
                    cube = self._debug_visuals_pub.get_cube()
                    cube.set_color(5*n)
                    cube.set_scale(0.5, 0.5, 0.5)
                    pose = Pose()
                    pose.position.x = float(splineFitter._closest_x)
                    pose.position.y = float(splineFitter._closest_y)
                    cube.add_marker(pose)

                self.plot_warmstart()
                self._debug_visuals_pub.publish()

            if self._trajectory is not None and self._mpc_feasible:
                cylinder = self._visuals.get_cylinder()
                cylinder.set_color(n, alpha=0.25)
                cylinder.set_scale(0.5, 0.5, 0.05)

                pose = Pose()
                for k in range(1, self._N):
                    pose.position.x = float(self._planner.get_model().get(k, f"x_{n}"))
                    pose.position.y = float(self._planner.get_model().get(k, f"y_{n}"))
                    cylinder.add_marker(deepcopy(pose))
        self._visuals.publish()
    
    # Callback functions
    def state_pose_callback(self, msg):
        if self._dart_simulator:
            self._state_msg = msg
            self._state[0] = msg.pose.position.x
            self._state[1] = msg.pose.position.y

            # Extract yaw angle (rotation around the Z-axis)
            self._state[2] = quaternion_to_yaw(msg.pose.orientation)

            # Velocity is in the local frame, x is the forward velocity
            self._state[3] = msg.pose.position.z

            self._states_save.append(deepcopy(self._state[:self._nx_one_robot]))
            # print("-------- State ----------")
            # print(f"x = {self._state[0]:.2f}")
            # print(f"y = {self._state[1]:.2f}")
            # print(f"theta = {self._state[2]:.2f}")
            # print(f"vx = {self._state[3]:.2f}"))

    def vy_pose_callback(self, msg):
        if self._dart_simulator:
            self._state[4] = msg.data

    
    def w_pose_callback(self, msg):
        if self._dart_simulator:
            self._state[5] = msg.data
    
    def path_callback(self, msg):

        # Filter equal paths
        if self._path_msg is not None and len(self._path_msg.poses) == len(msg.poses):
            return

        self._path_msg = msg
        self._spline_fitter.fit_path(msg)
        # plot_splines(self._spline_fitter._splines)
        self.plot_path()

    def trajectory_callback(self, traj_msg):
        idx = rospy.get_name()[-1]

        if not traj_msg.poses:
            rospy.logwarn(f"Received empty path robot {idx}")
            return
        
        for k, pose in enumerate(traj_msg.poses):
            self._params_ca.set(k, f"x_{idx}", pose.pose.position.x)
            self._params_ca.set(k, f"y_{idx}", pose.pose.position.y)
            self._params_ca.set(k, f"theta_{idx}", quaternion_to_yaw(pose.pose.orientation))

    # For debugging purposes
    def plot_warmstart(self): 
        warmstart_u, warmstart_x = self._planner.get_initial_guess() 
        for n in range(1, self._number_of_robots+1):
            cylinder = self._debug_visuals_pub.get_cylinder()
            cylinder.set_color(10, alpha=1.0)
            cylinder.set_scale(0.65, 0.65, 0.05)
            pose = Pose()
            for k in range(1, self._N):
                pose.position.x = float(warmstart_x[0 + (n-1)*self._nx_one_robot, k])
                pose.position.y = float(warmstart_x[1 + (n-1)*self._nx_one_robot, k])
                cylinder.add_marker(deepcopy(pose))

    def plot_path(self):
        dist = 0.2
        for n in range(1, self._number_of_robots+1):
            path_msg = getattr(self, f'_path_msg_{n}')
            spline_fitter = getattr(self, f'_spline_fitter_{n}')
            if path_msg is not None:
                line = self._path_visual.get_line()
                line.set_scale(0.05)
                line.set_color(1, alpha=1.0)
                points = self._path_visual.get_cube()
                points.set_color(3)
                points.set_scale(0.1, 0.1, 0.1)
                s = 0.0
                for i in range(50):
                    a = spline_fitter.evaluate(s)
                    b = spline_fitter.evaluate(s + dist)
                    pose_a = Pose()
                    pose_a.position.x = float(a[0])
                    pose_a.position.y = float(a[1])
                    points.add_marker(pose_a)
                    pose_b = Pose()
                    pose_b.position.x = float(b[0])
                    pose_b.position.y = float(b[1])
                    s += dist
                    line.add_line_from_poses(pose_a, pose_b)
        self._path_visual.publish()
        
    def print_stats(self):
        self._planner.print_stats()
        # self._decomp_constraints.print_stats()

    def print_contouring_ref(self):     # Not adjusted for multi robot
        s = self._state[6]
        x, y = self._spline_fitter.evaluate(s)
        print(f"Path at s = {s}: ({x}, {y})")
        print(f"State: ({self._spline_fitter._closest_x}, {self._spline_fitter._closest_y})")
    
    def plot_states(self):
        state_labels = ["x", "y", "theta", "vx", "vy", "omega", "s"]
        output_labels = ["throttle", "steering"]
        for n in range(1, self._number_of_robots + 1):
            plt.figure(figsize=(12, 6))
            
            # Plot states
            plt.subplot(1, 2, 1)
            num_states = len(state_labels)
            for i in range(num_states):
                state_values = [state[i] for state in getattr(self, f'_states_save_{n}')]
                plt.plot(state_values, label=state_labels[i])
            plt.xlabel('Time Step')
            plt.ylabel('State Values')
            plt.legend()
            plt.grid(True)
            plt.title(f'Robot {n} States')

            # Plot outputs
            plt.subplot(1, 2, 2)
            for i in range(len(output_labels)):
                output_values = [output[i] for output in getattr(self, f'_outputs_save_{n}')]
                plt.plot(output_values, label=output_labels[i])
            plt.xlabel('Time Step')
            plt.ylabel('Output Values')
            plt.legend()
            plt.grid(True)
            plt.title(f'Robot {n} Outputs')

            plt.tight_layout()
            plt.savefig(os.path.join(os.path.dirname(__file__), 'plots', f'states_outputs_plot_{n}.png'))  # Save the plot to a file
            plt.close()
    
    def plot_duals(self):
        plt.figure(figsize=(12, 6))
        

        # Plot s
        plt.subplot(1, 2, 1)
        plt.plot(self._save_s, label=['s_0', 's_1'])
        plt.xlabel('Time Step')
        plt.ylabel('s Values')
        plt.legend()
        plt.grid(True)
        plt.title('s')

        # Plot lams
        plt.subplot(1, 2, 2)
        plt.plot(self._save_lam, label=['lam_0', 'lam_1', 'lam_2', 'lam_3','lam_0', 'lam_1', 'lam_2', 'lam_3'])
        plt.xlabel('Time Step')
        plt.ylabel('Lam Values')
        plt.legend()
        plt.grid(True)
        plt.title('lam')

        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__), 'plots', f'duals_plot.png'))  # Save the plot to a file
        plt.close()

    def plot_pred_traj(self):
        state_labels = ["x", "y", "theta", "vx", "vy", "omega", "s"]
        time = np.linspace(0, (self._N-1) * self._integrator_step, self._N)

        plt.figure(figsize=(6, 12))
        for n in range(1, self._number_of_robots + 1):
            
            # Plot states
            plt.subplot(self._number_of_robots, 1, n)
            num_states = len(state_labels)
            plt.plot(time, self._trajectory[n * self._nx_one_robot : n +1 * self._nx_one_robot, :].T)
            plt.legend(state_labels)

            plt.xlabel('Time Steps')
            plt.ylabel('State Values')
            plt.legend()
            plt.grid(True)
            plt.title(f'Robot {n} Predictions')

        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__), 'plots', 'prediction_plot.png'))  # Save the plot to a file
        plt.close() 
    
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
    rospy.loginfo("Initializing MPC")
    rospy.init_node("dmpc_planner", anonymous=False)

    mpc = ROSMPCPlanner()

    while not rospy.is_shutdown():
        rospy.spin()
        
    # mpc.plot_outputs()
    mpc.plot_states()
    mpc.plot_duals()
    mpc.plot_min_distance()
    # mpc.plot_pred_traj()
    mpc.print_stats()
