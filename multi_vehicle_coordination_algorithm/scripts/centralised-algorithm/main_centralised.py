#!/usr/bin/env python3

import os, sys
import pathlib
path = pathlib.Path(__file__).parent.resolve()


# Add the project root directory to sys.path
# sys.path.append("/home/dock_user/ros_ws/src/multi-vehicle-coordination-algorithm")
sys.path.append(os.path.join(path))
sys.path.append(os.path.join(sys.path[-1], "..", "..", "..", "solver_generator"))
sys.path.append(os.path.join(sys.path[-2], "..", "..", "..", "mpc_planner_modules"))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import threading

import rospy
from std_msgs.msg import Int32, Float32, Empty
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped, Pose, Twist

import numpy as np
from copy import deepcopy
import math
import matplotlib.pyplot as plt

from solver_generator.util.files import load_settings
from solver_generator.util.realtime_parameters import RealTimeParameters
from solver_generator.util.convertion import quaternion_to_yaw, yaw_to_quaternion
from solver_generator.util.logging import print_value 
from timer import Timer

from contouring_spline import SplineFitter
from mpc_controller import MPCPlanner
from ros_visuals import ROSMarkerPublisher
from plot_utils import plot_duals, plot_distance, plot_trajectory, get_reference_from_path_msg, plot_slack_centralised


class ROSMPCPlanner:
    def __init__(self):
        self._settings = load_settings(package="multi_vehicle_coordination_algorithm")
        self._N = self._settings["N"]
        self._integrator_step = self._settings["integrator_step"]
        self._braking_acceleration = self._settings["braking_acceleration"]
        self._number_of_robots = self._settings["number_of_robots"]

        self._verbose = self._settings["verbose"]
        self._debug_visuals = self._settings["debug_visuals"]
        self._dart_simulator = self._settings["dart_simulator"]
        self._scheme = self._settings["scheme"]

        self._planner = MPCPlanner(self._settings)

        self._spline_fitter_1 = SplineFitter(self._settings)
        self._spline_fitter_2 = SplineFitter(self._settings)

        self._solver_settings = load_settings(
            "solver_settings_centralised", package="mpc_planner_solver"
        )

        # Tied to the solver
        self._params = RealTimeParameters(self._settings, parameter_map_name='parameter_map_centralised' )  # This maps to parameters used in the solver by name
        self._weights = self._settings["weights"]

        self._nx = self._solver_settings["nx"]
        self._nu = self._solver_settings["nu"]
        self._nx_one_robot = self._nx // self._number_of_robots
        self._nu_one_robot = self._nu // self._number_of_robots

        self._state = np.zeros((self._nx,))
        for n in range(1, self._number_of_robots+1):
            self._state[(n-1)*self._nx_one_robot: 3 + (n-1)*self._nx_one_robot] = [ self._settings[f"robot_{n}"]["start_x"], \
                                                                                    self._settings[f"robot_{n}"]["start_y"], \
                                                                                    self._settings[f"robot_{n}"]["start_theta"] * np.pi]

        self._visuals = ROSMarkerPublisher("mpc_visuals_1", 100)
        self._path_visual = ROSMarkerPublisher("reference_path_1", 10)
        self._debug_visuals_pub = ROSMarkerPublisher("mpc_planner_py/debug_1", 10)

        self._state_msg_1 = None
        self._state_msg_2 = None

        self._path_msg_1 = None
        self._path_msg_2 = None

        self._trajectory = None

        self._states_save_1 = []
        self._states_save_2 = []
        self._outputs_save_1 = []
        self._outputs_save_2 = []
        self._save_output = []
        self._save_lam = []
        self._save_s = []
        self._cumulative_tracking_error = 0.0


        self._neighbour_pos_1 = np.array([])
        self._neighbour_pos_2 = np.array([])

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

        # Subscribers
        self._state_sub_1 = rospy.Subscriber("vicon/jetracer1", PoseStamped, self.state_pose_callback_1, queue_size=1)
        self._vy_sub_1 = rospy.Subscriber("vy_1", Float32, self.vy_pose_callback_1, queue_size=1)
        self._w_sub_1 = rospy.Subscriber("omega_1", Float32, self.w_pose_callback_1, queue_size=1)
        self._state_sub_2 = rospy.Subscriber("vicon/jetracer2", PoseStamped, self.state_pose_callback_2, queue_size=1)
        self._vy_sub_2 = rospy.Subscriber("vy_2", Float32, self.vy_pose_callback_2, queue_size=1)
        self._w_sub_2 = rospy.Subscriber("omega_2", Float32, self.w_pose_callback_2, queue_size=1)

        self._path_sub_1 = rospy.Subscriber("roadmap/reference_1", Path, lambda msg: self.path_callback_1(msg), queue_size=1)
        self._path_sub_2 = rospy.Subscriber("roadmap/reference_2", Path, lambda msg: self.path_callback_2(msg), queue_size=1)

        # Publishers
        self._th_pub_1 = rospy.Publisher("throttle_1", Float32, queue_size=1) # Throttle publisher
        self._st_pub_1 = rospy.Publisher("steering_1", Float32, queue_size=1) # Steering publisher
        self._th_pub_2 = rospy.Publisher("throttle_2", Float32, queue_size=1)
        self._st_pub_2 = rospy.Publisher("steering_2", Float32, queue_size=1)

    def run(self, timer):
        # Check if splines exist
        if not self._spline_fitter_1._splines:
            rospy.logwarn("Splines have not been computed yet. Waiting for splines to be available.")
            return
        if self._number_of_robots > 1 and not self._spline_fitter_2._splines:
            rospy.logwarn("Splines have not been computed yet. Waiting for splines to be available.")
            return
        
        timer = Timer("loop")
        
        self.set_parameters()
        # self._params.print()
        # self._params.check_for_nan()

        mpc_timer = Timer("MPC")
        output, self._mpc_feasible, self._trajectory = self._planner.solve( 
            self._state, self._params.get_solver_params()
        )
        del mpc_timer

        if self._verbose:
            time = timer.stop_and_print()

        if self._mpc_feasible:
            lam = {}
            s = {}
            for i in range(1, self._number_of_robots+1): 
                output_keys = [f"x_{i}", f"y_{i}", f"theta_{i}", f"vx_{i}", f"vy_{i}", f"w_{i}", f"s_{i}"]
                
                if not self._dart_simulator:
                    self._state[(i-1) * self._nx_one_robot : self._nx_one_robot * (i)] = [output[key] for key in output_keys]
                    getattr(self, f'_states_save_{i}').append(deepcopy(self._state[(i-1)*self._nx_one_robot : i*self._nx_one_robot ]))
                
                getattr(self, f'_outputs_save_{i}').append([output[f"throttle_{i}"], output[f"steering_{i}"]])
                
                for j in range(1, self._number_of_robots+1):
                    if i == j:
                        continue
                    lam[f"lam_{i}_{j}"] = [output[f"lam_{i}_{j}_0"], output[f"lam_{i}_{j}_1"], output[f"lam_{i}_{j}_2"], output[f"lam_{i}_{j}_3"]]
                    lam[f"lam_{j}_{i}"] = [output[f"lam_{j}_{i}_0"], output[f"lam_{j}_{i}_1"], output[f"lam_{j}_{i}_2"], output[f"lam_{j}_{i}_3"]]
                    if i < j:
                        s[f"s_{i}_{j}"] = [output[f"s_{i}_{j}_0"], output[f"s_{i}_{j}_1"]]
                
            self._save_lam.append(lam)
            self._save_s.append(s)

            # self.plot_pred_traj() # slows down the simulation

        self.publish_throttle(output, self._mpc_feasible) if self._dart_simulator else None
        self.publish_steering(output, self._mpc_feasible) if self._dart_simulator else None
            
        self.visualize()
        _, _, calls = self._planner.time_tracker.get_stats()
        print_value("calls", calls)
        
        # self.publish_robot_state() # Not used in simulator.rviz

    def set_parameters(self):
        for n in range(1, self._number_of_robots + 1):
            path_msg = getattr(self, f'_path_msg_{n}')
            spline_fitter = getattr(self, f'_spline_fitter_{n}')

            splines = None
            if path_msg is not None and spline_fitter.ready():
                splines = spline_fitter.get_active_splines(
                    np.array([ self._state[ 0 + (n-1) * self._nx_one_robot ], self._state[1 + (n-1) * self._nx_one_robot ]])
                )
                self._state[n * self._nx_one_robot-1] = spline_fitter.find_closest_s(
                    np.array([ self._state[ 0 + (n-1) * self._nx_one_robot ], self._state[1 + (n-1) * self._nx_one_robot ]])
                )

            # Set parameters for all k
            for k in range(self._N):

                # Tuning parameters
                for weight, value in self._weights.items():
                    self._params.set(k, weight, value)

                if splines is not None:
                    for i in range(self._settings["contouring"]["num_segments"]):
                        self._params.set(k, f"spline_x{i}_a_{n}", splines[i]["a_x"])
                        self._params.set(k, f"spline_x{i}_b_{n}", splines[i]["b_x"])
                        self._params.set(k, f"spline_x{i}_c_{n}", splines[i]["c_x"])
                        self._params.set(k, f"spline_x{i}_d_{n}", splines[i]["d_x"])

                        self._params.set(k, f"spline_y{i}_a_{n}", splines[i]["a_y"])
                        self._params.set(k, f"spline_y{i}_b_{n}", splines[i]["b_y"])
                        self._params.set(k, f"spline_y{i}_c_{n}", splines[i]["c_y"])
                        self._params.set(k, f"spline_y{i}_d_{n}", splines[i]["d_y"])

                        self._params.set(k, f"spline{i}_start_{n}", splines[i]["s"])

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
    
    def visualize(self):
        for n in range(1, self._number_of_robots+1):
            self.update_tracking_error(n)

            state_msg = getattr(self, f'_state_msg_{n}')
            splineFitter = self.get_spline_fitter(n)
            if state_msg:
                robot_pos = self._visuals.get_sphere()
                robot_pos.set_color(0)
                robot_pos.set_scale(0.3, 0.3, 0.3)

                pose = Pose()
                pose.position.x = float(self._state[0 + (n-1) * self._nx_one_robot])
                pose.position.y = float(self._state[1 + (n-1) * self._nx_one_robot])
                robot_pos.add_marker(pose)
            if self._save_s:
                line = self._visuals.get_line()
                line.set_scale(0.05)
                line.set_color(n*7, alpha=1.0)
                ego_pos = np.array([self._state[0 + (n-1)*self._nx_one_robot], self._state[1 + (n-1)*self._nx_one_robot]])

                for j in range(n, self._number_of_robots+1):
                    if j == n:
                        continue

                    s = self._save_s[-1][f's_{n}_{j}'] if n < j else self._save_s[-1][f's_{j}_{n}']

                    #setattr(self, f'neighbour_pos_{j}', np.array([]))
                    neighbour_pos = np.array([self._state[0 + (j-1)*self._nx_one_robot], self._state[1 + (j-1)*self._nx_one_robot]])
                    midpoint = (ego_pos + neighbour_pos) / 2
            
                    # Calculate the direction vector of the line (perpendicular to the normal vector)
                    direction_vector = np.array([-s[1], s[0]]) 
                    assert np.dot(direction_vector, np.array(s)) == 0
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
                length = self._settings["polytopic"]["length"]
                width = self._settings["polytopic"]["width"]

                box = self._visuals.get_cube()
                box.set_color(80, alpha=0.3)
                box.set_scale(width, length, 0.05)
                
                pose = Pose()
                for k in range(1, self._N):
                    pose.position.x = self._planner.get_model().get(k, f"x_{n}")
                    pose.position.y = self._planner.get_model().get(k, f"y_{n}")
                    theta = self._planner.get_model().get(k, f"theta_{n}")
                    quaternion = yaw_to_quaternion(theta)
                    pose.orientation.x = quaternion[0]
                    pose.orientation.y = quaternion[1]
                    pose.orientation.z = quaternion[2]
                    pose.orientation.w = quaternion[3]
                    box.add_marker(deepcopy(pose))
        self._visuals.publish()
    
    # Callback functions
    def state_pose_callback_1(self, msg):
        if self._dart_simulator:
            self._state_msg_1 = msg
            self._state[0] = msg.pose.position.x
            self._state[1] = msg.pose.position.y

            # Extract yaw angle (rotation around the Z-axis)
            self._state[2] = quaternion_to_yaw(msg.pose.orientation)

            # Velocity is in the local frame, x is the forward velocity
            self._state[3] = msg.pose.position.z

            self._states_save_1.append(deepcopy(self._state[:self._nx_one_robot]))
            # print("-------- State ----------")
            # print(f"x = {self._state[0]:.2f}")
            # print(f"y = {self._state[1]:.2f}")
            # print(f"theta = {self._state[2]:.2f}")
            # print(f"vx = {self._state[3]:.2f}")
    
    def state_pose_callback_2(self, msg):
        if self._dart_simulator and self._number_of_robots > 1:
            self._state_msg_2 = msg
            self._state[7] = msg.pose.position.x
            self._state[8] = msg.pose.position.y

            # Extract yaw angle (rotation around the Z-axis)
            self._state[9] = quaternion_to_yaw(msg.pose.orientation)

            # Velocity is in the local frame, x is the forward velocity
            self._state[10] = msg.pose.position.z

            self._states_save_2.append(deepcopy(self._state[self._nx_one_robot:]))

    def vy_pose_callback_1(self, msg):
        if self._dart_simulator:
            self._state[4] = msg.data

    def vy_pose_callback_2(self, msg):
        if self._dart_simulator and self._number_of_robots > 1:
            self._state[11] = msg.data
    
    def w_pose_callback_1(self, msg):
        if self._dart_simulator:
            self._state[5] = msg.data
    
    def w_pose_callback_2(self, msg):
        if self._dart_simulator and self._number_of_robots > 1:
            self._state[12] = msg.data

    def path_callback_1(self, msg):

        # Filter equal paths
        if self._path_msg_1 is not None and len(self._path_msg_1.poses) == len(msg.poses):
            return

        self._path_msg_1 = msg
        self._spline_fitter_1.fit_path(msg)
        # plot_splines(self._spline_fitter_1._splines)
        self.plot_path()
    
    def path_callback_2(self, msg):

        # Filter equal paths
        if self._path_msg_2 is not None and len(self._path_msg_2.poses) == len(msg.poses) or self._number_of_robots < 1:
            return

        self._path_msg_2 = msg
        self._spline_fitter_2.fit_path(msg)
        # plot_splines(self._spline_fitter_2._splines)
        self.plot_path()

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
            if path_msg is None:
                continue
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
            plt.savefig(os.path.join(os.path.dirname(__file__), 'plots', f'states_{n}_{self._scheme}.png'))  # Save the plot to a file
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
    
    def plot_duals(self):
        plot_duals(self, "1-2")
    
    def plot_distance(self):    
        
        assert len(self._states_save_1) == len(self._states_save_2), "The two lists must have the same length."
        poses1 = self._states_save_1
        poses2 = self._states_save_2

        length = self._settings["polytopic"]["length"]
        width = self._settings["polytopic"]["width"]

        plot_distance(poses1, poses2, width, length, scheme=self._settings["scheme"])
    
    def plot_trajectory(self):
        
        reference_1 = get_reference_from_path_msg(self._path_msg_1)
        reference_2 = get_reference_from_path_msg(self._path_msg_2)

        plot_trajectory(np.array(self._states_save_1), np.array(self._states_save_2), reference_1, reference_2, self._settings['track_choice'], self._scheme)

    def update_tracking_error(self, robot_id):
        # Ensure _spline_fitter and _closest_s are available
        splineFitter = self.get_spline_fitter(robot_id)
        if splineFitter is None or splineFitter._closest_s is None:
            return

        robot_position = self.get_state(robot_id)
        closest_position = np.array([splineFitter._closest_x, splineFitter._closest_y])
        distance = np.linalg.norm(robot_position - closest_position)
        self._cumulative_tracking_error += distance
    
    def get_spline_fitter(self, robot_id):
        attribute_name = f'_spline_fitter_{robot_id}'
        if not hasattr(self, attribute_name):
            raise AttributeError(f"robot_id '{robot_id}' does not exist.")
        return getattr(self, attribute_name)

    def get_state(self, robot_id):
        return np.array([self._state[0 + (robot_id-1)*self._nx_one_robot], self._state[1 + (robot_id-1)*self._nx_one_robot]])
        
    def get_cumulative_tracking_error(self):
        return self._cumulative_tracking_error
    
    def log_tracking_error(self):
        rospy.loginfo(f"Cumulative Tracking Error: {self._cumulative_tracking_error:.2f}")
    
    def plot_slack(self):

        plot_slack_centralised(self._planner.get_slack_tracker())

def run_centralised_algorithm():
    """
    Initializes the ROS node and runs the centralised coordination algorithm.
    """
    rospy.loginfo("Initializing MPC")
    rospy.init_node("multi_vehicle_coordination_algorithm", anonymous=False)

    mpc = ROSMPCPlanner()

    while not rospy.is_shutdown():
        rospy.spin()
        
    # mpc.plot_outputs()
    mpc.plot_states()
    mpc.plot_duals()
    mpc.plot_distance()
    mpc.plot_trajectory()
    mpc.log_tracking_error()
    mpc.print_stats()
    mpc.plot_distance()
    mpc.plot_slack()

if __name__ == "__main__":
    run_centralised_algorithm()