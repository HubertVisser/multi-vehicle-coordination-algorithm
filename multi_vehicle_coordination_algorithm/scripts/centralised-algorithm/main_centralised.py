#!/usr/bin/env python3

import os, sys
import pathlib
path = pathlib.Path(__file__).parent.resolve()


# Add the project root directory to sys.path
sys.path.append("/home/dock_user/ros_ws/src/multi-vehicle-coordination-algorithm")
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
import functools

from solver_generator.util.files import load_settings
from solver_generator.util.realtime_parameters import RealTimeParameters
from solver_generator.util.convertion import quaternion_to_yaw, yaw_to_quaternion
from solver_generator.util.logging import print_value 
from timer import Timer

from contouring_spline import SplineFitter
from mpc_controller import MPCPlanner
from ros_visuals import ROSMarkerPublisher
from plot_utils import plot_duals, plot_distance, plot_trajectory, get_reference_from_path_msg, plot_slack_centralised
from helpers import get_robot_pairs_one, get_robot_pairs_both

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
        self._scenario = self._settings["track_choice"]

        self._planner = MPCPlanner(self._settings)
        self._spline_fitters = {n: SplineFitter(self._settings) for n in range(1, self._number_of_robots + 1)}

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
            self.get_state_per_robot(n)[:3] = [ self._settings[f"robot_{n}"]["start_x"], \
                                                self._settings[f"robot_{n}"]["start_y"], \
                                                self._settings[f"robot_{n}"]["start_theta"] * np.pi]

        self._visuals = ROSMarkerPublisher("mpc_visuals_1", 100)
        self._path_visual = ROSMarkerPublisher("reference_path_1", 10)
        self._debug_visuals_pub = ROSMarkerPublisher("mpc_planner_py/debug_1", 10)

        self._state_msgs = {n: None for n in range(1, self._number_of_robots + 1)}
        self._path_msgs = {n: None for n in range(1, self._number_of_robots + 1)}

        self._trajectory = None

        self._states_history = {n: [] for n in range(1, self._number_of_robots + 1)}
        self._outputs_history = {n: [] for n in range(1, self._number_of_robots + 1)}
        self._lam_history = []
        self._s_dual_history = []
        self._all_outputs = []

        self._cumulative_tracking_error = 0.0

        self._enable_output = False
        self._mpc_feasible = True

        self._timer = rospy.Timer(
            rospy.Duration(1.0 / self._settings["control_frequency"]), self.run
        )

        self.initialize_publishers_and_subscribers()
        self._enable_output = True

    def initialize_publishers_and_subscribers(self):

        for n in range(1, self._number_of_robots + 1):

            # Subscribers for the reference path
            topic = f"roadmap/reference_{n}"
            rospy.Subscriber(topic, Path, functools.partial(self.path_callback, robot_id=n), queue_size=1) 
            
            if not self._dart_simulator:
                continue

            self._throttle_pubs = {}
            self._steering_pubs = {}

            # Subscribers for the robot state  
            rospy.Subscriber(f"vicon/jetracer{n}", PoseStamped, functools.partial(self.state_pose_callback, robot_id=n), queue_size=1)
            rospy.Subscriber(f"vy_{n}", Float32, functools.partial(self.vy_pose_callback, robot_id=n), queue_size=1)    
            rospy.Subscriber(f"omega_{n}", Float32, functools.partial(self.w_pose_callback, robot_id=n), queue_size=1)

            # Publishers for control inputs
            self._throttle_pubs[n] = rospy.Publisher(f"throttle_{n}", Float32, queue_size=1)
            self._steering_pubs[n] = rospy.Publisher(f"steering_{n}", Float32, queue_size=1)


    def run(self, timer):
        # Check if splines exist
        for n in range(1, self._number_of_robots + 1):
            if not self._spline_fitters[n]._splines:
                rospy.logwarn(f"Splines for robot {n} have not been computed yet. Waiting for splines to be available.")
                return
        
        timer = Timer("loop")
        
        self.set_parameters()

        # self._params.print()
        mpc_timer = Timer("MPC")
        output, self._mpc_feasible, self._trajectory = self._planner.solve( 
            self._state, self._params.get_solver_params()
        )
        del mpc_timer

        if self._verbose:
            time = timer.stop_and_print()

        if self._mpc_feasible:

            lam = get_robot_pairs_both(self._number_of_robots)
            s = get_robot_pairs_one(self._number_of_robots)

            for i in range(1, self._number_of_robots+1): 
                output_keys = [f"x_{i}", f"y_{i}", f"theta_{i}", f"vx_{i}", f"vy_{i}", f"w_{i}", f"s_{i}"]

                if not self._dart_simulator:
                    self.get_state_per_robot(i)[:] = [output[key] for key in output_keys]
                
                self._states_history[i].append(deepcopy(self.get_state_per_robot(i)[:]))
                self._outputs_history[i].append([output[f"throttle_{i}"], output[f"steering_{i}"]])
                
                for j in range(1, self._number_of_robots+1):
                    if i == j:
                        continue
                    key = f"{i}_{j}"
                    lam[key] = [output[f"lam_{i}_{j}_0"], output[f"lam_{i}_{j}_1"], output[f"lam_{i}_{j}_2"], output[f"lam_{i}_{j}_3"]]
                    if key in s:
                        s[key] = [output[f"s_{i}_{j}_0"], output[f"s_{i}_{j}_1"]]
                
            self._lam_history.append(lam)
            self._s_dual_history.append(s)
            self._all_outputs.append(output)

            if self._dart_simulator:
                self.publish_throttle(output)
                self.publish_steering(output)
            
        self.visualize()
        

    def set_parameters(self):
        for n in range(1, self._number_of_robots + 1):
            spline_fitter = self._spline_fitters[n]
            state = self.get_state_per_robot(n)


            splines = None
            if self._path_msgs[n] is not None and spline_fitter.ready():
                splines = spline_fitter.get_active_splines(state[0:2])
                state[6] = spline_fitter.find_closest_s(state[0:2])

            # Set parameters for all k
            for k in range(self._N):

                # Tuning parameters
                for weight, value in self._weights.items():
                    self._params.set(k, weight, value)

                if splines is None:
                    continue
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

    def publish_throttle(self, output):
        for n in range(1, self._number_of_robots + 1):
            throttle = Float32()
            if not self._mpc_feasible or not self._enable_output:
                if not self._mpc_feasible:
                    rospy.logwarn_throttle(1, "Infeasible MPC. Braking!")
                    throttle.data = max(0.0, self.get_state_per_robot(n)[3] - self._braking_acceleration * self._integrator_step,)
                else:
                    rospy.logwarn_throttle(1, "Output is disabled. Sending zero velocity!")
                    throttle.data = 0.0
            else:
                throttle.data = output[f"throttle_{n}"]
                rospy.loginfo_throttle(1000, "MPC is driving")
                self._throttle_pubs[n].publish(throttle)
    
    def publish_steering(self, output):
        for n in range(1, self._number_of_robots + 1):
            steering = Float32()
            if not self._mpc_feasible or not self._enable_output:
                steering.data = 0.0
            else:
                steering.data = output[f"steering_{n}"]
                self._steering_pubs[n].publish(steering)
    
    def visualize(self):
        for n in range(1, self._number_of_robots + 1):
            self.update_tracking_error(n)
            self.visualize_robot_position(n)
            self.visualize_seperating_hyperplanes(n)
            self.visualize_debug(n)
            self.visualize_predicted_trajectory(n)
        self._visuals.publish()

    def visualize_robot_position(self, n):
        state_msg = self._state_msgs[n]
        if state_msg is None:
            return
        robot_pos = self._visuals.get_sphere()
        robot_pos.set_color(0)
        robot_pos.set_scale(0.3, 0.3, 0.3)
        pose = Pose()
        state = self.get_state_per_robot(n)
        pose.position.x = float(state[0])
        pose.position.y = float(state[1])
        robot_pos.add_marker(pose)

    def visualize_seperating_hyperplanes(self, n):
        if not self._s_dual_history:
            return
        line = self._visuals.get_line()
        line.set_scale(0.05)
        line.set_color(n * 7, alpha=1.0)
        pairs = self._s_dual_history[0]
        for pair in pairs:
            i, j = map(int, pair.split('_'))
            s = self._s_dual_history[-1][pair]
            pos1 = np.array(self.get_state_per_robot(i)[0:2])
            pos2 = np.array(self.get_state_per_robot(j)[0:2])
            midpoint = (pos1 + pos2) / 2
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

    def visualize_debug(self, n):
        if not self._debug_visuals:
            return
        spline_fitter = self._spline_fitters[n]
        if spline_fitter._closest_s is not None:
            cube = self._debug_visuals_pub.get_cube()
            cube.set_color(5 * n)
            cube.set_scale(0.5, 0.5, 0.5)
            pose = Pose()
            pose.position.x = float(spline_fitter._closest_x)
            pose.position.y = float(spline_fitter._closest_y)
            cube.add_marker(pose)
        self.plot_warmstart()
        self._debug_visuals_pub.publish()

    def visualize_predicted_trajectory(self, n):
        if self._trajectory is None or not self._mpc_feasible:
            return
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
    
    # Callback functions
    def state_pose_callback(self, msg, robot_id):
        self._state_msgs[robot_id] = msg
        state = self.get_state_per_robot(robot_id)
        state[0] = msg.pose.position.x
        state[1] = msg.pose.position.y
        state[2] = quaternion_to_yaw(msg.pose.orientation)
        state[3] = msg.pose.position.z

    def vy_pose_callback(self, msg, robot_id):
        self.get_state_per_robot(robot_id)[4] = msg.data

    def w_pose_callback(self, msg, robot_id):
        self.get_state_per_robot(robot_id)[5] = msg.data

    def path_callback(self, msg, robot_id):

        # Filter equal paths
        prev_msg = self._path_msgs[robot_id]
        if prev_msg is not None and len(prev_msg.poses) == len(msg.poses):
            return

        self._path_msgs[robot_id] = msg
        self._spline_fitters[robot_id].fit_path(msg)
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
            path_msg = self._path_msgs[n]
            spline_fitter = self._spline_fitters[n]
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
    
    def get_state_per_robot(self, robot_id):
        """
        Returns the full state vector segment for the given robot_id.
        """
        start = (robot_id - 1) * self._nx_one_robot
        end = robot_id * self._nx_one_robot
        return self._state[start:end]
        
    def print_stats(self):
        self._planner.print_stats()
        # self._decomp_constraints.print_stats()
    
    def plot_states(self):
        state_labels = ["x", "y", "theta", "vx", "vy", "omega", "s"]
        output_labels = ["throttle", "steering"]
        for n in range(1, self._number_of_robots + 1):
            plt.figure(figsize=(12, 6))
            
            # Plot states
            plt.subplot(1, 2, 1)
            num_states = len(state_labels)
            for i in range(num_states):
                state_values = [state[i] for state in self._states_history[n]]
                plt.plot(state_values, label=state_labels[i])
            plt.xlabel('Time Step')
            plt.ylabel('State Values')
            plt.legend()
            plt.grid(True)
            plt.title(f'Robot {n} States')

            # Plot outputs
            plt.subplot(1, 2, 2)
            for i in range(len(output_labels)):
                output_values = [output[i] for output in self._outputs_history[n]]
                plt.plot(output_values, label=output_labels[i])
            plt.xlabel('Time Step')
            plt.ylabel('Output Values')
            plt.legend()
            plt.grid(True)
            plt.title(f'Robot {n} Outputs')

            plt.tight_layout()
            plt.savefig(os.path.join(os.path.dirname(__file__), 'plots', f'states_{n}_{self._scheme}.png'))  # Save the plot to a file
            plt.close()
    
    def plot_duals(self):
        plot_duals(self._lam_history, self._s_dual_history, self._scheme)
    
    def plot_distance(self):    
        
        for pair in get_robot_pairs_one(self._number_of_robots):
            i, j = map(int, pair.split('_'))
            assert len(self._states_history[i]) == len(self._states_history[j]), "The two lists must have the same length."
            poses1 = self._states_history[i]
            poses2 = self._states_history[j]

            length = self._settings["polytopic"]["length"]
            width = self._settings["polytopic"]["width"]

            plot_distance(poses1, poses2, width, length, scheme=self._settings["scheme"])
    
    def plot_trajectory(self):
        reference_1 = get_reference_from_path_msg(self._path_msgs[1])
        reference_2 = get_reference_from_path_msg(self._path_msgs[2])

        plot_trajectory(np.array(self._states_history[1]), np.array(self._states_history[2]), reference_1, reference_2, self._settings['track_choice'], self._scheme)

    def update_tracking_error(self, robot_id):
        # Ensure _spline_fitter and _closest_s are available
        splineFitter = self._spline_fitters.get(robot_id)
        if splineFitter is None or splineFitter._closest_s is None:
            return

        robot_position = self.get_state_per_robot(robot_id)[0:2]
        closest_position = np.array([splineFitter._closest_x, splineFitter._closest_y])
        distance = np.linalg.norm(robot_position - closest_position)
        self._cumulative_tracking_error += distance
   
    def get_cumulative_tracking_error(self):
        return self._cumulative_tracking_error
    
    def log_tracking_error(self):
        rospy.loginfo(f"Cumulative Tracking Error: {self._cumulative_tracking_error:.2f}")
    
    def plot_slack(self):

        plot_slack_centralised(self._planner.get_slack_tracker())

    def save_states(self):
        centralised_traj_1 = np.array(self._states_history[1])
        centralised_traj_2 = np.array(self._states_history[2])

        data_dir = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'data')
        os.makedirs(data_dir, exist_ok=True)

        np.save(os.path.join(data_dir, f'1_{self._scenario}_centralised_traj.npy'), centralised_traj_1)
        np.save(os.path.join(data_dir, f'2_{self._scenario}_centralised_traj.npy'), centralised_traj_2)

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
    mpc.save_states()
    # mpc.plot_slack()

if __name__ == "__main__":
    run_centralised_algorithm()