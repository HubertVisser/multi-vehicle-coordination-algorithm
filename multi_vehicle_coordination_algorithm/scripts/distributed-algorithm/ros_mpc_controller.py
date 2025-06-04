
import os, sys
import pathlib
path = pathlib.Path(__file__).parent.resolve()
sys.path.append(os.path.join(path))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.join(sys.path[0], "..", "..", "..", "solver_generator"))
sys.path.append(os.path.join(sys.path[0], "..", "..", "..", "mpc_planner_modules"))

import threading

import rospy
from std_msgs.msg import Int32, Float32, Empty
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped, Pose, Twist
from multi_vehicle_coordination_algorithm.msg import LambdaArrayList, LambdaArray

import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt

from util.files import load_settings, load_model
from util.realtime_parameters import RealTimeParameters
from util.convertion import quaternion_to_yaw, yaw_to_quaternion
from util.logging import print_value 
from util.math import get_A, get_b

from timer import Timer
from spline import Spline, Spline2D
from dual_initialiser import set_initial_x_plan, get_all_initial_duals, dual_initialiser_previous

from contouring_spline import SplineFitter
from mpc_controller import MPCPlanner
from ros_visuals import ROSMarkerPublisher
from plot_utils import plot_warmstart, plot_path, plot_states, plot_duals, plot_slack_distributed



class ROSMPCPlanner:
    def __init__(self, idx, settings=None):
        self._settings = settings
        self._N = self._settings["N"]
        self._integrator_step = self._settings["integrator_step"]
        self._braking_acceleration = self._settings["braking_acceleration"]
        self._number_of_robots = self._settings["number_of_robots"]
        self._iterations = self._settings["solver_settings"]["iterations_distributed"]
        self._scheme = self._settings["scheme"]
        self._scenario = self._settings["track_choice"] #TODO: use one name
        self._idx = idx

        self._verbose = self._settings["verbose"]
        self._debug_visuals = self._settings["debug_visuals"]
        self._dart_simulator = self._settings["dart_simulator"]

        self._planner = MPCPlanner(self._settings, idx)
        self._spline_fitter = SplineFitter(self._settings)

        self._solver_settings_nmpc = load_settings(f"solver_settings_nmpc_{self._idx}", package="mpc_planner_solver")
        self._solver_settings_ca = load_settings(f"solver_settings_ca_{self._idx}", package="mpc_planner_solver")
        self._model_maps_ca = {n: load_model(f"model_map_ca_{n}", package="mpc_planner_solver") for n in range(1, self._number_of_robots + 1)}

        # Tied to the solver
        self._params_nmpc = RealTimeParameters(self._settings, parameter_map_name=f"parameter_map_nmpc_{idx}")  
        self._params_ca = RealTimeParameters(self._settings, parameter_map_name=f"parameter_map_ca_{idx}")  
        self._weights = self._settings["weights"]

        self._nx_nmpc = self._solver_settings_nmpc["nx"]
        self._nu_nmpc = self._solver_settings_nmpc["nu"]
        self._nlam = self._solver_settings_nmpc["nlam"]
        self._nvar_nmpc = self._solver_settings_nmpc["nvar"]
        self._nx_ca = self._solver_settings_ca["nx"]
        self._nu_ca = self._solver_settings_ca["nu"]
        self._nvar_ca = self._solver_settings_ca["nvar"]

        self._states_history = []
        self._outputs_history = []
        self._lam_history = []
        self._s_dual_history = []
        self._cumulative_tracking_error = 0.0
        
        self._state = np.zeros((self._nx_nmpc,))
        self._state[: 3] = [self._settings[f"robot_{self._idx}"]["start_x"], \
                            self._settings[f"robot_{self._idx}"]["start_y"], \
                            self._settings[f"robot_{self._idx}"]["start_theta"] * np.pi]
        
        self.initialise_duals()
        self.init_duals_dict = get_all_initial_duals(settings=self._settings)

        self._visuals = ROSMarkerPublisher(f"mpc_visuals_{self._idx}", 100)
        self._path_visual = ROSMarkerPublisher(f"reference_path_{self._idx}", 10)
        self._debug_visuals_pub = ROSMarkerPublisher(f"mpc_planner_py/debug_{self._idx}", 10)

        self._state_msg = None
        self._path_msg = None
        self._ca_solution = None

        self._trajectories = {n: np.zeros((3, self._N)) for n in range(1, self._number_of_robots + 1)}
        self._trajectory_received = {n: False for n in range(1, self._number_of_robots + 1) if n != self._idx}

        self._lambdas = {n: np.zeros((self._N, 4)) for n in range(1, self._number_of_robots + 1)}

        self._enable_output = False
        self._mpc_feasible = True
        self._ca_feasible = True
        self._r= 0

        self._callbacks_enabled = False
        self.initialize_publishers_and_subscribers()
        self._callbacks_enabled = True
        self._enable_output = True

    def initialize_publishers_and_subscribers(self):

        # Subscribers DART
        self._state_sub = rospy.Subscriber(f"vicon/jetracer{self._idx}", PoseStamped, self.state_pose_callback, queue_size=1)
        self._vy_sub = rospy.Subscriber(f"vy_{self._idx}", Float32, self.vy_pose_callback, queue_size=1)
        self._w_sub = rospy.Subscriber(f"omega_{self._idx}", Float32, self.w_pose_callback, queue_size=1)

        # Subscriber path generator
        self._path_sub = rospy.Subscriber(f"roadmap/reference_{self._idx}", Path, lambda msg: self.path_callback(msg), queue_size=1)
        
        for j in range(1, self._number_of_robots+1):
            setattr(self, f'_traj_{j}_sub', rospy.Subscriber(f"trajectory_{j}", Path, self.trajectory_callback, callback_args=j))
            # setattr(self, f'_lam_{j}_sub', rospy.Subscriber(f"lambda_{j}_{self._idx}", LambdaArrayList, self.lambda_callback, callback_args=j))
            # setattr(self, f'_lam_{j}_pub', rospy.Publisher(f"lambda_{j}_{self._idx}", LambdaArrayList, queue_size=1))
            
        # Publishers
        self._th_pub = rospy.Publisher(f"throttle_{self._idx}", Float32, queue_size=1) 
        self._st_pub = rospy.Publisher(f"steering_{self._idx}", Float32, queue_size=1) 
        self._traj_pub = rospy.Publisher(f"trajectory_{self._idx}", Path, queue_size=1)
        

    def run_nmpc(self, timer, it=None):
        if not it: it = self._iterations

        # Check if splines exist
        if not self._spline_fitter._splines:
            rospy.logwarn("Splines have not been computed yet. Waiting for splines to be available.")
            return
        
        self.set_nmpc_parameters()
        # self._params.check_for_nan()
        
        # if it == self._iterations:
        #     self._params_nmpc.write_to_file(f"params_output_multi_vehicle_nmpc_{self._idx}_call_{self._r}.txt")
        #     self._r += 1

        # self._params_nmpc.print() if self._idx == 1 else None
        mpc_timer = Timer("NMPC")

        output, self._mpc_feasible, trajectory = self._planner.solve_nmpc(self._state, self._params_nmpc.get_solver_params())
        
        del mpc_timer

        if self._verbose:
            time = timer.stop_and_print()

        if self._mpc_feasible:
            self.publish_trajectory(trajectory)
            self._trajectories[self._idx] = trajectory[:3, :]

            if it == self._iterations:
                if self._dart_simulator == False:
                    output_keys = [f"x_{self._idx}", f"y_{self._idx}", f"theta_{self._idx}", f"vx_{self._idx}", f"vy_{self._idx}", f"w_{self._idx}", f"s_{self._idx}"]
                    self._state = [output[key] for key in output_keys]
                    self._states_history.append(deepcopy(self._state))
                lam = {}
    
                self._outputs_history.append([output["throttle"], output["steering"]])
            
    def run_ca(self, timer, it):
        # Check if splines exist
        
        timer = Timer("loop")

        self.set_ca_parameters()
        # self._params.print()
        # self._params.check_for_nan()

        self._params_ca.print() if self._idx == 2 else None
        ca_timer = Timer("CA")
        output, self._ca_feasible, self._ca_solution = self._planner.solve_ca(self._params_ca.get_solver_params())
        del ca_timer

        if self._verbose:
            time = timer.stop_and_print()

        if self._ca_feasible and it == self._iterations:
            lam = {}
            s = {}
            for j in range(1, self._number_of_robots+1):
                if j == self._idx:
                    continue
                lam[f"lam_{self._idx}_{j}"] = [output[f"lam_{self._idx}_{j}_0"], 
                                               output[f"lam_{self._idx}_{j}_1"], 
                                               output[f"lam_{self._idx}_{j}_2"], 
                                               output[f"lam_{self._idx}_{j}_3"]
                                               ]
                lam[f"lam_{j}_{self._idx}"] = [output[f"lam_{j}_{self._idx}_0"], 
                                               output[f"lam_{j}_{self._idx}_1"], 
                                               output[f"lam_{j}_{self._idx}_2"], 
                                               output[f"lam_{j}_{self._idx}_3"]
                                               ]
                s[f"s_{self._idx}_{j}"] = [output[f"s_{self._idx}_{j}_0"], output[f"s_{self._idx}_{j}_1"]]
        
            self._lam_history.append(lam)
            self._s_dual_history.append(s)
        
            if self._dart_simulator:
                control_output = self._outputs_history[-1] 
                self.publish_throttle(control_output, self._mpc_feasible) 
                self.publish_steering(control_output, self._mpc_feasible) 
        
        for j in self._trajectory_received:
            self._trajectory_received[j] = False
        self.visualize()


    def set_nmpc_parameters(self):

        splines = None
        if self._path_msg is not None and self._spline_fitter.ready():
            splines = self._spline_fitter.get_active_splines(np.array([self._state[0], self._state[1]]))
            self._state[-1] = self._spline_fitter.find_closest_s(np.array([self._state[0], self._state[1]]))

        # Set parameters for all k
        for k in range(self._N):

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

        # Set neighbouring trajectories
        for j in range(1, self._number_of_robots+1):
            if j == self._idx:
                continue
            # Set neighbouring trajectories
            trajectory_j = self._trajectories[j]
            if np.all(trajectory_j == 0):
                xinit_j = np.array([self._settings[f"robot_{j}"]["start_x"], 
                                    self._settings[f"robot_{j}"]["start_y"], 
                                    self._settings[f"robot_{j}"]["start_theta"] * np.pi,
                                    ])
                x_plan_j = set_initial_x_plan(self._settings, xinit_j)
                for k in range(self._N):
                    self._params_nmpc.set(k, f"x_{j}", x_plan_j[0, k])
                    self._params_nmpc.set(k, f"y_{j}", x_plan_j[1, k])
                    self._params_nmpc.set(k, f"theta_{j}", x_plan_j[2, k])
            else:
                for k in range(self._N):
                    self._params_nmpc.set(k, f"x_{j}", trajectory_j[0, k])
                    self._params_nmpc.set(k, f"y_{j}", trajectory_j[1, k])
                    self._params_nmpc.set(k, f"theta_{j}", trajectory_j[2, k])

            # Set duals
            if self._ca_solution is None: 
                # Use initial duals
                dual_key = f"{self._idx}_{j}"
                dual_dict = self.init_duals_dict[dual_key]
                for key, value in dual_dict.items():
                    for k in range(self._N):
                        self._params_nmpc.set(k, key, value[k])
            else:
                # Use CA solution for duals
                model_map = self._model_maps_ca[self._idx]
                for key, value in model_map.items():
                        if value[0] == 'u':
                            idx = value[1]
                            for k in range(self._N):
                                self._params_nmpc.set(k, key, self._ca_solution[idx, k])
                        

    def set_ca_parameters(self):
        # Set parameters for all k
        for k in range(self._N):
            for weight, value in self._weights.items():
                self._params_ca.set(k, weight, value)

        # Set ego trajectory
        trajectory_i = self._trajectories[self._idx]
        
        if np.all(trajectory_i == 0):   # If CA is computed first
            xinit_i = np.array([self._settings[f"robot_{self._idx}"]["start_x"], self._settings[f"robot_{self._idx}"]["start_y"], self._settings[f"robot_{self._idx}"]["start_theta"] * np.pi])
            x_plan_i = set_initial_x_plan(self._settings, xinit_i)
            for k in range(self._N):
                if k != self._N - 1:
                    self._params_ca.set(k, f"x_{self._idx}", x_plan_i[0, k+1])
                    self._params_ca.set(k, f"y_{self._idx}", x_plan_i[1, k+1])
                    self._params_ca.set(k, f"theta_{self._idx}", x_plan_i[2, k+1])
                elif k == self._N - 1:
                    self._params_ca.set(k, f"x_{self._idx}", x_plan_i[0, k] + (x_plan_i[0, k] - x_plan_i[0, k-1]))
                    self._params_ca.set(k, f"y_{self._idx}", x_plan_i[1, k] + (x_plan_i[1, k] - x_plan_i[1, k-1]))
                    self._params_ca.set(k, f"theta_{self._idx}", x_plan_i[2, k] + (x_plan_i[2, k] - x_plan_i[2, k-1]))

        else:
            for k in range(self._N):
                if k != self._N - 1:
                    self._params_ca.set(k, f"x_{self._idx}", trajectory_i[0, k+1])
                    self._params_ca.set(k, f"y_{self._idx}", trajectory_i[1, k+1])
                    self._params_ca.set(k, f"theta_{self._idx}", trajectory_i[2, k+1])
                elif k == self._N - 1:
                    self._params_ca.set(k, f"x_{self._idx}", trajectory_i[0, k] + (trajectory_i[0, k] - trajectory_i[0, k - 1]))
                    self._params_ca.set(k, f"y_{self._idx}", trajectory_i[1, k] + (trajectory_i[1, k] - trajectory_i[1, k - 1]))
                    self._params_ca.set(k, f"theta_{self._idx}", trajectory_i[2, k] + (trajectory_i[2, k] - trajectory_i[2, k - 1]))
                    
        # Set neighbour trajectories
        for j in range(1, self._number_of_robots+1):
            if j == self._idx:
                continue

            # If the trajectory of neighbour j is not received yet, set with initial trajectory
            if self._trajectory_received[j] == False:
                xinit_j = np.array([self._settings[f"robot_{j}"]["start_x"], self._settings[f"robot_{j}"]["start_y"], self._settings[f"robot_{j}"]["start_theta"] * np.pi])
                x_plan_j = set_initial_x_plan(self._settings, xinit_j)
                for k in range(self._N):
                    if k != self._N-1:
                        self._params_ca.set(k, f"x_{j}", x_plan_j[0, k+1])
                        self._params_ca.set(k, f"y_{j}", x_plan_j[1, k+1])
                        self._params_ca.set(k, f"theta_{j}", x_plan_j[2, k+1])
                    elif k == self._N-1:
                        self._params_ca.set(k, f"x_{j}", x_plan_j[0, k])
                        self._params_ca.set(k, f"y_{j}", x_plan_j[1, k])
                        self._params_ca.set(k, f"theta_{j}", x_plan_j[2, k])
            else:
                trajectory_j = self._trajectories[j]
                for k in range(self._N):
                    self._params_ca.set(k, f"x_{j}", trajectory_j[0, k])
                    self._params_ca.set(k, f"y_{j}", trajectory_j[1, k])
                    self._params_ca.set(k, f"theta_{j}", trajectory_j[2, k])

    def initialise_duals(self):
        for j in range(1, self._number_of_robots+1):
            if j == self._idx:
                continue
            setattr(self, f'initial_duals_{j}', dual_initialiser_previous(self._settings, self._idx, j))              
                                            
    def publish_throttle(self, control, exit_flag):
        throttle = Float32()
        if not self._mpc_feasible or not self._enable_output:
            if not self._mpc_feasible:
                rospy.logwarn_throttle(1, "Infeasible MPC. Braking!")
                throttle.data = max(0.0, self._state[3] - self._braking_acceleration * self._integrator_step,)
            else:
                rospy.logwarn_throttle(1, "Output is disabled. Sending zero velocity!")
                throttle.data = 0.0
        else:
            throttle.data = control[0]
            rospy.loginfo_throttle(1000, "MPC is driving")
            self._th_pub.publish(throttle)
    
    def publish_steering(self, control, exit_flag):
        steering = Float32()
        if not self._mpc_feasible or not self._enable_output:
            steering.data = 0.0
        else:
            steering.data = control[1]
            self._st_pub.publish(steering)

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
            for k in range(1, self._N+1):
                pose = PoseStamped()
                pose.header.stamp = rospy.Time.now()
                pose.header.frame_id = "map"
                if k != self._N:
                    pose.pose.position.x = trajectory[0, k]
                    pose.pose.position.y = trajectory[1, k]
                    q = yaw_to_quaternion(trajectory[2, k])
                elif k == self._N:  # Interpolate the last point
                    pose.pose.position.x = trajectory[0, k - 1] + (trajectory[0, k - 1] - trajectory[0, k - 2])
                    pose.pose.position.y = trajectory[1, k - 1] + (trajectory[1, k - 1] - trajectory[1, k - 2])
                    q = yaw_to_quaternion(trajectory[2, k - 1] + (trajectory[2, k - 1] - trajectory[2, k - 2]))
                pose.pose.orientation.x = q[0]
                pose.pose.orientation.y = q[1]
                pose.pose.orientation.z = q[2]
                pose.pose.orientation.w = q[3]
                traj_msg.poses.append(pose)
            assert len(traj_msg.poses) == self._N
            
            self._traj_pub.publish(traj_msg)
    
    # Publish lambdas shifted by 1
    def publish_lambdas(self, lambdas):

        if lambdas is None:
            return
        
        for j in range(1, self._number_of_robots+1):
            if j == self._idx:
                continue
            arr_msg = LambdaArrayList()
            for k in range(1, self._N+1):
                lam_pub = getattr(self, f'_lam_{j}_pub')
                n = 0
                if k != self._N:
                    row = LambdaArray()
                    row.data = lambdas[n:n+4, k].tolist()
                    arr_msg.rows.append(row)
                elif k == self._N:  
                    row = LambdaArray()
                    row.data = lambdas[n:n+4, k-1].tolist()
                    arr_msg.rows.append(row)
                n += 1
            assert len(arr_msg.rows) == self._N
            lam_pub.publish(arr_msg)
                
    def visualize(self): 
        # Update the tracking error
        self.update_tracking_error()

        self.visualize_robot_position()
        self.visualize_seperating_hyperplane()
        self.visualize_debug_visuals()
        self.visualize_trajectory()

    def visualize_robot_position(self):
        if self._state_msg is None:
            return
        robot_pos = self._visuals.get_sphere()
        robot_pos.set_color(0)
        robot_pos.set_scale(0.3, 0.3, 0.3)

        pose = Pose()
        pose.position.x = float(self._state[0])
        pose.position.y = float(self._state[1])
        robot_pos.add_marker(pose)
    
    def visualize_seperating_hyperplane(self):
        if not self._s_dual_history:
            return
        line = self._visuals.get_line()
        line.set_scale(0.05)
        line.set_color(7**self._idx, alpha=1.0)
        ego_pos = np.array([self._state[0], self._state[1]])

        for j in range(1, self._number_of_robots+1):
            if j == self._idx:
                continue

            s = self._s_dual_history[-1][f's_{self._idx}_{j}'] 
            
            neighbour_pos = self._trajectories[j][:2, 0]
            midpoint = (ego_pos + neighbour_pos) / 2
    
            # Calculate the direction vector of the line (perpendicular to the normal vector)
            direction_vector = np.array([-s[1], s[0]]) 
            assert np.dot(direction_vector, np.array(s)) == 0
            line_length = 1000
            line_start = midpoint - (line_length / 2) * direction_vector
            line_end = midpoint + (line_length / 2) * direction_vector
            pose_a = Pose()
            pose_a.position.x = float(line_start[0])
            pose_a.position.y = float(line_start[1])
            pose_b = Pose()
            pose_b.position.x = float(line_end[0])
            pose_b.position.y = float(line_end[1])
            line.add_line_from_poses(pose_a, pose_b)

    def visualize_debug_visuals(self):
        if not self._debug_visuals:
            return
        if self._spline_fitter._closest_s is not None:
            cube = self._debug_visuals_pub.get_cube()
            cube.set_color(5)
            cube.set_scale(0.5, 0.5, 0.5)
            pose = Pose()
            pose.position.x = float(self._spline_fitter._closest_x)
            pose.position.y = float(self._spline_fitter._closest_y)
            cube.add_marker(pose)

        plot_warmstart(self)
        self._debug_visuals_pub.publish()

    def visualize_trajectory(self):
        
        if np.all(self._trajectories[self._idx] == 0):
            return
        
        length = self._settings["polytopic"]["length"]
        width = self._settings["polytopic"]["width"]

        box = self._visuals.get_cube()
        box.set_color(80, alpha=0.3)
        box.set_scale(width, length, 0.05)
        
        pose = Pose()
        for k in range(1, self._N):
            pose.position.x = self._trajectories[self._idx][0, k]
            pose.position.y = self._trajectories[self._idx][1, k]
            theta = self._trajectories[self._idx][2, k]
            quaternion = yaw_to_quaternion(theta)
            pose.orientation.x = quaternion[0]
            pose.orientation.y = quaternion[1]
            pose.orientation.z = quaternion[2]
            pose.orientation.w = quaternion[3]
            box.add_marker(deepcopy(pose))
        self._visuals.publish()
    
    # Callback functions
    def state_pose_callback(self, msg):
        if not self._dart_simulator:
            return
        self._state_msg = msg
        self._state[0] = msg.pose.position.x
        self._state[1] = msg.pose.position.y

        # Extract yaw angle (rotation around the Z-axis)
        self._state[2] = quaternion_to_yaw(msg.pose.orientation)

        # Velocity is in the local frame, x is the forward velocity
        self._state[3] = msg.pose.position.z

        self._states_history.append(deepcopy(self._state))
 

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
        plot_path(self)

    def trajectory_callback(self, traj_msg, j):

        if not traj_msg.poses:
            rospy.logwarn(f"Received empty path robot {j}")
            return
        if j == self._idx:
            return
        
        
        for k, pose in enumerate(traj_msg.poses):
            self._trajectories[j][0, k] = pose.pose.position.x
            self._trajectories[j][1, k] = pose.pose.position.y
            self._trajectories[j][2, k] = quaternion_to_yaw(pose.pose.orientation)
        self._trajectory_received[j] = True
    
    def lambda_callback(self, msg, j):
        if not msg.rows:
            rospy.logwarn(f"Received empty lambda robot {j}")
            return
        if j == self._idx:
            return

        for k, lam_msg in enumerate(msg.rows):
            self._lambdas[j][k, :] = [lam_msg.data[0], lam_msg.data[1], lam_msg.data[2], lam_msg.data[3]]
                
    def print_stats(self):
        self._planner.print_stats()
        # self._decomp_constraints.print_stats()
    
    def plot_states(self):
        plot_states(self)
    
    def plot_duals(self):
        plot_duals(self._lam_history, self._s_dual_history, self._scheme)

    def update_tracking_error(self):
        # Ensure _spline_fitter and _closest_s are available
        if self._spline_fitter is None or self._spline_fitter._closest_s is None:
            return

        robot_position = np.array([self._state[0], self._state[1]])
        closest_position = np.array([self._spline_fitter._closest_x, self._spline_fitter._closest_y])
        distance = np.linalg.norm(robot_position - closest_position)
        self._cumulative_tracking_error += distance
    
    def get_cumulative_tracking_error(self):
        return self._cumulative_tracking_error
    
    def log_tracking_error(self):
        rospy.loginfo(f"Cumulative Tracking Error: {self._cumulative_tracking_error:.2f}")
    
    def plot_slack(self):

        slack_ca = self._planner.get_slack_tracker_ca()
        slack_nmpc = self._planner.get_slack_tracker_nmpc()

        plot_slack_distributed(slack_nmpc, slack_ca)

    def all_neighbor_trajectories_received(self):
        # Exclude self._idx
        return all(self._trajectory_received[j] for j in range(1, self._number_of_robots + 1) if j != self._idx)
    
    def load_centralised_traj(self):
        """Load trajectory from file and set self._states_history."""
        script_dir = os.path.abspath(os.path.dirname(__file__))
        data_dir = os.path.join(script_dir, '..', '..', '..', '..', 'data')
        traj_path = os.path.join(data_dir, f"{self._idx}_{self._scenario}_centralised_traj.npy")
        if not os.path.exists(traj_path):
            print(f"File {traj_path} does not exist.")
            self._states_history = []
            return
        states_centralised = np.load(traj_path)
        print(f"Loaded {len(states_centralised)} states from {traj_path}")
        return states_centralised

    def evaluate_tracking_error(self):
        """Evaluate cumulative tracking error between states_centralised and self._states_history."""
        states_centralised = self.load_centralised_traj()
        if states_centralised is None or len(self._states_history) == 0:
            print("No data to compare.")
            return

        # Ensure both arrays are the same length
        n = min(len(states_centralised), len(self._states_history))
        cumulative_tracking_error_centralised = 0.0

        for i in range(n):
            pos_centralised = np.array([states_centralised[i, 0], states_centralised[i, 1]])
            pos_distributed = np.array([self._states_history[i][0], self._states_history[i][1]])
            distance = np.linalg.norm(pos_centralised - pos_distributed)
            cumulative_tracking_error_centralised += distance

        rospy.loginfo(f"Cumulative Tracking Error {self._idx} With Centralised: {cumulative_tracking_error_centralised}")


if __name__ == "__main__":
    rospy.loginfo("Initializing MPC")
    rospy.init_node("coordination-algorithm", anonymous=False)

    mpc = ROSMPCPlanner()

    while not rospy.is_shutdown():
        rospy.spin()
        
    # mpc.plot_outputs()
    plot_states(mpc)
    plot_duals(mpc)
    # mpc.plot_pred_traj()
    plot_states(mpc)
