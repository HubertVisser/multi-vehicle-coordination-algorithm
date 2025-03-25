
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

from mpc_planner_msgs.msg import ObstacleArray
from mpc_planner_msgs.msg import WeightArray

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
from dual_initialiser import dual_initialiser, set_initial_x_plan

from contouring_spline import SplineFitter
from mpc_controller import MPCPlanner
from ros_visuals import ROSMarkerPublisher
from plot_utils import plot_warmstart, plot_path, plot_states, plot_duals


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
        self._spline_fitter = SplineFitter(self._settings)

        self._solver_settings_nmpc = load_settings(f"solver_settings_nmpc_{self._idx}", package="mpc_planner_solver")
        self._solver_settings_ca = load_settings(f"solver_settings_ca_{self._idx}", package="mpc_planner_solver")

        # Tied to the solver
        self._params_nmpc = RealTimeParameters(self._settings, parameter_map_name=f"parameter_map_nmpc_{idx}")  
        self._params_ca = RealTimeParameters(self._settings, parameter_map_name=f"parameter_map_ca_{idx}")  
        self._weights = self._settings["weights"]

        self._nx_nmpc = self._solver_settings_nmpc["nx"]
        self._nu_nmpc = self._solver_settings_nmpc["nu"]
        self._nvar_nmpc = self._solver_settings_nmpc["nvar"]
        self._nx_ca = self._solver_settings_ca["nx"]
        self._nu_ca = self._solver_settings_ca["nu"]
        self._nvar_ca = self._solver_settings_ca["nvar"]

        self._states_save = []
        self._outputs_save = []
        self._save_output = []
        self._save_lam = []
        self._save_s = []
        
        self._state = np.zeros((self._nx_nmpc,))
        self._state[: 3] = [self._settings[f"robot_{self._idx}"]["start_x"], \
                            self._settings[f"robot_{self._idx}"]["start_y"], \
                            self._settings[f"robot_{self._idx}"]["start_theta"] * np.pi]
        
        self._uinit = np.zeros(self._nu_ca)
        self.initialise_duals()
 

        self._visuals = ROSMarkerPublisher(f"mpc_visuals_{self._idx}", 100)
        self._path_visual = ROSMarkerPublisher(f"reference_path_{self._idx}", 10)
        self._debug_visuals_pub = ROSMarkerPublisher(f"mpc_planner_py/debug_{self._idx}", 10)

        self._state_msg = None
        self._path_msg = None
        self._ca_solution = None

        for n in range(1, self._number_of_robots+1):
            setattr(self, f'_trajectory_{n}', np.zeros((3, self._N)))

        self._enable_output = False
        self._mpc_feasible = True
        self._ca_feasible = True

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
        self._traj_pub = rospy.Publisher(f"trajectory_{self._idx}", Path, queue_size=1)

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
        # self._params_nmpc.plot_parameters(["s_1_2_0", "s_1_2_1", "lam_1_2_0", "lam_1_2_1", "lam_1_2_2", "lam_1_2_3"])
        
        mpc_timer = Timer("NMPC")

        output, self._mpc_feasible, trajectory = self._planner.solve_nmpc(self._state, self._params_nmpc.get_solver_params())
        setattr(self, f'_trajectory_{self._idx}', trajectory)

        del mpc_timer

        if self._verbose:
            time = timer.stop_and_print()

        if self._mpc_feasible:
            if self._dart_simulator == False:
                output_keys = [f"x_{self._idx}", f"y_{self._idx}", f"theta_{self._idx}", f"vx_{self._idx}", f"vy_{self._idx}", f"w_{self._idx}", f"s_{self._idx}"]
                self._state = [output[key] for key in output_keys]
                self._states_save.append(deepcopy(self._state))
            
            self._outputs_save.append([output["throttle"], output["steering"]])
            # self.plot_pred_traj() # slows down the simulation

            self.publish_trajectory(trajectory) 
            self.visualize()
    
    def run_ca(self, timer):
        # Check if splines exist
        
        timer = Timer("loop")

        self.set_ca_parameters()
        # self._params.print()
        # self._params.check_for_nan()

        ca_timer = Timer("CA")
        # self._params_ca.plot_parameters(["x_1", "x_2", "y_1", "y_2", "theta_1", "theta_2"])
        output, self._ca_feasible, self._ca_solution = self._planner.solve_ca(self._uinit, self._params_ca.get_solver_params())
        del ca_timer

        if self._verbose:
            time = timer.stop_and_print()

        if self._ca_feasible:
            lam = {}
            s = {}
            for j in range(1, self._number_of_robots+1):
                if self._idx != j:
                        lam[f"lam_{self._idx}_{j}"] = [output[f"lam_{self._idx}_{j}_0"], output[f"lam_{self._idx}_{j}_1"], output[f"lam_{self._idx}_{j}_2"], output[f"lam_{self._idx}_{j}_3"]]
                        lam[f"lam_{j}_{self._idx}"] = [output[f"lam_{j}_{self._idx}_0"], output[f"lam_{j}_{self._idx}_1"], output[f"lam_{j}_{self._idx}_2"], output[f"lam_{j}_{self._idx}_3"]]
                        if j > self._idx:
                            s[f"s_{self._idx}_{j}"] = [output[f"s_{self._idx}_{j}_0"], output[f"s_{self._idx}_{j}_1"]]
                        else:
                            s[f"s_{j}_{self._idx}"] = [output[f"s_{j}_{self._idx}_0"], output[f"s_{j}_{self._idx}_1"]]
                
            self._save_lam.append(lam)
            self._save_s.append(s)
        
        control_output = self._outputs_save[-1]
        self.publish_throttle(control_output, self._mpc_feasible) if self._dart_simulator else None
        self.publish_steering(control_output, self._mpc_feasible) if self._dart_simulator else None
        
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

            for j in range(1, self._number_of_robots+1):
                if j != self._idx:
                    trajectory = getattr(self, f'_trajectory_{j}')
                    if np.all(trajectory == 0):
                        xinit_j = np.array([self._settings[f"robot_{j}"]["start_x"], self._settings[f"robot_{j}"]["start_y"], self._settings[f"robot_{j}"]["start_theta"] * np.pi])
                        x_plan_j = set_initial_x_plan(self._settings, xinit_j)
                        self._params_nmpc.set(k, f"x_{j}", x_plan_j[0, k])
                        self._params_nmpc.set(k, f"y_{j}", x_plan_j[1, k])
                        self._params_nmpc.set(k, f"theta_{j}", x_plan_j[2, k])
                    else:
                        self._params_nmpc.set(k, f"x_{j}", trajectory[0, k])
                        self._params_nmpc.set(k, f"y_{j}", trajectory[1, k])
                        self._params_nmpc.set(k, f"theta_{j}", trajectory[2, k])

                    if self._ca_solution is None:
                        # Set initial duals
                        initial_duals = getattr(self, f'initial_duals_{j}')
                        self._params_nmpc.set(k, f"lam_{self._idx}_{j}_0", initial_duals[k][0])
                        self._params_nmpc.set(k, f"lam_{self._idx}_{j}_1", initial_duals[k][1])
                        self._params_nmpc.set(k, f"lam_{self._idx}_{j}_2", initial_duals[k][2])
                        self._params_nmpc.set(k, f"lam_{self._idx}_{j}_3", initial_duals[k][3])

                        self._params_nmpc.set(k, f"lam_{j}_{self._idx}_0", initial_duals[k][4])
                        self._params_nmpc.set(k, f"lam_{j}_{self._idx}_1", initial_duals[k][5])
                        self._params_nmpc.set(k, f"lam_{j}_{self._idx}_2", initial_duals[k][6])
                        self._params_nmpc.set(k, f"lam_{j}_{self._idx}_3", initial_duals[k][7])

                        if self._idx < j:
                            self._params_nmpc.set(k, f"s_{self._idx}_{j}_0", initial_duals[k][8])
                            self._params_nmpc.set(k, f"s_{self._idx}_{j}_1", initial_duals[k][9])
                        else:
                            self._params_nmpc.set(k, f"s_{j}_{self._idx}_0", initial_duals[k][8])
                            self._params_nmpc.set(k, f"s_{j}_{self._idx}_1", initial_duals[k][9])
                    else:
                        if k == self._N -1:
                            # Hardcoded for 2 robots TODO: Generalize
                            self._params_nmpc.set(k, f"lam_{self._idx}_{j}_0", self._ca_solution[1, k])
                            self._params_nmpc.set(k, f"lam_{self._idx}_{j}_1", self._ca_solution[2, k])
                            self._params_nmpc.set(k, f"lam_{self._idx}_{j}_2", self._ca_solution[3, k])
                            self._params_nmpc.set(k, f"lam_{self._idx}_{j}_3", self._ca_solution[4, k])

                            self._params_nmpc.set(k, f"lam_{j}_{self._idx}_0", self._ca_solution[5, k])
                            self._params_nmpc.set(k, f"lam_{j}_{self._idx}_1", self._ca_solution[6, k])
                            self._params_nmpc.set(k, f"lam_{j}_{self._idx}_2", self._ca_solution[7, k])
                            self._params_nmpc.set(k, f"lam_{j}_{self._idx}_3", self._ca_solution[8, k])

                            if self._idx < j:
                                self._params_nmpc.set(k, f"s_{self._idx}_{j}_0", self._ca_solution[9, k])
                                self._params_nmpc.set(k, f"s_{self._idx}_{j}_1", self._ca_solution[10, k])
                            else:
                                self._params_nmpc.set(k, f"s_{j}_{self._idx}_0", self._ca_solution[9, k])
                                self._params_nmpc.set(k, f"s_{j}_{self._idx}_1", self._ca_solution[10, k])
                        else:
                            self._params_nmpc.set(k, f"lam_{self._idx}_{j}_0", self._ca_solution[1, k+1])
                            self._params_nmpc.set(k, f"lam_{self._idx}_{j}_1", self._ca_solution[2, k+1])
                            self._params_nmpc.set(k, f"lam_{self._idx}_{j}_2", self._ca_solution[3, k+1])
                            self._params_nmpc.set(k, f"lam_{self._idx}_{j}_3", self._ca_solution[4, k+1])

                            self._params_nmpc.set(k, f"lam_{j}_{self._idx}_0", self._ca_solution[5, k+1])
                            self._params_nmpc.set(k, f"lam_{j}_{self._idx}_1", self._ca_solution[6, k+1])
                            self._params_nmpc.set(k, f"lam_{j}_{self._idx}_2", self._ca_solution[7, k+1])
                            self._params_nmpc.set(k, f"lam_{j}_{self._idx}_3", self._ca_solution[8, k+1])

                            if self._idx < j:
                                self._params_nmpc.set(k, f"s_{self._idx}_{j}_0", self._ca_solution[9, k+1])
                                self._params_nmpc.set(k, f"s_{self._idx}_{j}_1", self._ca_solution[10, k+1])
                            else:
                                self._params_nmpc.set(k, f"s_{j}_{self._idx}_0", self._ca_solution[9, k+1])
                                self._params_nmpc.set(k, f"s_{j}_{self._idx}_1", self._ca_solution[10, k+1])

                        
                
    def set_ca_parameters(self):
        # Set parameters for all k
        for k in range(self._N):
            for weight, value in self._weights.items():
                self._params_ca.set(k, weight, value)

            for j in range(1, self._number_of_robots+1):
                if j != self._idx:
                    trajectory_j = getattr(self, f'_trajectory_{j}')

                    if np.all(trajectory_j == 0):
                        xinit_j = np.array([self._settings[f"robot_{j}"]["start_x"], self._settings[f"robot_{j}"]["start_y"], self._settings[f"robot_{j}"]["start_theta"] * np.pi])
                        x_plan_j = set_initial_x_plan(self._settings, xinit_j)
                        # Shift with + 1 so similiar to received trajectories
                        if k == self._N - 1:
                            self._params_ca.set(k, f"x_{j}", x_plan_j[0, k])
                            self._params_ca.set(k, f"y_{j}", x_plan_j[1, k])
                            self._params_ca.set(k, f"theta_{j}", x_plan_j[2, k])
                        else:
                            self._params_ca.set(k, f"x_{j}", x_plan_j[0, k+1])
                            self._params_ca.set(k, f"y_{j}", x_plan_j[1, k+1])
                            self._params_ca.set(k, f"theta_{j}", x_plan_j[2, k+1])
                        
                    else:
                        self._params_ca.set(k, f"x_{j}", trajectory_j[0, k])
                        self._params_ca.set(k, f"y_{j}", trajectory_j[1, k])
                        self._params_ca.set(k, f"theta_{j}", trajectory_j[2, k])
                        self._params_nmpc.set(k, f"x_{j}", trajectory_j[0, k])
                        self._params_nmpc.set(k, f"y_{j}", trajectory_j[1, k])
                        self._params_nmpc.set(k, f"theta_{j}", trajectory_j[2, k])

            # Set ego trajectory with one timestep shifted
            trajectory_i = getattr(self, f'_trajectory_{self._idx}')
            if k == self._N-1:
                self._params_ca.set(k, f"x_{self._idx}", trajectory_i[0, k-1] + (trajectory_i[0, k-1] - trajectory_i[0, k - 2]))
                self._params_ca.set(k, f"y_{self._idx}", trajectory_i[1, k-1] + (trajectory_i[1, k-1] - trajectory_i[1, k - 2]))
                self._params_ca.set(k, f"theta_{self._idx}", trajectory_i[2, k-1] + (trajectory_i[2, k-1] - trajectory_i[2, k - 2]))
            else:
                self._params_ca.set(k, f"x_{self._idx}", trajectory_i[0, k+1])
                self._params_ca.set(k, f"y_{self._idx}", trajectory_i[1, k+1])
                self._params_ca.set(k, f"theta_{self._idx}", trajectory_i[2, k+1])

    def initialise_duals(self):

        for j in range(1, self._number_of_robots+1):
            if self._idx != j:
                setattr(self, f'initial_duals_{j}', dual_initialiser(self._settings, self._idx, 2))
    
    # Not used atm
    def set_uinit(self): 
        # Set u init for the CA solver
        self._map = load_model(model_map_name=f"model_map_ca_{self._idx}")
        
        for j in range(1, self._number_of_robots+1):
                if j != self._idx:
                    lam_name = f"lam_{self._idx}_{j}"
                    lam = self._save_lam[0][lam_name]
                    map_value = self._map[lam_name + "_0"][1]
                    self._uinit[map_value - self._nx_ca : (map_value - self._nx_ca) + 4] = lam
                    lam_name = f"lam_{j}_{self._idx}"
                    lam = self._save_lam[0][lam_name]
                    map_value = self._map[lam_name + "_0"][1]
                    self._uinit[map_value - self._nx_ca : (map_value - self._nx_ca) + 4] = lam

                    if self._idx < j:
                        s_name = f"s_{self._idx}_{j}"
                    else:
                        s_name = f"s_{j}_{self._idx}"
                    s = self._save_s[0][s_name]
                    map_value = self._map[s_name + "_0"][1]
                    self._uinit[map_value - self._nx_ca : (map_value - self._nx_ca) + 2] = s
                    
                        
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
    
    def visualize(self): 
        if self._state_msg is not None:
            robot_pos = self._visuals.get_sphere()
            robot_pos.set_color(0)
            robot_pos.set_scale(0.3, 0.3, 0.3)

            pose = Pose()
            pose.position.x = float(self._state[0])
            pose.position.y = float(self._state[1])
            robot_pos.add_marker(pose)
        if self._save_s:
            line = self._visuals.get_line()
            line.set_scale(0.05)
            line.set_color(7**self._idx, alpha=1.0)
            ego_pos = np.array([self._state[0], self._state[1]])

            for j in range(1, self._number_of_robots+1):
                if j != self._idx:
                    s = self._save_s[-1][f's_{self._idx}_{j}'] if self._idx < j else self._save_s[-1][f's_{j}_{self._idx}']
                    neighbour_pos = np.array([self._params_ca.get(1, f"x_{j}"), self._params_ca.get(1, f"y_{j}")])
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

        trajectory_i = getattr(self, f'_trajectory_{self._idx}')
        if not np.all(trajectory_i == 0) and self._mpc_feasible:
            length = self._settings["polytopic"]["length"]
            width = self._settings["polytopic"]["width"]

            box = self._visuals.get_cube()
            box.set_color(80, alpha=0.3)
            box.set_scale(width, length, 0.05)
            
            pose = Pose()
            for k in range(1, self._N):
                pose.position.x = self._planner.get_model_nmpc().get(k, f"x_{self._idx}")
                pose.position.y = self._planner.get_model_nmpc().get(k, f"y_{self._idx}")
                theta = self._planner.get_model_nmpc().get(k, f"theta_{self._idx}")
                quaternion = yaw_to_quaternion(theta)
                pose.orientation.x = quaternion[0]
                pose.orientation.y = quaternion[1]
                pose.orientation.z = quaternion[2]
                pose.orientation.w = quaternion[3]
                box.add_marker(deepcopy(pose))
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

            self._states_save.append(deepcopy(self._state))
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
        plot_path(self)

    def trajectory_callback(self, traj_msg, i):
        # idx = traj_msg._connection_header['topic'][-1]
        j = i

        if not traj_msg.poses:
            rospy.logwarn(f"Received empty path robot {j}")
            return
        
        if i != self._idx:
            trajectory = getattr(self, f"_trajectory_{j}")
            for k, pose in enumerate(traj_msg.poses):
                trajectory[0, k] = pose.pose.position.x
                trajectory[1, k] = pose.pose.position.y
                trajectory[2, k] = quaternion_to_yaw(pose.pose.orientation)
    
    def print_stats(self):
        self._planner.print_stats()
        # self._decomp_constraints.print_stats()
    
    def plot_states(self):
        plot_states(self)
    
    
    def plot_duals(self):
        plot_duals(self)



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
