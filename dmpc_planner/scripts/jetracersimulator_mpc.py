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

from util.files import load_settings
from util.realtime_parameters import RealTimeParameters
from util.convertion import quaternion_to_yaw
from util.logging import print_value 
from timer import Timer

from contouring_spline import SplineFitter
# from static_constraints import StaticConstraints
from mpc_controller import MPCPlanner
from ros_visuals import ROSMarkerPublisher
from project_trajectory import project_trajectory_to_safety
from pyplot import plot_x_traj
from path_generator import generate_path_msg


class ROSMPCPlanner:
    def __init__(self):
        self._settings = load_settings(package="dmpc_planner")
        self._N = self._settings["N"]
        self._integrator_step = self._settings["integrator_step"]
        self._max_obstacles = self._settings["max_obstacles"]
        self._braking_acceleration = self._settings["braking_acceleration"]
        self._obstacle_radius = self._settings["obstacle_radius"]
        self._robot_radius = self._settings["robot_radius"]

        self._verbose = self._settings["verbose"]
        self._debug_visuals = self._settings["debug_visuals"]

        self._planner = MPCPlanner(self._settings)
        # self._planner.set_projection(lambda trajectory: self.project_to_safety(trajectory))

        self._spline_fitter = SplineFitter(self._settings)
        # self._decomp_constraints = StaticConstraints(self._settings)

        self._solver_settings = load_settings(
            "solver_settings", package="mpc_planner_solver"
        )

        # Tied to the solver
        self._params = RealTimeParameters(
            self._settings, package="mpc_planner_solver"
        )  # This maps to parameters used in the solver by name
        self._weights = self._settings["weights"]
        n_states = self._solver_settings["nx"]
        self._state = np.zeros((n_states,))
        self._state[1] = -2.0

        self._visuals = ROSMarkerPublisher("mpc_visuals", 100)
        self._path_visual = ROSMarkerPublisher("reference_path", 10)
        self._debug_visuals = ROSMarkerPublisher("mpc_planner_py/debug", 10)

        self._state_msg = None
        self._goal_msg = None
        self._path_msg = None
        self._trajectory = None
        self._obstacle_msg = None
        # self._obst_lock = threading.Lock()

        self._goal_x = 0.0
        self._goal_y = 0.0

        self._enable_output = False
        self._mpc_feasible = True

        self._timer = rospy.Timer(
            rospy.Duration(1.0 / self._settings["control_frequency"]), self.run
        )

        self._callbacks_enabled = False
        self.initialize_publishers_and_subscribers()
        msg = generate_path_msg(self._settings)
        self.path_callback(msg)
        self._callbacks_enabled = True

        # self.start_environment()

        

    def initialize_publishers_and_subscribers(self):

        # Sub
        self._state_sub = rospy.Subscriber(
            "vicon/jetracer1", PoseStamped, self.state_pose_callback, queue_size=1
        )
        # self._obs_sub = rospy.Subscriber(
        #     "/pedestrian_simulator/trajectory_predictions",
        #     ObstacleArray,
        #     self.obstacle_callback,
        #     queue_size=1,
        # )
        # self._goal_sub = rospy.Subscriber(
        #     "/move_base_simple/goal",
        #     PoseStamped,
        #     lambda msg: self.goal_callback(msg),
        #     queue_size=1,
        # )
        self._path_sub = rospy.Subscriber(
            "roadmap/reference", Path, lambda msg: self.path_callback(msg), queue_size=1
        )

        # self._weight_sub = rospy.Subscriber(
        #     "hey_robot/weights",
        #     WeightArray,
        #     lambda msg: self.weight_callback(msg),
        #     queue_size=1,
        # )
        # self._reload_solver_sub = rospy.Subscriber(
        #     "/mpc/reload_solver",
        #     Empty,
        #     lambda msg: self.reload_solver_callback(msg),
        #     queue_size=1,
        # )

        # self._reset_sub = rospy.Subscriber(
        #     "/jackal_socialsim/reset", Empty, lambda msg: self.reset(), queue_size=1
        # )

        # Pub
        self._th_pub = rospy.Publisher("throttle_1", Float32, queue_size=1) # Throttle publisher
        self._st_pub = rospy.Publisher("steering_1", Float32, queue_size=1) # Steering publisher

        self._ped_robot_state_pub = rospy.Publisher(
            "/pedestrian_simulator/robot_state", PoseStamped, queue_size=1
        )

        # # Services
        # self._ped_horizon_pub = rospy.Publisher(
        #     "/pedestrian_simulator/horizon", Int32, queue_size=1
        # )
        # self._ped_integrator_step_pub = rospy.Publisher(
        #     "/pedestrian_simulator/integrator_step", Float32, queue_size=1
        # )
        # self._ped_clock_frequency_pub = rospy.Publisher(
        #     "/pedestrian_simulator/clock_frequency", Float32, queue_size=1
        # )

        # self._reset_simulation_pub = rospy.Publisher(
        #     "/lmpcc/reset_environment", Empty, queue_size=5
        # )
        # self._reset_simulation_client = rospy.ServiceProxy(
        #     "/gazebo/reset_world", std_srvs.srv.Empty
        # )
        # self._reset_ekf_client = rospy.ServiceProxy("/set_pose", SetPose)

    def start_environment(self):
        rospy.loginfo("Starting pedestrian simulator")
        rospy.wait_for_service("/pedestrian_simulator/start")
        self._ped_start_client = rospy.ServiceProxy(
            "/pedestrian_simulator/start", std_srvs.srv.Empty
        )

        for i in range(20):
            horizon_msg = Int32()
            horizon_msg.data = self._settings["N"]
            self._ped_horizon_pub.publish(horizon_msg)

            integrator_step_msg = Float32()
            integrator_step_msg.data = self._settings["integrator_step"]
            self._ped_integrator_step_pub.publish(integrator_step_msg)

            clock_frequency_msg = Float32()
            clock_frequency_msg.data = self._settings["control_frequency"]
            self._ped_clock_frequency_pub.publish(clock_frequency_msg)

            empty_srv = std_srvs.srv.Empty._request_class()
            try:
                resp = self._ped_start_client(empty_srv)
                break
            except rospy.ServiceException:
                rospy.loginfo_throttle(3, "Waiting for pedestrian simulator to start")
                rospy.sleep(1.0)

        self._enable_output = True
        rospy.sleep(0.5)
        self.reset()
        rospy.loginfo("Environment ready.")

    def reset(self):
        msg = Empty()
        srv = std_srvs.srv.Empty._request_class()
        pose_srv = SetPose._request_class()
        self._reset_simulation_pub.publish(msg)
        self._reset_simulation_client(srv)

    def run(self, timer):
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

        # if self._mpc_feasible:
        #     plot_x_traj(self._trajectory, self._N, self._integrator_step)

        self.publish_throttle(output, self._mpc_feasible)
        self.publish_steering(output, self._mpc_feasible)
        self.publish_robot_state()
        self.visualize()

        # self._state[0] = output["x"]
        # self._state[1] = output["y"]

        # # Extract yaw angle (rotation around the Z-axis)
        # self._state[2] = output["theta"]

        # # Velocity is in the local frame, x is the forward velocity
        # self._state[3] = output["vx"]
        # self._state[4] = output["vy"]
        # self._state[5] = output["w"]

    def set_parameters(self):
        splines = None
        if self._path_msg is not None and self._spline_fitter.ready():
            splines = self._spline_fitter.get_active_splines(
                np.array([self._state[0], self._state[1]])
            )
            self._state[-1] = self._spline_fitter.find_closest_s(
                np.array([self._state[0], self._state[1]])
            )

        # self._decomp_constraints.call(
        #     self._state, self._planner.get_model(), self._params, self._mpc_feasible
        # )

        # Set parameters for all k
        for k in range(self._N + 1):

            # Tuning parameters
            for weight, value in self._weights.items():
                self._params.set(k, weight, value)

            self._params.set(k, "goal_x", self._goal_x)
            self._params.set(k, "goal_y", self._goal_y)

            if splines is not None:
                for i in range(self._settings["contouring"]["num_segments"]):
                    self._params.set(k, f"spline_x{i}_a", splines[i]["a_x"])
                    self._params.set(k, f"spline_x{i}_b", splines[i]["b_x"])
                    self._params.set(k, f"spline_x{i}_c", splines[i]["c_x"])
                    self._params.set(k, f"spline_x{i}_d", splines[i]["d_x"])

                    self._params.set(k, f"spline_y{i}_a", splines[i]["a_y"])
                    self._params.set(k, f"spline_y{i}_b", splines[i]["b_y"])
                    self._params.set(k, f"spline_y{i}_c", splines[i]["c_y"])
                    self._params.set(k, f"spline_y{i}_d", splines[i]["d_y"])

                    self._params.set(k, f"spline{i}_start", splines[i]["s"])
                    # print(f"{splines[i]['a_x']:.1f}, {splines[i]['b_x']:.1f}, {splines[i]['c_x']:.1f}, {splines[i]['d_x']:.1f}, {splines[i]['a_y']:.1f}, {splines[i]['b_y']:.1f}, {splines[i]['c_y']:.1f}, {splines[i]['d_y']:.1f}, {splines[i]['s']:.1f}")

        # Lock obstacle data
        # with self._obst_lock:
        #     for k in range(self._N + 1):
        #         if self._obstacle_msg is None:
        #             obstacles = []
        #         else:
        #             obstacles = self._obstacle_msg.obstacles

        #         num_obs = len(obstacles)
        #         self._params.set(k, "ego_disc_radius", self._robot_radius)
        #         self._params.set(k, "ego_disc_0_offset", 0.0)
        #         for j in range(min(self._max_obstacles, num_obs)):
        #             obs = obstacles[j]

        #             if k == 0:  # or k == self._N:
        #                 self._params.set(
        #                     k, f"ellipsoid_obst_{j}_x", self._state[0] + 100.0
        #                 )
        #                 self._params.set(
        #                     k, f"ellipsoid_obst_{j}_y", self._state[1] + 100.0
        #                 )
        #                 self._params.set(k, f"ellipsoid_obst_{j}_chi", 1.0)
        #                 self._params.set(k, f"ellipsoid_obst_{j}_psi", 0.0)
        #                 self._params.set(k, f"ellipsoid_obst_{j}_r", 0.1)
        #                 self._params.set(k, f"ellipsoid_obst_{j}_major", 0.0)
        #                 self._params.set(k, f"ellipsoid_obst_{j}_minor", 0.0)
        #                 continue

        #             # Constant velocity prediction
        #             predicted_pose = obs.gaussians[0].mean.poses[k - 1].pose
        #             obs_predicted_pos = np.array(
        #                 [predicted_pose.position.x, predicted_pose.position.y]
        #             )

        #             yaw = quaternion_to_yaw(predicted_pose.orientation)

        #             self._params.set(k, f"ellipsoid_obst_{j}_x", obs_predicted_pos[0])
        #             self._params.set(k, f"ellipsoid_obst_{j}_y", obs_predicted_pos[1])
        #             self._params.set(k, f"ellipsoid_obst_{j}_chi", 1.0)
        #             self._params.set(k, f"ellipsoid_obst_{j}_psi", yaw)
        #             self._params.set(k, f"ellipsoid_obst_{j}_major", 0.0)
        #             self._params.set(k, f"ellipsoid_obst_{j}_minor", 0.0)
        #             self._params.set(k, f"ellipsoid_obst_{j}_r", self._obstacle_radius)

        #         # Dummies
        #         for j in range(num_obs, self._max_obstacles):
        #             self._params.set(k, f"ellipsoid_obst_{j}_x", self._state[0] + 100.0)
        #             self._params.set(k, f"ellipsoid_obst_{j}_y", self._state[1] + 100.0)
        #             self._params.set(k, f"ellipsoid_obst_{j}_chi", 1.0)
        #             self._params.set(k, f"ellipsoid_obst_{j}_psi", 0.0)
        #             self._params.set(k, f"ellipsoid_obst_{j}_r", 0.1)
        #             self._params.set(k, f"ellipsoid_obst_{j}_major", 0.0)
        #             self._params.set(k, f"ellipsoid_obst_{j}_minor", 0.0)

    def project_to_safety(self, trajectory):
        # Projects a trajectory to safety from the obstacles using Douglas Rachford projection
        # Trajectory is assumed to be nx x N
        N = trajectory.shape[1]

        if self._obstacle_msg is None:
            return trajectory

        for k in range(N):
            start_pose = deepcopy(trajectory[:2, k])
            for idx, obs in enumerate(self._obstacle_msg.obstacles):
                predicted_pose = obs.gaussians[0].mean.poses[k - 1].pose
                obs_predicted_pos = np.array(
                    [predicted_pose.position.x, predicted_pose.position.y]
                )

                if idx == 0:
                    anchor = obs_predicted_pos
                    continue  # We do not need to project for this obstacle

                project_trajectory_to_safety(
                    trajectory[:2, k],
                    obs_predicted_pos,
                    anchor,
                    self._obstacle_radius + self._robot_radius + 0.1,
                    start_pose,
                )

        return trajectory

    def publish_throttle(self, output, exit_flag):
        throttle = Float32()
        # print(f"v = {output['v']}, w = {output['w']}")
        if not self._mpc_feasible or not self._enable_output:
            if not self._mpc_feasible:
                rospy.logwarn_throttle(1, "Infeasible MPC. Braking!")
                throttle.data = max(
                    0.0,
                    self._state[3] - self._braking_acceleration * self._integrator_step,
                )
            else:
                rospy.logwarn_throttle(1, "Output is disabled. Sending zero velocity!")
                throttle.data = 0.0
        else:
            throttle.data = output["throttle"]
            rospy.loginfo_throttle(1000, "MPC is driving")
            self._th_pub.publish(throttle)
    
    def publish_steering(self, output, exit_flag):
        steering = Float32()
        # print(f"v = {output['v']}, w = {output['w']}")
        if not self._mpc_feasible or not self._enable_output:
            steering.data = 0.0
        else:
            steering.data = output["steering"]
            self._st_pub.publish(steering)

    def publish_robot_state(self):
        pose = PoseStamped()
        pose.header.stamp = rospy.Time.now()
        pose.header.frame_id = "map"

        pose.pose.position.x = self._state[0]
        pose.pose.position.y = self._state[1]
        pose.pose.position.z = self._state[3]
        self._ped_robot_state_pub.publish(pose)

    def reload_solver_callback(self, msg):
        self._callbacks_enabled = False
        self._timer.shutdown()
        self._planner.init_acados_solver()

        self._solver_settings = load_settings(
            "solver_settings", package="mpc_planner_solver"
        )
        self._params = RealTimeParameters(
            self._settings, package="mpc_planner_solver"
        )  # This maps to parameters used in the solver by name
        n_states = self._solver_settings["nx"]
        self._state = np.zeros((n_states,))
        self._trajectory = None

        self._timer = rospy.Timer(
            rospy.Duration(1.0 / self._settings["control_frequency"]), self.run
        )
        self._callbacks_enabled = True

    def weight_callback(self, msg):
        new_weights = dict()
        for i in range(len(msg.weights)):
            new_weights[msg.weights[i].name] = msg.weights[i].value
            # print(f"{msg.weights[i].name}: {msg.weights[i].value}")
        self._weights = new_weights

    def state_pose_callback(self, msg):
        self._state_msg = msg
        self._state[0] = msg.pose.position.x
        self._state[1] = msg.pose.position.y

        # Extract yaw angle (rotation around the Z-axis)
        self._state[2] = msg.pose.orientation.z

        # Velocity is in the local frame, x is the forward velocity
        self._state[3] = msg.pose.position.z

        # print("-------- State ----------")
        # print(f"x = {self._state[0]:.2f}")
        # print(f"y = {self._state[1]:.2f}")
        # print(f"theta = {self._state[2]:.2f}")
        # print(f"vx = {self._state[3]:.2f}")

    def obstacle_callback(self, msg):
        if self._state is None or not self._callbacks_enabled:
            return

        with self._obst_lock:
            obs_timer = Timer("obstacle callback")

            self._obstacle_msg = msg
            sorted_obstacles = sorted(
                self._obstacle_msg.obstacles,
                key=lambda obs: np.linalg.norm(
                    np.array([obs.pose.position.x, obs.pose.position.y])
                    - np.array([self._state[0], self._state[1]])
                ),
            )

            self._obstacle_msg.obstacles = sorted_obstacles

    def goal_callback(self, msg):
        if (
            np.linalg.norm(
                np.array([msg.pose.position.x, msg.pose.position.y])
                - np.array([self._goal_x, self._goal_y])
            )
            > 0.01
        ):
            print(f"New Goal Received ({msg.pose.position.x}, {msg.pose.position.y})")
        self._goal_msg = msg.pose
        self._goal_x = self._goal_msg.position.x
        self._goal_y = self._goal_msg.position.y

    def path_callback(self, msg):

        # Filter equal paths
        if self._path_msg is not None and len(self._path_msg.poses) == len(msg.poses):
            return

        self._path_msg = msg
        self._spline_fitter.fit_path(msg)
        self.plot_path()

    def visualize(self):
        if self._state_msg is not None:
            robot_pos = self._visuals.get_sphere()
            robot_pos.set_color(0)
            robot_pos.set_scale(0.3, 0.3, 0.3)

            pose = Pose()
            pose.position.x = self._state[0]
            pose.position.y = self._state[1]
            robot_pos.add_marker(pose)

        if self._goal_msg is not None:
            robot_pos = self._visuals.get_cube()
            robot_pos.set_color(3)
            robot_pos.set_scale(0.3, 0.3, 0.3)

            pose = Pose()
            pose.position.x = self._goal_msg.position.x
            pose.position.y = self._goal_msg.position.y
            robot_pos.add_marker(pose)

        if self._trajectory is not None and self._mpc_feasible:
            cylinder = self._visuals.get_cylinder()
            cylinder.set_color(1, alpha=0.25)
            cylinder.set_scale(0.65, 0.65, 0.05)

            pose = Pose()
            for k in range(1, self._N):
                pose.position.x = self._planner.get_model().get(k, "x")
                pose.position.y = self._planner.get_model().get(k, "y")
                cylinder.add_marker(deepcopy(pose))

        plot_obstacles = False  # Slows down the MPC!
        if self._obstacle_msg is not None and plot_obstacles:
            line = self._visuals.get_line()
            line.set_scale(0.1)

            for i in range(min(self._max_obstacles, len(self._obstacle_msg.obstacles))):
                line.set_color(i, alpha=1.0)
                try:
                    obs = self._obstacle_msg.obstacles[i]
                except:
                    break
                for k in range(0, self._N - 1, 5):
                    line.add_line_from_poses(
                        obs.gaussians[0].mean.poses[k].pose,
                        obs.gaussians[0].mean.poses[k + 1].pose,
                    )
        self._visuals.publish()

        if self._debug_visuals:
            if self._spline_fitter._closest_s is not None:
                cube = self._debug_visuals.get_cube()
                cube.set_color(5)
                cube.set_scale(0.3, 0.3, 0.3)
                pose = Pose()
                pose.position.x = self._spline_fitter._closest_x
                pose.position.y = self._spline_fitter._closest_y
                cube.add_marker(pose)

            self.plot_warmstart()
            self._debug_visuals.publish()

    # For debugging purposes
    def plot_warmstart(self):
        cylinder = self._debug_visuals.get_cylinder()
        cylinder.set_color(10, alpha=1.0)
        cylinder.set_scale(0.65, 0.65, 0.05)

        warmstart_u, warmstart_x = self._planner.get_initial_guess()

        pose = Pose()
        for k in range(1, self._N):
            pose.position.x = warmstart_x[0, k]
            pose.position.y = warmstart_x[1, k]
            cylinder.add_marker(deepcopy(pose))

    def plot_path(self):
        dist = 0.5
        if self._path_msg is not None:
            line = self._path_visual.get_line()
            line.set_scale(0.1)
            line.set_color(1, alpha=1.0)

            points = self._path_visual.get_cube()
            points.set_color(3)
            points.set_scale(0.3, 0.3, 0.3)
            s = 0.0
            for i in range(50):
                a = self._spline_fitter.evaluate(s)
                b = self._spline_fitter.evaluate(s + dist)
                pose_a = Pose()
                pose_a.position.x = a[0]
                pose_a.position.y = a[1]
                points.add_marker(pose_a)
                pose_b = Pose()
                pose_b.position.x = b[0]
                pose_b.position.y = b[1]
                s += dist
                line.add_line_from_poses(pose_a, pose_b)
        self._path_visual.publish()
        self._enable_output = True

    def print_stats(self):
        self._planner.print_stats()
        # self._decomp_constraints.print_stats()


if __name__ == "__main__":
    rospy.loginfo("Initializing MPC")
    rospy.init_node("dmpc_planner", anonymous=False)

    mpc = ROSMPCPlanner()

    while not rospy.is_shutdown():
        rospy.spin()

    mpc.print_stats()
