#!/usr/bin/env python3
# filepath: /home/dock_user/ros_ws/src/multi-vehicle-coordination-algorithm/multi_vehicle_coordination_algorithm/scripts/distributed-algorithm/robot_node.py

import os, sys
import pathlib
path = pathlib.Path(__file__).parent.resolve()
sys.path.append(os.path.join(path))
sys.path.append(os.path.join(sys.path[-1], "..", "..", "..", "solver_generator"))
sys.path.append(os.path.join(sys.path[-2], "..", "..", "..", "mpc_planner_modules"))
sys.path.append(os.path.join(sys.path[-3], ".."))

import rospy
import debugpy
from ros_mpc_controller import ROSMPCPlanner
from util.files import load_settings
from timer import Timer


class RobotNode:
    def __init__(self, robot_id):
        self._settings = load_settings(package="multi_vehicle_coordination_algorithm")
        self._robot_id = robot_id
        self._iterations = self._settings['solver_settings']['iterations_distributed']
        self._robot = ROSMPCPlanner(self._robot_id, self._settings)

        self._timer = rospy.Timer(
            rospy.Duration(1.0 / self._settings["control_frequency"]), self.run
        )

    def run(self, timer):
        if not self._robot._spline_fitter._splines:
            rospy.logwarn(f"Robot {self._robot_id}: Splines not computed yet.")
            return
        nmpc_ca_timer = Timer(f"Robot {self._robot_id} - NMPC-CA")
        for it in range(1, self._iterations + 1):
            self._robot.run_nmpc(timer, it)
            self._robot.run_ca(timer, it)
        self._robot.visualize()
                
        del nmpc_ca_timer


if __name__ == "__main__":
    rospy.init_node("robot_node", anonymous=True)
    robot_id = rospy.get_param("~robot_id", 1)  # Get robot ID from paramete
            
    rospy.loginfo(f"Starting Robot Node {robot_id}")
    node = RobotNode(robot_id)
    rospy.spin()