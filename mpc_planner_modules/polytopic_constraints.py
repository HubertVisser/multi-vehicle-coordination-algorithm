"""
Polytopic Constraints (Minimum distance constraints between robots represented as polytopic sets using dual variables)
"""

import numpy as np
import casadi as cd

import sys
import os

sys.path.append(os.path.join(sys.path[0], "..", "solver_generator"))

from util.math import rotation_matrix
from control_modules import ConstraintModule


class LinearizedConstraintModule(ConstraintModule):

    def __init__(self, settings, robot_idx):
        super().__init__()

        self.module_name = "PolytopicConstraints"  # c++ name of the module
        self.import_name = "polytopic_constraints.h"

        self.constraints.append(PolytopicConstraints(n_robots=settings["number_of_robots"], d_min=["polytopic"]["d_min"], length=settings["polytopic"]["length"], length=settings["polytopic"]["width"], robot_idx=robot_idx, use_slack=False))
        self.description = "Polytopic set based collision avoidance constraints in dual formulation"


# Constraints of the form -b_i * lam_ij^T - b_j * lam_ji^T >= d_min
class PolytopicConstraints:

    def __init__(self, n_robots, d_min, length, width, robot_idx, use_slack=False):
        self.n_robots = n_robots
        self.d_min = d_min
        self.length = length
        self.width = width
        self.robot_idx = robot_idx
        self.n_constraints = n_robots - 1
        self.nh = self.n_constraints
        self.use_slack = use_slack

    def define_parameters(self, params):
        pass

    def get_lower_bound(self):
        lower_bound = []
        for index in range(0, self.n_constraints):
            lower_bound.append(self.d_min)
        return lower_bound

    def get_upper_bound(self):
        upper_bound = []
        for index in range(0, self.n_constraints):
            upper_bound.append(1000)
        return upper_bound

    def get_constraints(self, model, params, settings, stage_idx):
        constraints = []

        # States
        pos_x = model.get(f"x_{self.robot_idx}")
        pos_y = model.get(f"y_{self.robot_idx}")
        pos = np.array([pos_x, pos_y])
        psi = model.get(f"theta_{self.robot_idx}")

        # try:
        #     if self.use_slack:
        #         slack = model.get("slack")
        #     else:
        #         slack = 0.0
        # except:
        #     slack = 0.0

        rotation_car = rotation_matrix(psi)
        for j in range(1, self.n_robots+1):
            pass  # stranded here    
            # disc_x = params.get(f"ego_disc_{disc_it}_offset")
            # disc_relative_pos = np.array([disc_x, 0])
            # disc_pos = pos + rotation_car.dot(disc_relative_pos)

            # for index in range(self.max_obstacles):
            #     a1 = params.get(self.constraint_name(index, disc_it) + "_a1")
            #     a2 = params.get(self.constraint_name(index, disc_it) + "_a2")
            #     b = params.get(self.constraint_name(index, disc_it) + "_b")

            #     constraints.append(a1 * disc_pos[0] + a2 * disc_pos[1] - (b + slack))

        return constraints
