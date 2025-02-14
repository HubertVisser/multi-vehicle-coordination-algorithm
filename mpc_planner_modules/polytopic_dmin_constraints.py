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


class PolytopicDminConstraintModule(ConstraintModule):

    def __init__(self, settings, robot_idx):
        super().__init__()

        self.module_name = f"PolytopicConstraints_dmin_{robot_idx}"  # c++ name of the module
        self.import_name = "polytopic_constraints.h"

        self.constraints.append(PolytopicDminConstraints(n_robots=settings["number_of_robots"], d_min=settings["polytopic"]["d_min"], length=settings["polytopic"]["length"], width=settings["polytopic"]["width"], robot_idx=robot_idx, use_slack=False))
        self.description = "Polytopic set based collision avoidance constraints in dual formulation (constraint 5a)"


# Constraints of the form -b_i * lam_ij^T - b_j * lam_ji^T >= d_min
class PolytopicDminConstraints:

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
            lower_bound.append(0)
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
        pos = cd.vertcat(pos_x, pos_y)   # Center of gravity
        theta = model.get(f"theta_{self.robot_idx}")


        # try:
        #     if self.use_slack:
        #         slack = model.get("slack")
        #     else:
        #         slack = 0.0
        # except:
        #     slack = 0.0

        rot_mat_i = rotation_matrix(theta)
        A_i = cd.vertcat(rot_mat_i.T, -rot_mat_i.T)
        assert A_i.shape == (4, 2)

        dim_vector = cd.DM([self.length/2, self.width/2, self.length/2, self.width/2])
        b_i = dim_vector + A_i @ pos
        assert b_i.shape == (4,1)

        # Constraints for all neighbouring robots (j)
        for j in range(1, self.n_robots+1): 
            if j == self.robot_idx:
                continue   
            
            pos_j_x = model.get(f"x_{j}")
            pos_j_y = model.get(f"y_{j}")
            pos_j = cd.vertcat(pos_j_x, pos_j_y)

            theta_j = model.get(f"theta_{j}")
            lamda_ij = model.get(f"lam_{self.robot_idx}_{j}")
            lamda_ji = model.get(f"lam_{j}_{self.robot_idx}")

            rot_mat_j = rotation_matrix(theta_j)
            A_j = cd.vertcat([rot_mat_j.T, -rot_mat_j.T])
            assert A_j.shape == (4, 2)

            b_j = dim_vector + A_j @ pos_j
            assert b_j.shape == (4,1)

            lam_vec_ij = cd.DM.ones(b_i.shape[0],1) * lamda_ij
            lam_vec_ji = cd.DM.ones(b_j.shape[0],1) * lamda_ji

            constraints.append(- b_i.T @ lam_vec_ij - b_j.T @ lam_vec_ji)

        return constraints
