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


class PolytopicSjdualConstraintModule(ConstraintModule):

    def __init__(self, settings, robot_idx):
        super().__init__()

        self.module_name = f"PolytopicConstraints_sj_{robot_idx}"  # c++ name of the module
        self.import_name = "polytopic_constraints.h"

        self.constraints.append(PolytopicSjdualConstraints(n_robots=settings["number_of_robots"], length=settings["polytopic"]["length"], width=settings["polytopic"]["width"], robot_idx=robot_idx, use_slack=False))
        self.description = "Polytopic set based collision avoidance constraints in dual formulation (constraint 5c)"


# Constraints of the form A_j^T @ lam_ji - S_ji = 0
class PolytopicSjdualConstraints:

    def __init__(self, n_robots, length, width, robot_idx, use_slack=False):
        self.n_robots = n_robots
        self.length = length
        self.width = width
        self.robot_idx = robot_idx
        self.n_constraints = (n_robots - self.robot_idx) * 2
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
            upper_bound.append(0)
        return upper_bound

    def get_constraints(self, model, params, settings, stage_idx):
        constraints = []

        # Constraints from all neighbouring robots (j) to the ego robot (i)
        for j in range(self.robot_idx, self.n_robots+1): 
            if j == self.robot_idx:

                theta_j = model.get(f"theta_{j}")            
                
                rot_mat_j = rotation_matrix(theta_j)
                A_j = cd.vertcat(rot_mat_j.T, -rot_mat_j.T)
                assert A_j.shape == (4, 2)

                lam_ji = cd.vertcat(model.get(f"lam_{j}_{self.robot_idx}_0"), 
                                    model.get(f"lam_{j}_{self.robot_idx}_1"), 
                                    model.get(f"lam_{j}_{self.robot_idx}_2"), 
                                    model.get(f"lam_{j}_{self.robot_idx}_3"))
                s_ji = model.get(f"s_{self.robot_idx}_{j}")

                constraint = A_j.T @ lam_ji - s_ji
                constraints.append(constraint[0])
                constraints.append(constraint[1])

        return constraints
