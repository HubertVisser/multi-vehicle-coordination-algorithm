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


class PolytopicSidualConstraintModule(ConstraintModule):

    def __init__(self, settings, robot_idx):
        super().__init__()

        self.module_name = f"PolytopicConstraints_si_{robot_idx}"  # c++ name of the module
        self.import_name = "polytopic_constraints.h"

        self.constraints.append(PolytopicSidualConstraints(n_robots=settings["number_of_robots"], length=settings["polytopic"]["length"], width=settings["polytopic"]["width"], robot_idx=robot_idx, use_slack=False))
        self.description = "Polytopic set based collision avoidance constraints in dual formulation (constraint 5b)"


# Constraints of the form A_i^T @ lam_ij + S_ij = 0
class PolytopicSidualConstraints:

    def __init__(self, n_robots, length, width, robot_idx, use_slack=False):
        self.n_robots = n_robots
        self.length = length
        self.width = width
        self.robot_idx = robot_idx
        self.n_constraints = (n_robots - 1)* 2
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

        # States
        theta = model.get(f"theta_{self.robot_idx}")

        rot_mat_i = rotation_matrix(theta)
        A_i = cd.vertcat(rot_mat_i.T, -rot_mat_i.T)
        assert A_i.shape == (4, 2)

        # Constraints for all neighbouring robots (j)
        for j in range(1, self.n_robots+1): 
            if j == self.robot_idx:
                continue   
            
            lam_ij = cd.vertcat(model.get(f"lam_{self.robot_idx}_{j}_0"), 
                                model.get(f"lam_{self.robot_idx}_{j}_1"), 
                                model.get(f"lam_{self.robot_idx}_{j}_2"), 
                                model.get(f"lam_{self.robot_idx}_{j}_3"))
            s_ij = model.get(f"s_{self.robot_idx}_{j}")

            constraint = A_i.T @ lam_ij + s_ij
            constraints.append(constraint[0])
            constraints.append(constraint[1])

        return constraints
