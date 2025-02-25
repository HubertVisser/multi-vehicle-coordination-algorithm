"""
Bounds on the 2-norm of the s in the dual formulation of the polytopic constraints
"""

import numpy as np
import casadi as cd

import sys
import os

sys.path.append(os.path.join(sys.path[0], "..", "solver_generator"))

from util.math import rotation_matrix
from control_modules import ConstraintModule


class s2normConstraintModule(ConstraintModule):

    def __init__(self, settings):
        super().__init__()

        self.module_name = f"s_2norm"  # c++ name of the module
        self.import_name = "s_2norm.h"

        self.constraints.append(s2normConstraintConstraints(n_robots=settings["number_of_robots"]))
        self.description = "Bounds on the 2-norm of the s in the dual formulation of the polytopic constraints"


# Constraints of the form sqrt(sum(s^2)) =< 1
class s2normConstraintConstraints:

    def __init__(self, n_robots):
        self.n_robots = n_robots
        self.n_constraints = ((n_robots * n_robots)- n_robots)//2
        self.nh = self.n_constraints

    def define_parameters(self, params):
        pass

    def get_lower_bound(self):
        lower_bound = []
        for index in range(0, self.n_constraints):
            lower_bound.append(-100)
        return lower_bound

    def get_upper_bound(self):
        upper_bound = []
        for index in range(0, self.n_constraints):
            upper_bound.append(1)
        return upper_bound

    def get_constraints(self, model, params, settings, stage_idx):
        constraints = []

        for i in range(1, self.n_robots+1):
            for j in range(i, self.n_robots+1): 
                if j != i:
                    s_ij = model.get(f"s_{i}_{j}")
        constraint = cd.norm_2(s_ij)
        constraints.append(constraint)

        return constraints
