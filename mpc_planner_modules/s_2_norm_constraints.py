"""
Bounds on the 2-norm of the s in the dual formulation of the polytopic constraints
"""

import numpy as np
import casadi as cd

import sys
import os

sys.path.append(os.path.join(sys.path[0], "..", "solver_generator"))

from control_modules import ConstraintModule


class s2normConstraintModule(ConstraintModule):

    def __init__(self, settings, idx):
        super().__init__()

        self.module_name = f"s_2norm"  # c++ name of the module
        self.import_name = "s_2norm.h"

        self.constraints.append(s2normConstraintConstraints(n_robots=settings["number_of_robots"], idx=idx))
        self.description = "Bounds on the 2-norm of the s in the dual formulation of the polytopic constraints"


# Constraints of the form sqrt(sum(s^2)) =< 1
class s2normConstraintConstraints:

    def __init__(self, n_robots, idx):
        self.n_robots = n_robots
        self.idx = idx
        self.n_constraints = (n_robots-1)*2
        self.nh = self.n_constraints

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
            upper_bound.append(0.5 * np.sqrt(2))
        return upper_bound

    def get_constraints(self, model, params, settings, stage_idx):
        constraints = []

        for j in range(1, self.n_robots+1): 
            if j== self.idx:
                continue

            if self.idx < j:
                s_ij_0 = model.get(f"s_{self.idx}_{j}_0")
                s_ij_1 = model.get(f"s_{self.idx}_{j}_1")
            else:
                s_ij_0 = model.get(f"s_{j}_{self.idx}_0")
                s_ij_1 = model.get(f"s_{j}_{self.idx}_1")

            constraints.append(cd.norm_2(s_ij_0))
            constraints.append(cd.norm_2(s_ij_1))

            
            # constraints.append(cd.norm_2(cd.vertcat(s_ij_0, s_ij_1)))

        return constraints
