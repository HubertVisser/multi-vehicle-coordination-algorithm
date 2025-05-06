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

    def __init__(self, settings, idx_i):
        super().__init__()

        self.module_name = f"s_2norm"  # c++ name of the module
        self.import_name = "s_2norm.h"

        self.constraints.append(s2normConstraintConstraints(settings, idx_i=idx_i))
        self.description = "Bounds on the 2-norm of the s in the dual formulation of the polytopic constraints"


# Constraints of the form sqrt(sum(s^2)) =< 1
class s2normConstraintConstraints:

    def __init__(self, settings, idx_i):
        self.number_of_robots = settings["number_of_robots"]
        self.idx_i = idx_i
        self.scheme = settings["scheme"]
        self.n_sides = 8
        self.n_constraints = (len(self.neighbour_range()) - 1) * self.n_sides #2
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
            # upper_bound.append(0.5 * np.sqrt(2))
            upper_bound.append(1)
            upper_bound.append(1)
        return upper_bound

    def neighbour_range(self):
        if self.scheme == 'distributed':
            return range(1, self.number_of_robots+1)
        elif self.scheme == 'centralised':
            return range(self.idx_i, self.number_of_robots+1)
        
    def get_constraints(self, model, params, settings, stage_idx):
        constraints = []

        for j in self.neighbour_range(): 
            if j== self.idx_i:
                continue

            s_ij_0 = model.get(f"s_{self.idx_i}_{j}_0")
            s_ij_1 = model.get(f"s_{self.idx_i}_{j}_1")
        

            angles = np.linspace(0, 2*np.pi, self.n_sides, endpoint=False)
            A = np.column_stack((np.cos(angles), np.sin(angles)))
            for k in range(self.n_sides):
                constraints.append(A[k, 0] * s_ij_0 + A[k, 1] * s_ij_1)
            
            # constraints.append(cd.norm_2(s_ij_0))
            # constraints.append(cd.norm_2(s_ij_1))

            # constraints.append(cd.norm_2(cd.vertcat(s_ij_0, s_ij_1)))

        return constraints
