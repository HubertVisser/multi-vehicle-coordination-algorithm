
"""
Polytopic Constraints (Minimum distance constraints between robots represented as polytopic sets using dual variables)
"""

import numpy as np
import casadi as cd

import sys
import os

sys.path.append(os.path.join(sys.path[0], "..", "solver_generator"))

from util.math import get_A
from control_modules import ConstraintModule


class PolytopicSjdualConstraintModule(ConstraintModule):

    def __init__(self, settings, idx_i):
        super().__init__()

        self.module_name = f"PolytopicConstraints_sj_{idx_i}"  # c++ name of the module
        self.import_name = "polytopic_constraints.h"

        self.constraints.append(PolytopicSjdualConstraints(settings, idx_i=idx_i, use_slack=False))
        self.description = "Polytopic set based collision avoidance constraints in dual formulation (constraint 5c)"


# Constraints of the form A_j^T @ lam_ji - S_ji = 0
class PolytopicSjdualConstraints:

    def __init__(self, settings, idx_i, use_slack=False):
        self.length = settings["polytopic"]["length"]
        self.width = settings["polytopic"]["width"]
        self.number_of_robots = settings["number_of_robots"]
        self.scheme = settings["scheme"]
        self.idx_i = idx_i
        self.n_constraints = (len(self.neighbour_range()) - 1) * 2 
        self.nh = self.n_constraints
        self.use_slack = use_slack
        self.solver_name = settings.get("solver_name", None)

    def define_parameters(self, params):
        pass
        # if self.scheme == 'distributed' and self.solver_name.startswith("solver_ca"):
        #     for i in range(1, self.number_of_robots+1):
        #         for j in range(1, self.number_of_robots+1):
        #                 if i != j and (j == self.idx_i):
        #                     params.add(f"lam_{i}_{j}_0")
        #                     params.add(f"lam_{i}_{j}_1")
        #                     params.add(f"lam_{i}_{j}_2")
        #                     params.add(f"lam_{i}_{j}_3")

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

    def get_theta_j(self, model, params, idx_j):
        if self.scheme == 'distributed' and self.solver_name.startswith("solver_ca"):
            return params.get(f"theta_{idx_j}")
        return model.get(f"theta_{idx_j}")
    
    def get_lam_ji(self, model, params, idx_j):
        # if self.scheme == 'distributed' and self.solver_name.startswith("solver_ca"):
        #     return cd.vertcat(  params.get(f"lam_{self.idx_i}_{idx_j}_0"), 
        #                         params.get(f"lam_{self.idx_i}_{idx_j}_1"), 
        #                         params.get(f"lam_{self.idx_i}_{idx_j}_2"), 
        #                         params.get(f"lam_{self.idx_i}_{idx_j}_3"))
        # else:
            return cd.vertcat(  model.get(f"lam_{idx_j}_{self.idx_i}_0"), 
                                model.get(f"lam_{idx_j}_{self.idx_i}_1"), 
                                model.get(f"lam_{idx_j}_{self.idx_i}_2"), 
                                model.get(f"lam_{idx_j}_{self.idx_i}_3"))
        
    def get_s_ij(self, model, params, idx_j):
        # if self.idx_i > idx_j:
        #     return cd.vertcat(  model.get(f"s_{idx_j}_{self.idx_i}_0"), 
        #                         model.get(f"s_{idx_j}_{self.idx_i}_1"))
        # else:
            return cd.vertcat(  model.get(f"s_{self.idx_i}_{idx_j}_0"), 
                                model.get(f"s_{self.idx_i}_{idx_j}_1"))
    
    def neighbour_range(self):
        if self.scheme == 'distributed':
            return range(1, self.number_of_robots+1)
        elif self.scheme == 'centralised':
            return range(self.idx_i, self.number_of_robots+1)
        
    def get_constraints(self, model, params, settings, stage_idx):
        constraints = []

        # Constraints from all neighbouring robots (j) to the ego robot (i)
        for j in self.neighbour_range(): 
            if j == self.idx_i:
                continue

            theta_j = self.get_theta_j(model, params, j)            
            A_j = get_A(theta_j)

            lam_ji = self.get_lam_ji(model, params, j)
            s_ij = self.get_s_ij(model, params, j)

            constraint = A_j.T @ lam_ji - s_ij
            constraints.append(constraint[0])
            constraints.append(constraint[1])

        return constraints
