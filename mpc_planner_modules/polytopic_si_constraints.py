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


class PolytopicSidualConstraintModule(ConstraintModule):

    def __init__(self, settings, idx_i):
        super().__init__()

        self.module_name = f"PolytopicConstraints_si_{idx_i}"  # c++ name of the module
        self.import_name = "polytopic_constraints.h"

        self.constraints.append(PolytopicSidualConstraints(settings, idx_i=idx_i, use_slack=False))
        self.description = "Polytopic set based collision avoidance constraints in dual formulation (constraint 5b)"


# Constraints of the form A_i^T @ lam_ij + s_ij = 0
class PolytopicSidualConstraints:

    def __init__(self, settings, idx_i, use_slack=False):
        self.length = settings["polytopic"]["length"]
        self.width = settings["polytopic"]["width"]
        self.number_of_robots = settings["number_of_robots"]
        self.scheme = settings["scheme"]
        self.idx_i = idx_i
        self.n_constraints = (self.number_of_robots - self.idx_i) * 2 if self.scheme == 'centralised' else (self.number_of_robots - 1) * 2
        self.nh = self.n_constraints
        self.use_slack = use_slack
        self.solver_name = settings.get("solver_name", None)

    def define_parameters(self, params):
        # pass
        if self.scheme == 'distributed' and self.solver_name.startswith("solver_nmpc"):
            for i in range(1, self.number_of_robots+1):
                for j in range(i, self.number_of_robots+1):
                        if i != j and (i == self.idx_i or j == self.idx_i):
                            params.add(f"s_{i}_{j}_0")
                            params.add(f"s_{i}_{j}_1")
        elif self.scheme == 'distributed' and self.solver_name.startswith("solver_ca"):
            for i in range(1, self.number_of_robots+1):
                params.add(f"x_{i}")
                params.add(f"y_{i}")
                params.add(f"theta_{i}")
                for j in range(1, self.number_of_robots+1):
                        if i != j and (i == self.idx_i):
                            params.add(f"lam_{i}_{j}_0")
                            params.add(f"lam_{i}_{j}_1")
                            params.add(f"lam_{i}_{j}_2")
                            params.add(f"lam_{i}_{j}_3")
                            

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

    def get_theta_i(self, model, params):
        if self.scheme == 'distributed' and self.solver_name.startswith("solver_ca"):
            return params.get(f"theta_{self.idx_i}")
        else:
            return model.get(f"theta_{self.idx_i}")
    
    def get_lam_ij(self, model, params, idx_j):
        if self.scheme == 'distributed' and self.solver_name.startswith("solver_ca"):
            return cd.vertcat(  params.get(f"lam_{self.idx_i}_{idx_j}_0"), 
                                params.get(f"lam_{self.idx_i}_{idx_j}_1"), 
                                params.get(f"lam_{self.idx_i}_{idx_j}_2"), 
                                params.get(f"lam_{self.idx_i}_{idx_j}_3"))
        else:
            return cd.vertcat(  model.get(f"lam_{self.idx_i}_{idx_j}_0"), 
                                model.get(f"lam_{self.idx_i}_{idx_j}_1"), 
                                model.get(f"lam_{self.idx_i}_{idx_j}_2"), 
                                model.get(f"lam_{self.idx_i}_{idx_j}_3"))
        
    def get_s_ij(self, model, params, idx_j):
        if self.scheme == 'distributed' and self.solver_name.startswith("solver_nmpc"):
            if self.idx_i > idx_j:
                return cd.vertcat(  params.get(f"s_{idx_j}_{self.idx_i}_0"), 
                                    params.get(f"s_{idx_j}_{self.idx_i}_1"))
            else:
                return cd.vertcat(  params.get(f"s_{self.idx_i}_{idx_j}_0"), 
                                    params.get(f"s_{self.idx_i}_{idx_j}_1"))
        else:
            if self.idx_i > idx_j:
                return cd.vertcat(  model.get(f"s_{idx_j}_{self.idx_i}_0"), 
                                    model.get(f"s_{idx_j}_{self.idx_i}_1"))
            else:
                return cd.vertcat(  model.get(f"s_{self.idx_i}_{idx_j}_0"), 
                                    model.get(f"s_{self.idx_i}_{idx_j}_1"))
            
    def get_constraints(self, model, params, settings, stage_idx):
        constraints = []

        # Constraints form ego robot (i) to all neighbouring robots (j)
        theta_i = self.get_theta_i(model, params)
        A_i = get_A(theta_i)

        start_idx = 1 if self.scheme == 'distributed' else self.idx_i
        for j in range(start_idx, self.number_of_robots+1):  
            if j != self.idx_i:
                lam_ij = self.get_lam_ij(model, params, j)
                s_ij = self.get_s_ij(model, params, j)

                constraint = A_i.T @ lam_ij + s_ij
                constraints.append(constraint[0])
                constraints.append(constraint[1])

        return constraints
