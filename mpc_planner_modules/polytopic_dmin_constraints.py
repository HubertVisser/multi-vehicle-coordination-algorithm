"""
Polytopic Constraints (Minimum distance constraints between robots represented as polytopic sets using dual variables)
"""

import numpy as np
import casadi as cd

import sys
import os

sys.path.append(os.path.join(sys.path[0], "..", "solver_generator"))

from util.math import get_b
from control_modules import ConstraintModule


class PolytopicDminConstraintModule(ConstraintModule):

    def __init__(self, settings, idx_i):
        super().__init__()

        self.module_name = f"PolytopicConstraints_dmin_{idx_i}"  # c++ name of the module
        self.import_name = "polytopic_constraints.h"

        self.constraints.append(PolytopicDminConstraints(settings, idx_i=idx_i, use_slack=False))
        self.description = "Polytopic set based collision avoidance constraints in dual formulation (constraint 5a)"


# Constraints of the form -b_i^T * lam_ij - b_j^T * lam_ji >= d_min
class PolytopicDminConstraints:

    def __init__(self, settings, idx_i, use_slack=False):
        self.d_min = settings["polytopic"]["d_min"]
        self.length = settings["polytopic"]["length"]
        self.width = settings["polytopic"]["width"]
        self.number_of_robots = settings["number_of_robots"]
        self.idx_i = idx_i
        self.n_constraints = self.number_of_robots - 1
        self.nh = self.n_constraints
        self.use_slack = use_slack
        self.solver_name = settings.get("solver_name", None)
        self.scheme = settings["scheme"]

    def define_parameters(self, params):
        if self.scheme == 'distributed' and self.solver_name.startswith("solver_nmpc"):
            for j in range(1, self.number_of_robots+1):
                if j != self.idx_i:
                    params.add(f"x_{j}")
                    params.add(f"y_{j}")
                    params.add(f"theta_{j}")
                for k in range(1, self.number_of_robots+1):
                    if j == k or k != self.idx_i:
                        continue
                    params.add(f"lam_{j}_{k}_0")
                    params.add(f"lam_{j}_{k}_1")
                    params.add(f"lam_{j}_{k}_2")
                    params.add(f"lam_{j}_{k}_3")
        if self.scheme == 'distributed' and self.solver_name.startswith("solver_ca"):
            for j in range(1, self.number_of_robots+1):
                params.add(f"x_{j}")
                params.add(f"y_{j}")
                params.add(f"theta_{j}")

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

    def get_pos_theta_i(self, model, params):
        if self.scheme == 'distributed' and self.solver_name.startswith("solver_ca"):
            pos_x_i = params.get(f"x_{self.idx_i}")
            pos_y_i = params.get(f"y_{self.idx_i}")
            pos_i = cd.vertcat(pos_x_i, pos_y_i)   # Center of gravity
            theta_i = params.get(f"theta_{self.idx_i}")
            return pos_i, theta_i
        else:
            pos_x_i = model.get(f"x_{self.idx_i}")
            pos_y_i = model.get(f"y_{self.idx_i}")
            pos_i = cd.vertcat(pos_x_i, pos_y_i)
            theta_i = model.get(f"theta_{self.idx_i}")
            return pos_i, theta_i
                
    def get_pos_theta_j(self, model, params, idx_j):
        if self.scheme == 'distributed':
            pos_x_j = params.get(f"x_{idx_j}")
            pos_y_j = params.get(f"y_{idx_j}")
            theta_j = params.get(f"theta_{idx_j}")
            return cd.vertcat(pos_x_j, pos_y_j), theta_j   # Center of gravity
        else:
            pos_x_j = model.get(f"x_{idx_j}")
            pos_y_j = model.get(f"y_{idx_j}")
            theta_j = model.get(f"theta_{idx_j}")
            return cd.vertcat(pos_x_j, pos_y_j), theta_j

    def get_lam_ij(self, model, params, idx_j):
        
        return cd.vertcat(  model.get(f"lam_{self.idx_i}_{idx_j}_0"), 
                            model.get(f"lam_{self.idx_i}_{idx_j}_1"), 
                            model.get(f"lam_{self.idx_i}_{idx_j}_2"), 
                            model.get(f"lam_{self.idx_i}_{idx_j}_3"))
    
    def get_lam_ji(self, model, params, idx_j):
        if self.scheme == 'distributed' and self.solver_name.startswith("solver_nmpc"):
            return cd.vertcat(  params.get(f"lam_{idx_j}_{self.idx_i}_0"), 
                                params.get(f"lam_{idx_j}_{self.idx_i}_1"), 
                                params.get(f"lam_{idx_j}_{self.idx_i}_2"), 
                                params.get(f"lam_{idx_j}_{self.idx_i}_3"))
        else:
            return cd.vertcat(  model.get(f"lam_{idx_j}_{self.idx_i}_0"), 
                                model.get(f"lam_{idx_j}_{self.idx_i}_1"), 
                                model.get(f"lam_{idx_j}_{self.idx_i}_2"), 
                                model.get(f"lam_{idx_j}_{self.idx_i}_3"))
    
    def get_constraints(self, model, params, settings, stage_idx):
        constraints = []

        # States
        pos_i, theta_i = self.get_pos_theta_i(model, params)
        b_i = get_b(pos_i, theta_i, self.length, self.width)
        
        # Constraints for all neighbouring robots (j)
        for j in range(1, self.number_of_robots+1): 
            if j == self.idx_i:
                continue
    
            pos_j, theta_j = self.get_pos_theta_j(model, params, j)
            lam_ij = self.get_lam_ij(model, params, j)
            lam_ji = self.get_lam_ji(model, params, j)
            assert lam_ij.shape == (4,1)

            b_j = get_b(pos_j, theta_j, self.length, self.width)

            constraints.append(- b_i.T @ lam_ij - b_j.T @ lam_ji)

        return constraints
