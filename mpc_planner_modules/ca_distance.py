import sys, os

sys.path.append(os.path.join(sys.path[0], "..", "solver_generator"))

import casadi as cd
import numpy as np

from control_modules import ObjectiveModule, Objective
from util.math import get_b


class MinimizeCollisionAvoidanceObjective:
    """
    Minimize the distance between two vehicles modeled as polytopes
    """

    def __init__(self, settings, idx_i):
        self.idx_i = idx_i
        self._decentralised = settings["decentralised"]
        self.length = settings["polytopic"]["length"]
        self.width = settings["polytopic"]["width"]
        self.number_of_robots = settings["number_of_robots"]

    def define_parameters(self, params):
        
        params.add("dmin_objective")
    
    def get_pos_theta_i(self, params):
        pos_x_i = params.get(f"x_{self.idx_i}")
        pos_y_i = params.get(f"y_{self.idx_i}")
        pos_i = cd.vertcat(pos_x_i, pos_y_i)   # Center of gravity
        theta_i = params.get(f"theta_{self.idx_i}")
        return pos_i, theta_i
        
    def get_pos_theta_j(self, params, idx_j):
        pos_x_j = params.get(f"x_{idx_j}")
        pos_y_j = params.get(f"y_{idx_j}")
        pos_j = cd.vertcat(pos_x_j, pos_y_j)   # Center of gravity
        theta_j = params.get(f"theta_{idx_j}")
        return pos_j, theta_j
        
    def get_lam_ij(self, model, idx_j):
        return cd.vertcat(  model.get(f"lam_{self.idx_i}_{idx_j}_0"), 
                            model.get(f"lam_{self.idx_i}_{idx_j}_1"), 
                            model.get(f"lam_{self.idx_i}_{idx_j}_2"), 
                            model.get(f"lam_{self.idx_i}_{idx_j}_3"))
    
    def get_lam_ji(self, model, idx_j):
        return cd.vertcat(  model.get(f"lam_{idx_j}_{self.idx_i}_0"), 
                            model.get(f"lam_{idx_j}_{self.idx_i}_1"), 
                            model.get(f"lam_{idx_j}_{self.idx_i}_2"), 
                            model.get(f"lam_{idx_j}_{self.idx_i}_3"))
        
    def get_value(self, model, params, settings, stage_idx):
        cost = 0

        pos_i, theta_i = self.get_pos_theta_i(params)
        b_i = get_b(pos_i, theta_i, self.length, self.width)
        dmin_weight = params.get("dmin_objective")

        # Objective for all neighbouring robots (j)
        for j in range(1, self.number_of_robots+1): 
            if j != self.idx_i:
    
                pos_j, theta_j = self.get_pos_theta_j(params, j)
                lam_ij = self.get_lam_ij(model, j)
                lam_ji = self.get_lam_ji(model, j)
                assert lam_ij.shape == (4,1)

                b_j = get_b(pos_j, theta_j, self.length, self.width)

                cost += -1 * dmin_weight *(- b_i.T @ lam_ij - b_j.T @ lam_ji)
        
        return cost
        


class MinimizeCollisionAvoidanceModule(ObjectiveModule):

    def __init__(self, settings, robot_idx=0):
        super().__init__()
        self.module_name = f"MinimizeCollisionAvoidanceModule_{robot_idx}"
        self.import_name = "MinimizeCollisionAvoidanceModule.h"
        self.type = "objective"
        self.description = "Minimize distance between two vehicles modeled as polytopes"

        self.objectives = []
        self.objectives.append(MinimizeCollisionAvoidanceObjective(settings, robot_idx))
