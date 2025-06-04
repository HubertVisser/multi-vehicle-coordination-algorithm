import sys, os

sys.path.append(os.path.join(sys.path[0], "..", "solver_generator"))

import casadi as cd
import numpy as np

from control_modules import ObjectiveModule, Objective
from spline import Spline, Spline2D


class LambdaConsensusObjective(Objective):
    """
    Aim for consensus on the lambda value calculated by nmpc solver
    """

    def __init__(self, settings, robot_idx):
        self._idx = robot_idx
        self.number_of_robots = settings["number_of_robots"]

    def define_parameters(self, params):
        for j in range(1, self.number_of_robots+1):
            if j == self._idx:
                continue
            for k in range(4):
                params.add(f"lam_{j}_{self._idx}_{k}")

        params.add("consensus") 
        return params

    def get_value(self, model, params, settings, stage_idx):
        cost = 0.0

        consensus_weight = params.get("consensus")

        for j in range(1, self.number_of_robots+1):
            if j == self._idx:
                continue
            
            for k in range(4):
                cost += consensus_weight * (model.get(f"lam_{j}_{self._idx}_{k}") - params.get(f"lam_{j}_{self._idx}_{k}"))**2
        return cost


class LambdaConsensusModule(ObjectiveModule):

    def __init__(self, settings, robot_idx=0):
        super().__init__()
        self.module_name = f"LambdaConsensus_{robot_idx}"
        self.import_name = "lambda_consensus.h"
        self.type = "objective"
        self.description = "Aims for consensus on the lambda value calculated by NMPC solver"

        self.objectives = []
        self.objectives.append(LambdaConsensusObjective(settings, robot_idx))
