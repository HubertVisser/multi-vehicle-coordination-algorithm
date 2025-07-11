import sys, os

sys.path.append(os.path.join(sys.path[0], "..", "solver_generator"))

import casadi as cd
import numpy as np

from control_modules import ObjectiveModule, Objective
from spline import Spline, Spline2D


class PathReferenceVelocityObjective:
    """
    Track a reference velocity along the path component that may change with time
    """

    def __init__(self, settings, robot_idx):
        self._idx = robot_idx

    def define_parameters(self, params):

        params.add("velocity", add_to_rqt_reconfigure=True)
        params.add("reference_velocity", add_to_rqt_reconfigure=True)

        return params

    def get_value(self, model, params, settings, stage_idx):

        # # The cost is computed in the contouring cost when using CA-MPC
        # return 0.0
        
        v = model.get(f"vx_{self._idx}")
        # s = model.get("s")

        # path_velocity = Spline(params, "spline_v", self.num_segments, s)
        # velocity_reference = path_velocity.at(s)
        velocity_reference = params.get("reference_velocity")
        velocity_weight = params.get("velocity")

        # spline = Spline2D(params, self.num_segments, s)
        # path_dx_normalized, path_dy_normalized = spline.deriv_normalized(s)
        # path_angle = cd.atan2(path_dy_normalized, path_dx_normalized)
        # diff_angle = path_angle - psi

        # # Only weigh the component of the velocity in the direction of the path
        # v_path = cd.cos(diff_angle) * v

        # track the given velocity reference
        # return velocity_weight * ((v_path - velocity_reference) ** 2)
        return velocity_weight * ((v - velocity_reference) ** 2)
        


class PathReferenceVelocityModule(ObjectiveModule):

    def __init__(self, settings, robot_idx=0):
        super().__init__()
        self.module_name = f"PathReferenceVelocity_{robot_idx}"
        self.import_name = "path_reference_velocity.h"
        self.type = "objective"
        self.description = "Tracks a velocity in the direction of a 2D path"

        self.objectives = []
        self.objectives.append(PathReferenceVelocityObjective(settings, robot_idx))
