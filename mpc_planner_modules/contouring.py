import sys, os

sys.path.append(os.path.join(sys.path[0], "..", "solver_generator"))

import copy
import casadi as cd
import numpy as np

from control_modules import ObjectiveModule, Objective

from spline import Spline, Spline2D
from util.math import haar_difference_without_abs, huber_loss

class ContouringObjective:

    """
        Objective for tracking a 2D reference path with contouring costs
    """

    def __init__(self, settings):

        self.num_segments = settings["contouring"]["num_segments"]

    def define_parameters(self, params):
        params.add("contour", add_to_rqt_reconfigure=True)
        params.add("lag", add_to_rqt_reconfigure=True)

        params.add("terminal_angle", add_to_rqt_reconfigure=True)
        params.add("terminal_lag", add_to_rqt_reconfigure=True)
        params.add("terminal_contour", add_to_rqt_reconfigure=True)

        for i in range(self.num_segments):
            params.add(f"spline_x{i}_a", bundle_name="spline_x_a")
            params.add(f"spline_x{i}_b", bundle_name="spline_x_b")
            params.add(f"spline_x{i}_c", bundle_name="spline_x_c")
            params.add(f"spline_x{i}_d", bundle_name="spline_x_d")

            params.add(f"spline_y{i}_a", bundle_name="spline_y_a")
            params.add(f"spline_y{i}_b", bundle_name="spline_y_b")
            params.add(f"spline_y{i}_c", bundle_name="spline_y_c")
            params.add(f"spline_y{i}_d", bundle_name="spline_y_d")

            params.add(f"spline{i}_start", bundle_name="spline_start")

        return params

    def get_value(self, model, params, settings, stage_idx):
        cost = 0

        pos_x = model.get("x")
        pos_y = model.get("y")
        theta = model.get("theta")
        # v = model.get("vx")
        s = model.get("s")

        quadratic_from = 0.125
        if settings["contouring"]["use_huber"]:
            cost_f = lambda value: huber_loss(value, quadratic_from=quadratic_from)
        else:
            cost_f = lambda value: value**2

        contour_weight = params.get("contour")
        lag_weight = params.get("contour")

        # For normalization
        max_contour = 4.0
        max_lag = 4.0

        path = Spline2D(params, self.num_segments, s)
        path_x, path_y = path.at(s)
        path_dx_normalized, path_dy_normalized = path.deriv_normalized(s)

        contour_error = path_dy_normalized * (pos_x - path_x) - path_dx_normalized * (pos_y - path_y)
        lag_error = path_dx_normalized * (pos_x - path_x) + path_dy_normalized * (pos_y - path_y)

        cost += contour_weight * cost_f(contour_error / max_contour)
        cost += lag_weight * cost_f(lag_error / max_lag)

        # Terminal cost
        if True and stage_idx == settings["N"] - 1:

            terminal_angle_weight = params.get("terminal_angle")
            terminal_contour_weight = params.get("terminal_contour")
            terminal_lag_weight = params.get("terminal_lag")

            # Compute the angle w.r.t. the path
            path_angle = cd.atan2(path_dy_normalized, path_dx_normalized)
            angle_error = haar_difference_without_abs(theta, path_angle)

            # Penalize the angle error
            cost += terminal_angle_weight * (angle_error / np.pi)**2
            cost += terminal_contour_weight * contour_weight * cost_f(contour_error / max_contour)
            cost += terminal_lag_weight * lag_weight * cost_f(lag_error / max_lag)

        return cost


class ContouringModule(ObjectiveModule):

    def __init__(self, settings):
        super().__init__()
        self.module_name = "Contouring"  # Needs to correspond to the c++ name of the module
        self.import_name = "contouring.h"
        self.type = "objective"

        self.description = "MPCC: Tracks a 2D reference path with contouring costs"

        self.objectives = []
        self.objectives.append(ContouringObjective(settings))