
import sys, os

sys.path.append(os.path.join(sys.path[0], "..", "..", "solver_generator"))

import importlib
import copy
import casadi as cd
import numpy as np

from util.math import huber_loss
import llm_generated


from control_modules import ObjectiveModule, Objective


class LLMObjective:

    def __init__(self, settings):
        self.num_segments = settings["contouring"]["num_segments"]

    def define_parameters(self, params):
        llm_generated.define_parameters(params)

        params.add("goal_x")
        params.add("goal_y")

        params.add("reference_velocity", add_to_rqt_reconfigure=True)

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
        return llm_generated.get_value(model, params, settings, stage_idx)
    

class LLMModule(ObjectiveModule):

    def __init__(self, settings):
        super().__init__()

        importlib.reload(llm_generated)

        self.module_name = "LLMModule"  # Needs to correspond to the c++ name of the module
        self.import_name = "llm_module.h"
        self.type = "objective"
        self.description = "LLM generated cost function"

        self.objectives = []
        self.objectives.append(LLMObjective(settings))