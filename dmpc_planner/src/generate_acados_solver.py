import os, sys

sys.path.append(os.path.join(sys.path[0], "..","..", "solver_generator"))
sys.path.append(os.path.join(sys.path[0], "..", "..", "modules", "scripts"))

import numpy as np

from util.files import load_settings, get_current_package
from control_modules import ModuleManager
from generate_solver import generate_solver

# Import modules here from mpc_planner_modules
from mpc_base import MPCBaseModule
# from contouring import ContouringModule
# from goal_module import GoalModule
from path_reference_velocity import PathReferenceVelocityModule

# Import solver models that you want to use
from solver_model import BicycleModel2ndOrder


def configuration_basic(settings):
    modules = ModuleManager()
    model = BicycleModel2ndOrder()

    # Penalize ||a||_2^2 and ||w||_2^2
    base_module = modules.add_module(MPCBaseModule(settings))
    base_module.weigh_variable(var_name="th", weight_names="throttle")
    base_module.weigh_variable(var_name="st", weight_names="steering")

    # modules.add_module(GoalModule(settings))
    # modules.add_module(ContouringModule(settings, num_segments=settings["contouring"]["num_segments"]))
    modules.add_module(PathReferenceVelocityModule(settings, num_segments=settings["contouring"]["num_segments"]))


    return model, modules

def generate():
    settings = load_settings(package="dmpc_planner")
    print(settings)

    model, modules = configuration_basic(settings)

    solver, simulator = generate_solver(modules, model, settings)
    return solver, simulator