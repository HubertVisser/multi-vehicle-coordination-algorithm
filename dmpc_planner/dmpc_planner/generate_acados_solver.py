import os, sys
import numpy as np


from solver_generator.util.files import load_settings, get_current_package
from solver_generator.control_modules import ModuleManager
from solver_generator.generate_solver import generate_solver

# Import modules here from mpc_planner_modules
from mpc_planner_modules.mpc_base import MPCBaseModule
# from contouring import ContouringModule
# from goal_module import GoalModule
from mpc_planner_modules.path_reference_velocity import PathReferenceVelocityModule

# Import solver models that you want to use
from solver_generator.solver_model import BicycleModel2ndOrder


def configuration_basic(settings):
    modules = ModuleManager()
    model = BicycleModel2ndOrder()

    # Penalize ||a||_2^2 and ||w||_2^2
    base_module = modules.add_module(MPCBaseModule(settings))
    base_module.weigh_variable(var_name="throttle", weight_names="throttle")
    base_module.weigh_variable(var_name="delta", weight_names="steering")

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

generate()
