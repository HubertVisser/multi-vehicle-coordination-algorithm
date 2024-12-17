import os, sys

sys.path.append(os.path.join(sys.path[0], "..", "..", "solver_generator"))
sys.path.append(os.path.join(sys.path[0], "..", "..", "mpc_planner_modules", "scripts"))

import numpy as np

from util.files import load_settings, get_current_package
from control_modules import ModuleManager
from generate_solver import generate_solver

# Import modules here from mpc_planner_modules
import llm_module

# from ellipsoid_constraints import EllipsoidConstraintModule
from guidance_constraints import GuidanceConstraintModule
from decomp_constraints import DecompConstraintModule

# Import solver models that you want to use
from solver_model import ContouringSecondOrderUnicycleModel


def configuration_generated_llm(settings):
    modules = ModuleManager()
    model = ContouringSecondOrderUnicycleModel()
    lower_bound = [-2.0, -0.8, -2000.0, -2000.0, -np.pi * 2, -1.0, -1.0]
    upper_bound = [2.0, 0.8, 2000.0, 2000.0, np.pi * 2, 3.0, 10000.0]
    model.set_bounds(lower_bound, upper_bound)

    modules.add_module(llm_module.LLMModule(settings))

    # modules.add_module(GuidanceConstraintModule(settings, constraint_submodule=EllipsoidConstraintModule))
    modules.add_module(EllipsoidConstraintModule(settings))
    modules.add_module(DecompConstraintModule(settings))
    return model, modules

def generate():
    settings = load_settings(package="mpc_planner_py")
    print(settings)

    model, modules = configuration_generated_llm(settings)

    solver, simulator = generate_solver(modules, model, settings)
    return solver, simulator