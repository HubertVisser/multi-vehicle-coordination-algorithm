import os, sys
sys.path.append(os.path.join(sys.path[0], "..", "..", "solver_generator"))
sys.path.append(os.path.join(sys.path[0], "..", "..", "mpc_planner_modules"))

import numpy as np
import casadi as cd

from util.files import load_settings, get_current_package
from control_modules import ModuleManager
from generate_solver import generate_solver

# Import modules here from mpc_planner_modules
from mpc_base import MPCBaseModule

from contouring import ContouringModule
from goal_module import GoalModule
from path_reference_velocity import PathReferenceVelocityModule
from polytopic_dmin_constraints import PolytopicDminConstraintModule
from polytopic_si_constraints import PolytopicSidualConstraintModule
from polytopic_sj_constraints import PolytopicSjdualConstraintModule
from s_2_norm_constraints import s2normConstraintModule

# Import solver models that you want to use
from solver_model import BicycleModel2ndOrderMultiRobot


def configuration_basic(settings):
    num_robots = settings["number_of_robots"]

    modules = ModuleManager()
    model = BicycleModel2ndOrderMultiRobot(num_robots)

    for n in range(1,num_robots+1):
        # Penalize ||steering||_2^2
        base_module = modules.add_module(MPCBaseModule(settings, n))
        base_module.weigh_variable(var_name=f"steering_{n}", weight_names="steering")
        base_module.weigh_variable(var_name=f"throttle_{n}", weight_names="throttle")
        for j in range(1,num_robots+1):
            if j != n:
                base_module.weigh_variable(var_name=f"lam_{n}_{j}_0", weight_names="lambda",)
                base_module.weigh_variable(var_name=f"lam_{n}_{j}_1", weight_names="lambda",)
                base_module.weigh_variable(var_name=f"lam_{n}_{j}_2", weight_names="lambda",)
                base_module.weigh_variable(var_name=f"lam_{n}_{j}_3", weight_names="lambda",)
                base_module.weigh_variable(var_name=f"s_{n}_{j}", weight_names="s_dual",
                                           cost_function=lambda x, w: w[0] * x.T @ x)
        
        modules.add_module(ContouringModule(settings, n))
        modules.add_module(PathReferenceVelocityModule(settings, n))
        
        modules.add_module(PolytopicDminConstraintModule(settings, n))
        modules.add_module(PolytopicSidualConstraintModule(settings, n))
        modules.add_module(PolytopicSjdualConstraintModule(settings, n))
    
    # modules.add_module(s2normConstraintModule(settings))
        
    return model, modules


def generate():
    settings = load_settings(package="dmpc_planner_decentralised")
    print(settings)

    model, modules = configuration_basic(settings)

    solver, simulator = generate_solver(modules, model, settings)
    return solver, simulator

if __name__ == "__main__":
    generate()
