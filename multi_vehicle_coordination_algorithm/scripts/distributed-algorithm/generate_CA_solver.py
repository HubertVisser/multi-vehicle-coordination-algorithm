import os, sys
sys.path.append(os.path.join(sys.path[0], "..", "..", "..", "solver_generator"))
sys.path.append(os.path.join(sys.path[0], "..", "..", "..", "mpc_planner_modules"))

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
from ca_objective import MinimizeCollisionAvoidanceModule
from polytopic_dmin_constraints import PolytopicDminConstraintModule
from polytopic_si_constraints import PolytopicSidualConstraintModule
from polytopic_sj_constraints import PolytopicSjdualConstraintModule
from s_2_norm_constraints import s2normConstraintModule

# Import solver models that you want to use
from solver_model import CollisionAvoidanceModel


def configuration_basic(settings, idx):
    num_robots = settings["number_of_robots"]

    modules = ModuleManager()
    model = CollisionAvoidanceModel(num_robots, idx)

    # Penalize ||steering||_2^2
    base_module = modules.add_module(MPCBaseModule(settings))
    for i in range(1, num_robots+1):
        if i != idx:
            base_module.weigh_variable(var_name=f"s_{idx}_{i}_0", weight_names="s_dual",)
            base_module.weigh_variable(var_name=f"s_{idx}_{i}_1", weight_names="s_dual",)
        for j in range(1, num_robots+1):
            if j == i or (j != idx and i != idx):
                continue

            base_module.weigh_variable(var_name=f"lam_{i}_{j}_0", weight_names="lambda",)
            base_module.weigh_variable(var_name=f"lam_{i}_{j}_1", weight_names="lambda",)
            base_module.weigh_variable(var_name=f"lam_{i}_{j}_2", weight_names="lambda",)
            base_module.weigh_variable(var_name=f"lam_{i}_{j}_3", weight_names="lambda",)
            if i > j:
                base_module.weigh_variable(var_name=f"s_{j}_{i}_0", weight_names="s_dual",)
                base_module.weigh_variable(var_name=f"s_{j}_{i}_1", weight_names="s_dual",)
    
    modules.add_module(s2normConstraintModule(settings, idx))
    modules.add_module(MinimizeCollisionAvoidanceModule(settings, idx))
    modules.add_module(s2normConstraintModule(settings, idx))
    
    
    modules.add_module(PolytopicDminConstraintModule(settings, idx))
    modules.add_module(PolytopicSidualConstraintModule(settings, idx))
    modules.add_module(PolytopicSjdualConstraintModule(settings, idx))

    return model, modules


def generate(idx):
    settings = load_settings(package="multi_vehicle_coordination_algorithm")
    settings["solver_name"] = f"solver_ca_{idx}"
    settings["idx"] = idx

    model, modules = configuration_basic(settings, idx)

    solver, simulator = generate_solver(modules, model, settings)
    
    return solver, simulator

if __name__ == "__main__":
    generate(idx=1)
