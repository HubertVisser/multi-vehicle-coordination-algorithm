import sys, os

sys.path.append(os.path.join(sys.path[0], ".."))
sys.path.append(os.path.join(sys.path[0], "..", "..", "mpc_planner_modules", "scripts"))

import numpy as np

from util.files import load_test_settings, get_current_package
from util.parameters import Parameters

from solver_model import BicycleModel2ndOrder

from generate_solver import generate_solver

from control_modules import ModuleManager, ObjectiveModule, ConstraintModule
from mpc_base import MPCBaseModule
from contouring import ContouringModule
from goal_module import GoalModule
from path_reference_velocity import PathReferenceVelocityModule

# from ellipsoid_constraints import EllipsoidConstraintModule
# from gaussian_constraints import GaussianConstraintModule
# from guidance_constraints import GuidanceConstraintModule
# from linearized_constraints import LinearizedConstraintModule
# from scenario_constraints import ScenarioConstraintModule

# from solver_definition import define_parameters, objective, constraints, constraint_lower_bounds, constraint_upper_bounds, constraint_number


def solver_configuration(settings):
    modules = ModuleManager()
    model = BicycleModel2ndOrder()

    # Penalize ||steering||_2^2
    base_module = modules.add_module(MPCBaseModule(settings))
    base_module.weigh_variable(
        var_name="steering", 
        weight_names="steering",
    )
    # Penalize ||v - v_ref||_2^2
    base_module.weigh_variable(
        var_name="vx",
        weight_names=["velocity", "reference_velocity"],
        cost_function=lambda x, w: w[0] * (x - w[1]) ** 2,
    )

    # modules.add_module(ContouringModule(settings))

    modules.add_module(PathReferenceVelocityModule(settings, num_segments=settings["contouring"]["num_segments"]))

    # modules.add_module(EllipsoidConstraintModule(settings))

    return model, modules


def test_acados_solver_generation():
    modules = ModuleManager()

    settings = load_test_settings()
    settings["name"] = "test_solver"
    settings["max_obstacles"] = 12
    settings["N"] = 20
    settings["integrator_step"] = 0.2
    settings["contouring"] = dict()
    settings["contouring"]["num_segments"] = 8

    solver_settings = settings["solver_settings"]

    model, modules = solver_configuration(settings)

    solver, simulator = generate_solver(modules, model, settings)
    ocp = solver.acados_ocp
    assert ocp.model.name == "Solver"
    assert len(ocp.constraints.lh) == 0
    assert len(ocp.constraints.uh) == 0

    ocp_model = ocp.model

    assert ocp_model.x.shape[0] == 7
    assert ocp_model.u.shape[0] == 3
    p = np.zeros((ocp_model.p.shape[0], 1))
    z = np.zeros((ocp_model.x.shape[0] + ocp_model.u.shape[0], 1))
    solver.get_cost()

test_acados_solver_generation()
