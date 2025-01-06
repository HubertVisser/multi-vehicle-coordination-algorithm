import sys, os

import numpy as np

from solver_generator.util.files import load_test_settings, get_current_package
from solver_generator.util.parameters import Parameters

from solver_generator.solver_model import BicycleModel2ndOrder

from solver_generator.generate_solver import generate_solver

from solver_generator.control_modules import ModuleManager, ObjectiveModule, ConstraintModule
from mpc_planner_modules.mpc_base import MPCBaseModule
from mpc_planner_modules.contouring import ContouringModule
from mpc_planner_modules.goal_module import GoalModule
from mpc_planner_modules.path_reference_velocity import PathReferenceVelocityModule

from mpc_planner_modules.ellipsoid_constraints import EllipsoidConstraintModule
from mpc_planner_modules.gaussian_constraints import GaussianConstraintModule
from mpc_planner_modules.guidance_constraints import GuidanceConstraintModule
from mpc_planner_modules.linearized_constraints import LinearizedConstraintModule
from mpc_planner_modules.scenario_constraints import ScenarioConstraintModule

# from solver_definition import define_parameters, objective, constraints, constraint_lower_bounds, constraint_upper_bounds, constraint_number


def solver_configuration(settings):
    modules = ModuleManager()
    model = BicycleModel2ndOrder()

    # Penalize ||a||_2^2 and ||w||_2^2
    base_module = modules.add_module(MPCBaseModule(settings))
    base_module.weigh_variable(var_name="a", weight_names="acceleration")
    base_module.weigh_variable(var_name="w", weight_names="angular_velocity")

    modules.add_module(ContouringModule(settings))

    modules.add_module(PathReferenceVelocityModule(settings))

    modules.add_module(EllipsoidConstraintModule(settings))

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
    settings["contouring"]["dynamic_velocity_reference"] = False

    solver_settings = settings["solver_settings"]
    solver_settings["solver"] = "acados"

    model, modules = solver_configuration(settings)

    solver, simulator = generate_solver(modules, model, settings)
    ocp = solver.acados_ocp
    assert ocp.model.name == "Solver"
    assert len(ocp.constraints.lh) == 12
    assert len(ocp.constraints.uh) == 12

    ocp_model = ocp.model

    assert ocp_model.x.shape[0] == 5
    assert ocp_model.u.shape[0] == 2
    p = np.zeros((ocp_model.p.shape[0], 1))
    z = np.zeros((ocp_model.x.shape[0] + ocp_model.u.shape[0], 1))
    solver.get_cost()