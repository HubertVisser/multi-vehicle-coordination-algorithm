import sys, os, shutil
import rospy

import numpy as np
import casadi as cd

from acados_template import AcadosModel
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver

from util.files import load_settings, write_to_yaml
from util.files import solver_name, solver_path, solver_settings_path, default_solver_path, default_acados_solver_path, acados_solver_path
from util.logging import print_value, print_success, print_header, print_warning, print_path
from util.parameters import Parameters, AcadosParameters

from solver_definition import define_parameters, objective, constraints, constraint_lower_bounds, constraint_upper_bounds
import solver_model



def create_acados_model(settings, model, modules):
    # Create an acados ocp model
    acados_model = AcadosModel()
    acados_model.name = solver_name(settings) 
    
    # Dynamics
    z = model.acados_symbolics_z()
    d = model.acados_symbolics_d()
    dyn_f_expl = model.get_acados_dynamics()

    # Parameters
    params = settings["params"]
    p = params.get_acados_p()

    # Constraints
    constr = cd.vertcat(*constraints(modules, z, d, p, model, settings, 1))

    if constr.shape[0] == 0:
        print("No constraints specified")
        constr = cd.SX()

    # stage cost
    cost_stage = objective(modules, z, d, p, model, settings, 1)

    # terminal cost
    cost_e = objective(modules, z, d, p, model, settings, settings["N"] - 1)

    # Formulating acados ocp model
    acados_model.x = model.get_acados_x()
    acados_model.u = model.get_acados_u()
    acados_model.f_expl_expr = dyn_f_expl
    acados_model.p = params.get_acados_parameters()
    acados_model.cost_expr_ext_cost = cost_stage
    acados_model.cost_expr_ext_cost_e = cost_e
    acados_model.con_h_expr = constr

    return acados_model


def generate_solver(modules, model, settings=None):

    if settings is None:
        settings = load_settings()

    scheme = settings["scheme"]
    solver_settings = settings["solver_settings"]
    skip_solver_generation = settings["skip_solver_generation"]

    print_header("Creating ACADOS" f"Solver: {settings['solver_name']}_solver") if scheme == 'distributed' else print_header("Creating ACADOS" f"Solver: {settings['name']}_solver")

    params = AcadosParameters()
    define_parameters(modules, params, settings)
    params.load_acados_parameters()
    settings["params"] = params

    modules.print()
    params.print()

    npar = params.length()

    model.load_settings(settings)
    model_acados = create_acados_model(settings, model, modules)

    # Create an acados ocp object
    ocp = AcadosOcp()
    ocp.model = model_acados

    # Set ocp dimensions
    ocp.dims.N = settings["N"]

    # Set cost types
    ocp.cost.cost_type = "EXTERNAL"
    # ocp.cost.cost_type_e = "EXTERNAL"

    # Number of inputs and states
    solver_name = model_acados.name
    nx = model.nx
    nlam = model.nlam
    ns_dual = model.ns
    nu = model.nu 

    if scheme == 'centralised':
        nu = nu + ns_dual + nlam

    # Set initial constraint
    ocp.constraints.x0 = np.zeros(nx)

    # Set state bound
    ocp.constraints.lbx = model.lower_bound_states.T.flatten()
    ocp.constraints.ubx = model.upper_bound_states.T.flatten() 
    ocp.constraints.idxbx = np.array(range(nx))

    # Set control input bound
    ocp.constraints.lbu = model.lower_bound_u.flatten() 
    ocp.constraints.ubu = model.upper_bound_u.flatten() 
    ocp.constraints.idxbu = np.array(range(nu))

    # Set path constraints bound 
    nc = ocp.model.con_h_expr.shape[0]
    ocp.constraints.lh = np.array(constraint_lower_bounds(modules))
    ocp.constraints.uh = np.array(constraint_upper_bounds(modules))

    # Slack for constraints
    add_slack = settings["add_slack"]
    slack_value = settings["slack_value"]
    if add_slack: #and ('nmpc' or 'centralised') in settings["solver_name"]:
        add_constraint_slack = True
        value = float(slack_value)

        ns = nx + nu 
        if add_constraint_slack:
            ns += nc
            ocp.constraints.idxsh = np.array(range(nc))
            # ns += 1
            # ocp.constraints.idxsh = np.array([0])  # dmin constraint

        ocp.constraints.idxsbx = np.array(range(nx))
        ocp.constraints.idxsbu = np.array(range(nu)) 

        # Slack for state bounds
        ocp.cost.zl = value * np.ones((ns,)) # gradient wrt lower slack at intermediate shooting nodes (0 to N-1)
        ocp.cost.zu = value * np.ones((ns,)) # gradient wrt upper slack at intermediate shooting nodes (0 to N-1)
        ocp.cost.Zl = value * np.ones((ns,)) # diagonal of Hessian wrt lower slack at intermediate shooting nodes (0 to N-1)
        ocp.cost.Zu = value * np.ones((ns,)) # diagonal of Hessian wrt upper slack at intermediate shooting nodes (0 to N-1)

        # ocp.constraints.idxsbx_e = np.array(range(nx))
        # ocp.cost.zl_e = value * np.ones(ns)
        # ocp.cost.zu_e = value * np.ones(ns)
        # ocp.cost.Zu_e = value * np.ones(ns)
        # ocp.cost.Zl_e = value * np.ones(ns)

        ocp.cost.zl_0 = value * np.ones(nu)
        ocp.cost.zu_0 = value * np.ones(nu)
        ocp.cost.Zu_0 = value * np.ones(nu)
        ocp.cost.Zl_0 = value * np.ones(nu)

    ocp.parameter_values = np.zeros(model_acados.p.size()[0])

    # horizon
    ocp.solver_options.tf = settings["N"] * settings["integrator_step"]
    ocp.solver_options.tol = 1e-2 # 1e-6  # 1e-2

    # Solver options
    # integrator option
    ocp.solver_options.integrator_type = "ERK"
    # ocp.solver_options.sim_method_num_stages = 4
    # ocp.solver_options.sim_method_num_steps = 3  # Number of divisions over the time horizon (ERK applied on each)

    # nlp solver options
    ocp.solver_options.nlp_solver_type = settings["solver_settings"]["solver_type"]
    if ocp.solver_options.nlp_solver_type == "SQP":
        ocp.solver_options.nlp_solver_max_iter = settings["solver_settings"]["iterations_centralised"]
    # ocp.solver_options.nlp_solver_warm_start_first_qp = 1
    # ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.hessian_approx = "EXACT"
    # ocp.solver_options.levenberg_marquardt = 1e-3  # Helps to resolve min step errors
    # ocp.solver_options.regularize_method = "MIRROR"
    # ocp.solver_options.globalization = "MERIT_BACKTRACKING"
    ocp.solver_options.globalization = "FIXED_STEP"
    # ocp.solver_options.eps_sufficient_descent = 1e-1
    # ocp.solver_options.qp_tol = 1e-5 # Important! (1e-3)
    ocp.solver_options.qp_tol = 1e-2 # Important! (1e-3)

    # qp solver options
    # Full Condensing: Suitable for small to medium-sized systems, leading to a dense QP with only control inputs as decision variables.
    # Offers efficiency for smaller problems but faces scalability and memory issues for larger systems.
    # Partial Condensing: Suitable for larger systems, providing a balance between problem size and computational complexity.
    # It allows for controlled reduction in problem size, making it more scalable and flexible but potentially more complex to implement.
    # ocp.solver_options.qp_solver = "FULL_CONDENSING_QPOASES"
    # ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM" 
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.qp_solver_iter_max = 500 # default = 50
    ocp.solver_options.qp_solver_warm_start = 1  # cold start / 1 = warm, 2 = warm primal and dual

    # code generation options
    ocp.code_export_directory = f"{os.path.dirname(os.path.abspath(__file__))}/../acados/{model_acados.name}"
    ocp.solver_options.print_level = 0

    # Generate the solver
    json_file_dir = f"{os.path.dirname(os.path.abspath(__file__))}/acados/{model_acados.name}/"
    json_file_name = json_file_dir + f"{model_acados.name}.json"
    os.makedirs(json_file_dir, exist_ok=True)

    if skip_solver_generation:
        print_header("Output")
        print_warning("Solver generation was disabled by the command line option. Skipped.", no_tab=True)

        solver_file = os.path.join(solver_path(solver_name), solver_name + ".json")
        solver = AcadosOcpSolver(acados_ocp=ocp, json_file=solver_file, build=False, generate=False)

        return solver, None
    
    else:
        print_header("Generating solver")
        solver = AcadosOcpSolver(acados_ocp=ocp, json_file=json_file_name)

        # simulator = AcadosSimSolver(ocp, json_file=json_file_name)
        print_header("Output")

        if os.path.exists(os.path.join(acados_solver_path(), f"{model_acados.name}")) and os.path.isdir(os.path.join(acados_solver_path(), f"{model_acados.name}")):
            shutil.rmtree(os.path.join(acados_solver_path(), f"{model_acados.name}"))

        shutil.move(os.path.join(default_acados_solver_path(settings), f"{model_acados.name}"), os.path.join(acados_solver_path(), f"{model_acados.name}"))  # Move the solver to this directory

    settings["params"].save_map(settings)
    model.save_map()

    # Save other settings
    solver_settings = dict()
    solver_settings["N"] = settings["N"]
    solver_settings["number_of_robots"] = settings["number_of_robots"]
    solver_settings["nx"] = nx
    solver_settings["nu"] = nu
    solver_settings["nlam"] = nlam
    solver_settings["ns_dual"] = ns_dual
    solver_settings["nd"] = nlam + ns_dual
    solver_settings["nx_nu"] = model.get_nx_nu()
    solver_settings["nvar"] = model.get_nvar()
    solver_settings["npar"] = settings["params"].length()

    if model_acados.name.startswith("solver"):
        path = solver_settings_path("solver_settings"+ model_acados.name[len("solver"):])   
    else:
        path = solver_settings_path()
    
    write_to_yaml(path, solver_settings)

    print_path("Solver", solver_path(settings), tab=True, end="")
    print_success(" -> generated")

    return solver, None


if __name__ == "__main__":
    generate_solver()