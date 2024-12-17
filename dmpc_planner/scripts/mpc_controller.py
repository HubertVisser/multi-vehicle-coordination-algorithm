import os, sys
import pathlib
path = pathlib.Path(__file__).parent.resolve()
sys.path.append(os.path.join(path))
sys.path.append(os.path.join(sys.path[0], "..", "..", "solver_generator"))

acados_path = os.path.join(path, "..", "..", "mpc_planner_solver", "acados", "solver")

import numpy as np
import math

import generate_system_solver
from acados_template import AcadosModel
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver

from timer import Timer
from util.files import solver_path, load_settings

from util.logging import print_warning, print_value, print_success, TimeTracker, print_header
from util.realtime_parameters import ForcesRealTimeModel, AcadosRealTimeModel


class MPCPlanner:

    def __init__(self, settings):
        self._settings = settings
        self._N = self._settings["N"]
        self._dt = self._settings["integrator_step"]
        self._braking_acceleration = self._settings["braking_acceleration"]

        self._projection_func = lambda trajectory: trajectory # No projection

        self.init_solver()
    
        self._mpc_feasible = False
        self.time_tracker = TimeTracker(self._settings["solver_settings"]["solver"])

        print_header("Starting MPC")

    def init_solver(self):
        if self._settings["solver_settings"]["solver"] == "forces":
            self.init_forces_solver()
        elif self._settings["solver_settings"]["solver"] == "acados":
            self.init_acados_solver()
        else:
            raise IOError("Unknown solver specified in settings.yaml (should be 'acados' or 'forces')")

    def init_forces_solver(self):
        import forcespro.nlp

        solver_file = solver_path(self._settings)
        if not os.path.isdir(solver_file):
            raise IOError(f"Solver {solver_file} does not exist")
        try:
            print("Loading solver %s" % solver_file)
            self._solver = forcespro.nlp.Solver.from_directory(solver_file)
            self._simulator = self._solver.dynamics
        except Exception as e:
            print("FAILED TO LOAD SOLVER")
            raise e

        self._model = ForcesRealTimeModel(self._settings, self._solver_settings)

    def init_acados_solver(self):
        # The generation software
        if hasattr(self, "_solver"):
            del self._solver, self._simulator, self._solver_settings
        # if hasattr(self, "_mpc_x_plan"):
            # del self._mpc_x_plan, self._mpc_u_plan

        self._solver, self._simulator = generate_system_solver.generate()
        self._solver_settings = load_settings("solver_settings", package="mpc_planner_solver")

        # acados_ocp = AcadosOcp(acados_path=acados_path)
        # self._solver = AcadosOcpSolver(acados_ocp)
        self._model = AcadosRealTimeModel(self._settings, self._solver_settings, package="mpc_planner_solver")

        self._nx = self._solver_settings["nx"]
        self._nu = self._solver_settings["nu"]
        self._nvar = self._solver_settings["nvar"]
        self._prev_trajectory = np.zeros((self._N, self._nvar))

        if hasattr(self, "_mpc_x_plan"):
            del self._mpc_x_plan, self._mpc_u_plan
        
        self._mpc_feasible = False

        print_success("Acados solver generated")

    def solve(self, xinit, p):
        if not hasattr(self, "_solver"):
            output = dict()
            self.set_infeasible(output)
            print("Solver is being regenerated...")
            return output, False, []

        # Initialize the initial guesses
        if not hasattr(self, "_mpc_x_plan"):
            self._mpc_x_plan = np.tile(np.array(xinit).reshape((-1, 1)), (1, self._N))

        if not hasattr(self, "_mpc_u_plan"):
            self._mpc_u_plan = np.zeros((self._nu, self._N))

        if self._mpc_feasible:
            # Shifted
            # self._x_traj_init = np.concatenate((self._mpc_x_plan[:, 1:], self._mpc_x_plan[:, -1:]), axis=1)
            # self._u_traj_init = np.concatenate((self._mpc_u_plan[:, 1:], self._mpc_u_plan[:, -1:]), axis=1)            

            # Not shifted
            self._x_traj_init =  self._mpc_x_plan
            self._u_traj_init = self._mpc_u_plan
        else:
            # Brake (model specific)
            self._x_traj_init = self.get_braking_trajectory(xinit)
            self._x_traj_init = self._projection_func(self._x_traj_init)

            # Xinit everywhere (could be infeasible)
            # self._x_traj_init = np.tile(np.array(xinit).reshape((-1, 1)), (1, self._N))

            self._u_traj_init = np.zeros((self._nu, self._N))
            self._solver.reset(reset_qp_solver_mem=1)
            self._solver.options_set('warm_start_first_qp', False)

        if self._settings["solver_settings"]["solver"] == "forces":
            return self.solve_forces(xinit, p)
        if self._settings["solver_settings"]["solver"] == "acados":
            return self.solve_acados(xinit, p)

    def solve_acados(self, xinit, p):
        try:
            # Set initial state
            self._solver.constraints_set(0, 'lbx', np.array(xinit))
            self._solver.constraints_set(0, 'ubx', np.array(xinit))

            npar = int(len(p) / (self._N + 1))
            for k in range(0, self._N):
                self._solver.set(k, 'x', self._x_traj_init[:, k])
                self._solver.set(k, 'u', self._u_traj_init[:, k])
                self._solver.set(k, 'p', np.array(p[k*npar:(k+1)*npar])) # params for the current stage

            self._solver.set(self._N, 'p', np.array(p[self._N*npar:(self._N + 1)*npar])) # Repeats the final set of parameters

            solve_time = 0.
            for it in range(self._settings["solver_settings"]["acados"]["iterations"]):
                status = self._solver.solve()
                solve_time += float(self._solver.get_stats('time_tot')) * 1000.

            output = dict()
            if status != 0: #and status != 2: # infeasible
                print_warning(f"Optimization was infeasible (exitflag = {status})")
                
                self.set_infeasible(output)
                return output, False, []

            self._mpc_feasible = True
            self._model.load(self._solver)

            output = dict()
            output["v"] = self._model.get(1, "v")
            output["a"] = self._model.get(0, "a")
            output["w"] = self._model.get(0, "w")

            self.time_tracker.add(solve_time)
            # print_value("action",f"{output['a']}, {output['w']}")

            self._prev_trajectory = self._model.get_trajectory(self._solver, self._mpc_x_plan, self._mpc_u_plan)
        except AttributeError:
            output = dict()
            self.set_infeasible(output)
            print("Solver is being regenerated...")
            return output, False, []

        return output, True, self._prev_trajectory

    def get_cost_acados(self):
        return self._solver.get_cost()

    def solve_forces(self, xinit, p):
        """
        FORCES
        """
        # set initial condition
        # x0 = np.transpose(np.tile(x0i, (1, self._N)))
        # x0i = np.zeros((self._solver_settings["nvar"],))

        x0 = np.concatenate([self._u_traj_init, self._x_traj_init])

        problem = {"x0": x0, "xinit": xinit, "all_parameters": p}

        # Time to solve the NLP!
        forces_output, exitflag, info = self._solver.solve(problem)

        # Make sure the solver has exited properly.
        output = dict()
        if exitflag != 1:
            print_warning(f"Optimization was infeasible (exitflag = {exitflag})")
            self.set_infeasible(output)
            return output, False, []

        self._mpc_feasible = True
        self._model.load(forces_output)

        output["v"] = self._model.get(1, "v")
        output["a"] = self._model.get(0, "a")
        output["w"] = self._model.get(0, "w")

        # Save the trajectory
        self._prev_trajectory = self._model.get_trajectory(forces_output, self._mpc_x_plan, self._mpc_u_plan)

        self.time_tracker.add(info.solvetime*1000.)

        return output, exitflag, self._prev_trajectory

    def set_infeasible(self, output):
        self._mpc_feasible = False
        output["v"] = 0.
        output["a"] = 0.
        output["w"] = 0.

    def get_braking_trajectory(self, state):
        x = state[0]
        y = state[1]
        psi = state[2]
        v = state[3]
        spline = state[4]

        result = np.zeros((5, self._N))
        result[:, 0] = np.array([x, y, psi, v, spline])
        for k in range(1, self._N):
            x += v * self._dt * math.cos(psi)
            y += v * self._dt * math.sin(psi)
            spline += v * self._dt
            v -= self._braking_acceleration * self._dt
            v = np.max([v, 0.])
            result[:, k] = np.array([x, y, psi, v, spline])
        return result


    def set_projection(self, projection_func):
        self._projection_func = projection_func

    def get_solver(self):
        return self._solver

    def get_simulator(self):
        return self._simulator

    def get_model(self):
        return self._model

    def get_initial_guess(self):
        return self._u_traj_init, self._x_traj_init

    def print_stats(self):
        self.time_tracker.print_stats()