import os, sys

import pathlib
path = pathlib.Path(__file__).parent.resolve()
sys.path.append(os.path.join(path))
sys.path.append(os.path.join(sys.path[0], "..", "..", "solver_generator"))

acados_path = os.path.join(path, "..", "..", "mpc_planner_solver", "acados", "solver")

import numpy as np
import math

import generate_jetracer_solver
from solver_model import BicycleModel2ndOrderMultiRobot
from acados_template import AcadosModel
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver

from timer import Timer
from util.files import solver_path, load_settings

from util.logging import print_warning, print_value, print_success, TimeTracker, print_header
from util.realtime_parameters import AcadosRealTimeModel


class MPCPlanner:

    def __init__(self, settings):
        self._settings = settings
        self._N = self._settings["N"]
        self._dt = self._settings["integrator_step"]
        self._braking_acceleration = self._settings["braking_acceleration"]
        self._number_of_robots = self._settings["number_of_robots"]
        self._dart_simulator = self._settings["dart_simulator"]

        self._projection_func = lambda trajectory: trajectory # No projection

        self.init_acados_solver()

        self._mpc_feasible = True
        self.time_tracker = TimeTracker(self._settings["solver_settings"])

        print_header("Starting MPC with DART simulator") if self._dart_simulator else print_header("Starting MPC without simulator")

    def init_acados_solver(self):
        # The generation software
        if hasattr(self, "_solver"):
            del self._solver, self._simulator, self._solver_settings
        if hasattr(self, "_mpc_x_plan"):
            del self._mpc_x_plan, self._mpc_u_plan

        self._solver, self._simulator = generate_jetracer_solver.generate()
        self._solver_settings = load_settings("solver_settings", package="mpc_planner_solver")

        # acados_ocp = AcadosOcp(acados_path=acados_path)
        # self._solver = AcadosOcpSolver(acados_ocp)
        self._model = AcadosRealTimeModel(self._settings, self._solver_settings, package="mpc_planner_solver")
        self._dynamic_model = BicycleModel2ndOrderMultiRobot(self._number_of_robots)
        self.reference_velocity = self._settings["weights"]["reference_velocity"]

        self._nx = self._solver_settings["nx"]
        self._nu = self._solver_settings["nu"]
        self._nvar = self._solver_settings["nvar"]
        self._nx_one_robot = self._nx // self._number_of_robots

        self._prev_trajectory = np.zeros((self._N, self._nvar)) 

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
            self.set_initial_x_plan_1(xinit)
            self.set_initial_x_plan_2(xinit) if self._number_of_robots > 1 else None

        if not hasattr(self, "_mpc_u_plan"):
            self._mpc_u_plan = np.zeros((self._nu, self._N))
            self.set_initial_u_plan()

        self._u_traj_init = self._mpc_u_plan
        self._x_traj_init =  self._mpc_x_plan

        if self._mpc_feasible:
            pass
            # Shifted
            # self._x_traj_init = np.concatenate((self._mpc_x_plan[:, 1:], self._mpc_x_plan[:, -1:]), axis=1)
            # self._u_traj_init = np.concatenate((self._mpc_u_plan[:, 1:], self._mpc_u_plan[:, -1:]), axis=1)            

            # Not shifted
        else:
            pass
            # Brake (model specific)
            # self._x_traj_init = self.get_braking_trajectory(xinit)
            # self._x_traj_init = self._projection_func(self._x_traj_init)

            # Xinit everywhere (could be infeasible)
            self._x_traj_init = np.tile(np.array(xinit).reshape((-1, 1)), (1, self._N))
            self._u_traj_init = np.zeros((self._nu, self._N))
            self._solver.reset(reset_qp_solver_mem=1)
            self._solver.options_set('warm_start_first_qp', False)

   
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
                # print(f"u_{k}: {self._u_traj_init[:, k]}")
                self._solver.set(k, 'p', np.array(p[k*npar:(k+1)*npar])) # params for the current stage

            self._solver.set(self._N, 'p', np.array(p[self._N*npar:(self._N + 1)*npar])) # Repeats the final set of parameters

            solve_time = 0.
            for it in range(self._settings["solver_settings"]["iterations"]):
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
            for n in range(1, self._number_of_robots+1):

                output[f"x_{n}"] = self._model.get(1, f"x_{n}")
                output[f"y_{n}"] = self._model.get(1, f"y_{n}")
                output[f"theta_{n}"] = self._model.get(1, f"theta_{n}")
                output[f"vx_{n}"] = self._model.get(1, f"vx_{n}")
                output[f"vy_{n}"] = self._model.get(1, f"vy_{n}")
                output[f"w_{n}"] = self._model.get(1, f"w_{n}")
                output[f"s_{n}"] = self._model.get(1, f"s_{n}")
                
                output[f"throttle_{n}"] = self._model.get(0, f"throttle_{n}")
                output[f"steering_{n}"] = self._model.get(0, f"steering_{n}")
                for j in range(1, self._number_of_robots+1):
                    if j != n:
                        output[f"lam_{n}_{j}"] = self._model.get(0, f"lam_{n}_{j}")
                        output[f"s_dual_{n}_{j}"] = self._model.get(0, f"s_dual_{n}_{j}")
            
            

            self.time_tracker.add(solve_time)

            print_value("Current cost", f"{self.get_cost_acados():.2f}")
            self._prev_trajectory = self._model.get_trajectory(self._solver, self._mpc_x_plan, self._mpc_u_plan)
        except AttributeError:
            output = dict()
            self.set_infeasible(output)
            print("Solver is being regenerated...")
            return output, False, []

        return output, True, self._prev_trajectory

    def set_initial_x_plan_1(self, xinit):
        # x = x y theta vx vy w s
        #     2 3   4   5  6  7 8

        # assign initial guess for the states by forward euler integration on the reference path
        # refinement for first guess needs to be higher because the forward euler is a bit lame
        N_0 = 1000

        s_0_vec = np.linspace(0, 0 + self.reference_velocity * 1.5, N_0+1)
        x_ref_0 = np.ones(N_0+1) * xinit[0]
        y_ref_0 = np.ones(N_0+1) * xinit[1]
        theta_ref_0 = np.ones(N_0+1) * xinit[2]

        for i in range(1,N_0+1):
            x_ref_0[i] = x_ref_0[i-1] + self.reference_velocity * self._dt * np.cos(theta_ref_0[i-1])
            y_ref_0[i] = y_ref_0[i-1] + self.reference_velocity * self._dt * np.sin(theta_ref_0[i-1])
            # theta_ref_0[i] = theta_ref_0[i-1] + k_0_vals[i-1] * self.reference_velocity * self._dt
        
        # now down sample to the N points
        self._mpc_x_plan[0,:] = np.interp(np.linspace(0,1,self._N), np.linspace(0,1,N_0+1), x_ref_0)
        self._mpc_x_plan[1,:] = np.interp(np.linspace(0,1,self._N), np.linspace(0,1,N_0+1), y_ref_0)
        self._mpc_x_plan[3,:] = self.reference_velocity
        self._mpc_x_plan[6,:] = np.interp(np.linspace(0,1,self._N), np.linspace(0,1,N_0+1), s_0_vec) 
        
    def set_initial_x_plan_2(self, xinit):
        # x = x y theta vx vy w s
        #     2 3   4   5  6  7 8

        # assign initial guess for the states by forward euler integration on the reference path
        # refinement for first guess needs to be higher because the forward euler is a bit lame
        N_0 = 1000

        s_0_vec = np.linspace(0, 0 + self.reference_velocity * 1.5, N_0+1)
        x_ref_0 = np.ones(N_0+1) * xinit[0 + self._nx_one_robot]
        y_ref_0 = np.ones(N_0+1) * xinit[1 + self._nx_one_robot]
        theta_ref_0 = np.ones(N_0+1) * xinit[2 + self._nx_one_robot]

        for i in range(1,N_0+1):
            x_ref_0[i] = x_ref_0[i-1] + self.reference_velocity * self._dt * np.cos(theta_ref_0[i-1])
            y_ref_0[i] = y_ref_0[i-1] + self.reference_velocity * self._dt * np.sin(theta_ref_0[i-1])
            # theta_ref_0[i] = theta_ref_0[i-1] + k_0_vals[i-1] * self.reference_velocity * self._dt
        
        # now down sample to the N points
        self._mpc_x_plan[0 + self._nx_one_robot,:] = np.interp(np.linspace(0,1,self._N), np.linspace(0,1,N_0+1), x_ref_0)
        self._mpc_x_plan[1 + self._nx_one_robot,:] = np.interp(np.linspace(0,1,self._N), np.linspace(0,1,N_0+1), y_ref_0)
        self._mpc_x_plan[3 + self._nx_one_robot,:] = self.reference_velocity
        self._mpc_x_plan[6 + self._nx_one_robot,:] = np.interp(np.linspace(0,1,self._N), np.linspace(0,1,N_0+1), s_0_vec)
    
    def set_initial_u_plan(self):
        # Evaluate throttle to keep the constant velocity
        throttle_search = np.linspace(0,1,30)
        mass_vehicle = self._dynamic_model.get_mass()
        fx = self._dynamic_model.fx_wheels(throttle_search, self.reference_velocity)
        acceleration_x = fx / mass_vehicle
        throttle_initial_guess = throttle_search[np.argmin(np.abs(acceleration_x))]
        self._mpc_u_plan[0, :] = throttle_initial_guess
        if self._number_of_robots == 2 : self._mpc_u_plan[2, :] = throttle_initial_guess

    def get_cost_acados(self):
        return self._solver.get_cost()

    def set_infeasible(self, output):
        self._mpc_feasible = False
        for n in range(1, self._number_of_robots+1):
            output[f"vx_{n}"] = 0.
            output[f"throttle_{n}"] = 0.
            output[f"steering_{n}"] = 0.

    def get_braking_trajectory(self, state):    # Not adjusted for multi robot
        x = state[0]
        y = state[1]
        theta = state[2]
        v = state[3]
        spline = state[4]

        result = np.zeros((5, self._N))
        result[:, 0] = np.array([x, y, theta, v, spline])
        for k in range(1, self._N):
            x += v * self._dt * math.cos(theta)
            y += v * self._dt * math.sin(theta)
            spline += v * self._dt
            v -= self._braking_acceleration * self._dt
            v = np.max([v, 0.])
            result[:, k] = np.array([x, y, theta, v, spline])
        return result


    def set_projection(self, projection_func):  # # Not adjusted for multi robot
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

    