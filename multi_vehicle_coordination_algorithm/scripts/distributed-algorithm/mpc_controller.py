import os, sys
import rospy

import pathlib
path = pathlib.Path(__file__).parent.resolve()
sys.path.append(os.path.join(path))
sys.path.append(os.path.join(sys.path[0], "..", "..", "solver_generator"))

import numpy as np
import math

import generate_NMPC_solver
import generate_CA_solver
from solver_model import BicycleModel2ndOrder
from acados_template import AcadosModel
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver

from timer import Timer
from util.files import solver_path, load_settings

from util.logging import print_warning, print_value, print_success, TimeTracker, print_header
from util.realtime_parameters import AcadosRealTimeModel


class MPCPlanner:

    def __init__(self, settings, idx):
        self._settings = settings
        self._N = self._settings["N"]
        self._dt = self._settings["integrator_step"]
        self._braking_acceleration = self._settings["braking_acceleration"]
        self._number_of_robots = self._settings["number_of_robots"]
        self._dart_simulator = self._settings["dart_simulator"]
        self._idx = idx

        self.init_nmpc_solver()
        self.init_ca_solver()

        self._mpc_feasible = True
        self.time_tracker = TimeTracker(self._settings["solver_settings"])

        print_header(f"Starting MPC {idx}")

    def init_nmpc_solver(self):
        # The generation software
        if hasattr(self, "_solver_nmpc"):
            del self._solver_nmpc, self._solver_settings_nmpc
        if hasattr(self, "_mpc_x_plan"):
            del self._mpc_x_plan, self._mpc_u_plan

        self._solver_nmpc, _ = generate_NMPC_solver.generate(self._idx)
        print_success(f"NMPC {self._idx} solver generated")

        self._solver_settings_nmpc = load_settings(f"solver_settings_nmpc_{self._idx}", package="mpc_planner_solver")
        self._model_nmpc = AcadosRealTimeModel(self._settings, self._solver_settings_nmpc, model_map_name=f"model_map_nmpc_{self._idx}", package="mpc_planner_solver")
        self._dynamic_model = BicycleModel2ndOrder(self._idx, self._number_of_robots)
        self.reference_velocity = self._settings["weights"]["reference_velocity"]

        self._nx_nmpc = self._solver_settings_nmpc["nx"]
        self._nu_nmpc = self._solver_settings_nmpc["nu"]
        self._nvar_nmpc = self._solver_settings_nmpc["nvar"]

        self._prev_trajectory = np.zeros((self._N, self._nvar_nmpc)) 

        print_success(f"NMPC {self._idx} solver generated")
    
    def init_ca_solver(self):
        # The generation software
        if hasattr(self, "_solver_ca"):
            del self._solver_ca, self._solver_settings_ca

        self._solver_ca, _ = generate_CA_solver.generate(self._idx)
        print_success(f"CA {self._idx} solver generated")

        self._solver_settings_ca = load_settings(f"solver_settings_ca_{self._idx}", package="mpc_planner_solver")
        self._model_ca = AcadosRealTimeModel(self._settings, self._solver_settings_ca, model_map_name=f"model_map_ca_{self._idx}", package="mpc_planner_solver")

        self._nx_ca = self._solver_settings_ca["nx"]
        self._nu_ca = self._solver_settings_ca["nu"]
        self._nlam_ca = self._solver_settings_ca["nlam"]
        self._nvar_ca = self._solver_settings_ca["nvar"]

        self._prev_solution_ca = np.zeros((self._N, self._nvar_ca)) 

        print_success(f"CA {self._idx} solver generated")

    def solve_nmpc(self, xinit, p):
        if not hasattr(self, "_solver_nmpc"):
            output = dict()
            self.set_infeasible(output)
            print("Solver is being regenerated...")
            return output, False, None

        # Initialize the initial guesses
        if not hasattr(self, "_mpc_x_plan"):
            self._mpc_x_plan = np.tile(np.array(xinit).reshape((-1, 1)), (1, self._N))


        if not hasattr(self, "_mpc_u_plan"):
            self._mpc_u_plan = np.zeros((self._nu_nmpc, self._N))
            self.set_initial_u_plan()

        if self._mpc_feasible:

            self._x_traj_init = self._mpc_x_plan
            self._u_traj_init = self._mpc_u_plan

        else:

            # Brake (model specific)
            self._x_traj_init = self.get_braking_trajectory(xinit)
            self._u_traj_init = np.zeros((self._nu_nmpc, self._N))

            self._solver_nmpc.options_set('warm_start_first_qp', False)

        return self.solve_acados_nmpc(xinit, p)

    def solve_acados_nmpc(self, xinit, p):
        try:
            # Set initial state
            self._solver_nmpc.constraints_set(0, 'lbx', np.array(xinit))
            self._solver_nmpc.constraints_set(0, 'ubx', np.array(xinit))
            
            npar = int(len(p) / (self._N))
            for k in range(0, self._N):
                self._solver_nmpc.set(k, 'x', self._x_traj_init[:, k])
                self._solver_nmpc.set(k, 'u', self._u_traj_init[:, k]) 
                self._solver_nmpc.set(k, 'p', np.array(p[k*npar:(k+1)*npar])) # params for the current stage

            self._solver_nmpc.set(self._N, 'p', np.array(p[(self._N - 1)*npar : (self._N)*npar])) # Repeats the final set of parameters

            solve_time = 0.
            # for it in range(self._settings["solver_settings"]["iterations_distributed"]):
            status = self._solver_nmpc.solve()
            solve_time += float(self._solver_nmpc.get_stats('time_tot')) * 1000.
           
            output = dict()
            if status != 0: # infeasible
                print_warning(f"Optimization for NMPC {self._idx} was infeasible (exitflag = {status})")
                
                output["status"] = status
                self.set_infeasible(output)
                return output, False, None

            self._mpc_feasible = True
            self._model_nmpc.load(self._solver_nmpc)

            output[f"x_{self._idx}"] = self._model_nmpc.get(1, f"x_{self._idx}")
            output[f"y_{self._idx}"] = self._model_nmpc.get(1, f"y_{self._idx}")
            output[f"theta_{self._idx}"] = self._model_nmpc.get(1, f"theta_{self._idx}")
            output[f"vx_{self._idx}"] = self._model_nmpc.get(1, f"vx_{self._idx}")
            output[f"vy_{self._idx}"] = self._model_nmpc.get(1, f"vy_{self._idx}")
            output[f"w_{self._idx}"] = self._model_nmpc.get(1, f"w_{self._idx}")
            output[f"s_{self._idx}"] = self._model_nmpc.get(1, f"s_{self._idx}")
            
            output[f"throttle"] = self._model_nmpc.get(0, f"throttle_{self._idx}")
            output[f"steering"] = self._model_nmpc.get(0, f"steering_{self._idx}")
            
            for j in range(1, self._number_of_robots+1):
                if j == self._idx:
                    continue
                output[f"lam_{self._idx}_{j}_0"] = self._model_nmpc.get(1, f"lam_{self._idx}_{j}_0")
                output[f"lam_{self._idx}_{j}_1"] = self._model_nmpc.get(1, f"lam_{self._idx}_{j}_1")
                output[f"lam_{self._idx}_{j}_2"] = self._model_nmpc.get(1, f"lam_{self._idx}_{j}_2")
                output[f"lam_{self._idx}_{j}_3"] = self._model_nmpc.get(1, f"lam_{self._idx}_{j}_3")
                
                output[f"lam_{j}_{self._idx}_0"] = self._model_nmpc.get(1, f"lam_{j}_{self._idx}_0")
                output[f"lam_{j}_{self._idx}_1"] = self._model_nmpc.get(1, f"lam_{j}_{self._idx}_1")
                output[f"lam_{j}_{self._idx}_2"] = self._model_nmpc.get(1, f"lam_{j}_{self._idx}_2")
                output[f"lam_{j}_{self._idx}_3"] = self._model_nmpc.get(1, f"lam_{j}_{self._idx}_3")

            self.time_tracker.add(solve_time)

            print_value(f"Current cost (nmpc {self._idx}):", f"{self.get_cost_nmpc():.2f}")
            self._prev_trajectory = self._model_nmpc.get_trajectory(self._solver_nmpc, self._mpc_x_plan, self._mpc_u_plan)
        except AttributeError:
            output = dict()
            self.set_infeasible(output)
            print("Solver is being regenerated...")
            return output, False, None

        return output, True, self._prev_trajectory
   
    def solve_ca(self, uinit, p): 

        if not hasattr(self, "_x_init_ca"):
            self._x_init_ca = np.zeros((self._nx_ca, self._N))
        if not hasattr(self, "_u_init_ca"):
            self._u_init_ca = np.tile(np.array(uinit).reshape((-1, 1)), (1, self._N))

        try:
            # Set initial state
            self._solver_ca.constraints_set(0, 'lbx', 0)
            self._solver_ca.constraints_set(0, 'ubx', 0)

            npar = int(len(p) / (self._N))
            for k in range(0, self._N):
                self._solver_ca.set(k, 'x', self._x_init_ca[:, k])
                self._solver_ca.set(k, 'u', self._u_init_ca[:, k]) 
                self._solver_ca.set(k, 'p', np.array(p[k*npar : (k+1)*npar])) # params for the current stage

            self._solver_ca.set(self._N, 'p', np.array(p[(self._N - 1)*npar : self._N*npar])) # Repeats the final set of parameters


            # for it in range(self._settings["solver_settings"]["iterations_distributed"]):
            status = self._solver_ca.solve()
           

            output = dict()
            if status != 0: # infeasible
                print_warning(f"Optimization for CA {self._idx} was infeasible (exitflag = {status})")
                
                return output, False, None

            self._model_ca.load(self._solver_ca)

            for j in range(1, self._number_of_robots+1):
                if j == self._idx:
                    continue
                output[f"lam_{self._idx}_{j}_0"] = self._model_ca.get(1, f"lam_{self._idx}_{j}_0")
                output[f"lam_{self._idx}_{j}_1"] = self._model_ca.get(1, f"lam_{self._idx}_{j}_1")
                output[f"lam_{self._idx}_{j}_2"] = self._model_ca.get(1, f"lam_{self._idx}_{j}_2")
                output[f"lam_{self._idx}_{j}_3"] = self._model_ca.get(1, f"lam_{self._idx}_{j}_3")
                
                output[f"lam_{j}_{self._idx}_0"] = self._model_ca.get(1, f"lam_{j}_{self._idx}_0")
                output[f"lam_{j}_{self._idx}_1"] = self._model_ca.get(1, f"lam_{j}_{self._idx}_1")
                output[f"lam_{j}_{self._idx}_2"] = self._model_ca.get(1, f"lam_{j}_{self._idx}_2")
                output[f"lam_{j}_{self._idx}_3"] = self._model_ca.get(1, f"lam_{j}_{self._idx}_3")

                if self._idx > j:
                    output[f"s_{j}_{self._idx}_0"] = self._model_ca.get(1, f"s_{j}_{self._idx}_0")
                    output[f"s_{j}_{self._idx}_1"] = self._model_ca.get(1, f"s_{j}_{self._idx}_1")
                else:    
                    output[f"s_{self._idx}_{j}_0"] = self._model_ca.get(1, f"s_{self._idx}_{j}_0")
                    output[f"s_{self._idx}_{j}_1"] = self._model_ca.get(1, f"s_{self._idx}_{j}_1")
                        
            
            

            print_value(f"Current cost (ca {self._idx}):", f"{self.get_cost_ca():.2f}")
            self._prev_solution_ca = self._model_ca.get_solution_ca(self._solver_ca, self._x_init_ca, self._u_init_ca)
        except AttributeError:
            output = dict()
            print("Solver is being regenerated...")
            return output, False, None

        return output, True, self._prev_solution_ca
    
    def set_initial_u_plan(self):
        # Evaluate throttle to keep the constant velocity
        throttle_search = np.linspace(0,1,30)
        mass_vehicle = self._dynamic_model.get_mass()
        fx = self._dynamic_model.fx_wheels(throttle_search, self.reference_velocity)
        acceleration_x = fx / mass_vehicle
        throttle_initial_guess = throttle_search[np.argmin(np.abs(acceleration_x))]
        self._mpc_u_plan[0, :] = throttle_initial_guess

    def get_braking_trajectory(self, state):    # Not adjusted for multi robot
        x = state[0]
        y = state[1]
        theta = state[2]
        vx = state[3]
        vy = state[4]
        w = state[5]
        spline = state[6]

        result = np.zeros((self._nx_nmpc, self._N))
        result[:, 0] = np.array([x, y, theta, vx, vy, w, spline])
        for k in range(1, self._N):
            x += vx * self._dt * math.cos(theta)
            y += vx * self._dt * math.sin(theta)
            spline += vx * self._dt
            vx -= self._braking_acceleration * self._dt
            vx = np.max([vx, 0.])
            result[:, k] = np.array([x, y, theta, vx, vy, w, spline])
        return result

    def get_cost_nmpc(self):
        return self._solver_nmpc.get_cost()
    
    def get_cost_ca(self):
        return self._solver_ca.get_cost()

    def set_infeasible(self, output):
        self._mpc_feasible = False
        output[f"vx_{self._idx}"] = 0.
        output[f"throttle_{self._idx}"] = 0.
        output[f"steering_{self._idx}"] = 0.

    def get_solver_nmpc(self):
        return self._solver_nmpc
    
    def get_solver(self):
        return self._solver_ca

    def get_simulator(self):
        return self._simulator

    def get_model_nmpc(self):
        return self._model_nmpc
    
    def get_model_ca(self):
        return self._model_ca

    def get_initial_guess(self):
        return self._u_traj_init, self._x_traj_init

    def print_stats(self):
        self.time_tracker.print_stats()
    
