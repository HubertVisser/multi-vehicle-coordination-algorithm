
"""
For python real-time scheme of the controller
"""
import numpy as np
import matplotlib.pyplot as plt
import os

from util.files import load_parameters, load_settings, planner_path

# Python real-time
class RealTimeParameters:

    def __init__(self, settings, parameter_map_name="parameter_map"):
        self._map = load_parameters(parameter_map_name=parameter_map_name)
        self._settings = settings

        self._num_p = self._map['num parameters']
        self._params = np.zeros((settings["N"], self._num_p))

    def set(self, k, parameter, value):
        if parameter in self._map.keys():
            self._params[k, self._map[parameter]] = value

    def get(self, k, parameter):
        return self._params[k, self._map[parameter]]

    def get_solver_params(self):
        out = []
        for k in range(self._settings["N"]):
            for i in range(self._num_p):
                out.append(self._params[k, i])
        return out

    def get_solver_params_for_stage(self, k):
        out = []
        for i in range(self._num_p):
            out.append(self._params[k, i])
        return out

    def get_num_par(self):
        return self._num_p
    
    def print(self):
        for k in range(self._params.shape[0]):
            print(f"--- {k} ---")
            for key, value in self._map.items():
                if key != "num parameters" and key.startswith(("lam_", "s_", "x_", "y_")):
                    print(f"{key}: {self._params[k, value]}")
    
    def write_to_file(self, filename):
        with open(filename, 'w') as f:
            for k in range(self._params.shape[0]):
                f.write(f"--- {k} ---\n")
                for key, value in self._map.items():
                    if key != "num parameters" and key.startswith(("lam_", "s_")):
                        f.write(f"{key}: {self._params[k, value]}\n")

    def check_for_nan(self):
        for k in range(self._params.shape[0]):
            for key, value in self._map.items():
                if key != "num parameters" and np.isnan(self._params[k, value]):
                    print(f"NaN detected for {key} at k = {k}: {self._params[k, value]}")
    
# Python real-time
class RealTimeModel:
    def __init__(self, settings, solver_settings, model_map_name="model_map", package=None):
        self._map = load_settings(model_map_name, package=package)
        self._settings = settings

        self._N = settings["N"]
        self._nu = solver_settings["nu"]
        self._nx = solver_settings["nx"]
        self._nvar = solver_settings["nvar"]
        self._vars = np.zeros((settings["N"], self._nvar))

    def get(self, k, var_name):
        map_value = self._map[var_name]
        return self._vars[k, map_value[1]]



# Python real-time
class AcadosRealTimeModel(RealTimeModel):

    def __init__(self, settings, solver_settings, model_map_name="model_map", package=None):
        super().__init__(settings, solver_settings, model_map_name, package)

    def load(self, solver):
        # Load the solver data into a numpy array
        for k in range(self._settings["N"]):
            for var in range(self._nx):
                self._vars[k, var] = solver.get(k, 'x')[var]
            for var in range(self._nx, self._nvar):
                self._vars[k, var] = solver.get(k, 'u')[var - self._nx]

    def get_trajectory(self, solver, mpc_x_plan, mpc_u_plan):
        # Retrieve the trajectory
        for k in range(0, self._N):
            mpc_x_plan[:, k] = solver.get(k, 'x')
            mpc_u_plan[:, k] = solver.get(k, 'u')

        return np.concatenate([mpc_x_plan,mpc_u_plan])
   
    def get_solution_ca(self, solver, x_init, u_init):
        # Retrieve the trajectory
        for k in range(0, self._N-1):
            x_init[:, k] = solver.get(k + 1, 'x')
            u_init[:, k] = solver.get(k + 1, 'u')

        # Last step
        x_init[:, -1] = solver.get(self._N - 1, 'x')
        u_init[:, -1] = solver.get(self._N - 1, 'u')

        return np.concatenate([x_init, u_init])
  