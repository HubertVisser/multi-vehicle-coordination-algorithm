import copy
import numpy as np
import os

import casadi as cd  # Acados

from util.files import write_to_yaml, parameter_map_path, load_parameters, rqt_config_path, get_current_package
from util.logging import print_value, print_header
from dynamic_reconfigure.parameter_generator_catkin import ParameterGenerator


class Parameters:

    def __init__(self):
        self._params = dict()

        self.parameter_bundles = dict()  # Used to generate function names in C++ with an integer parameter

        self.rqt_params = []
        self.rqt_param_config_names = []
        self.rqt_param_min_values = []
        self.rqt_param_max_values = []

        self._param_idx = 0
        self._p = None

    def add(
        self,
        parameter,
        add_to_rqt_reconfigure=False,
        rqt_config_name=lambda p: f'["weights"]["{p}"]',
        bundle_name=None,
        rqt_min_value=0.0,
        rqt_max_value=100.0,
    ):
        """
        Adds a parameter to the parameter dictionary.

        Args:
            parameter (Any): The parameter to be added.
            add_to_rqt_reconfigure (bool, optional): Whether to add the parameter to the RQT Reconfigure. Defaults to False.
            rqt_config_name (function, optional): A function that returns the name of the parameter in CONFIG for the parameter in RQT Reconfigure. Defaults to lambda p: f'["weights"]["{p}"]'.
        """

        if parameter in self._params.keys():
            return

        self._params[parameter] = copy.deepcopy(self._param_idx)
        if bundle_name is None:
            bundle_name = parameter

        if bundle_name not in self.parameter_bundles.keys():
            self.parameter_bundles[bundle_name] = [copy.deepcopy(self._param_idx)]
        else:
            self.parameter_bundles[bundle_name].append(copy.deepcopy(self._param_idx))

        self._param_idx += 1

        if add_to_rqt_reconfigure:
            self.rqt_params.append(parameter)
            self.rqt_param_config_names.append(rqt_config_name)
            self.rqt_param_min_values.append(rqt_min_value)
            self.rqt_param_max_values.append(rqt_max_value)

    def length(self):
        return self._param_idx

    def load(self, p):
        self._p = p

    def save_map(self):
        file_path = parameter_map_path()

        map = self._params
        map["num parameters"] = self._param_idx
        write_to_yaml(file_path, self._params)

    def get_p(self) -> float:
        return self._p

    def get(self, parameter):
        if self._p is None:
            print("Load parameters before requesting them!")

        return self._p[self._params[parameter]]

    def print(self):
        print_header("Parameters")
        print("----------")
        for param, idx in self._params.items():
            if param in self.rqt_params:
                print_value(f"{idx}", f"{param} (in rqt_reconfigure)", tab=True)
            else:
                print_value(f"{idx}", f"{param}", tab=True)
        print("----------")
    
    # TODO: use the following function to generate dynamic reconfigure cfg file (Hubert)
    def generate_dynamic_reconfigure_cfg(self):
        gen = ParameterGenerator()
        package_name = get_current_package()
        
        for param, min_val, max_val in zip(self.rqt_params, self.rqt_param_min_values, self.rqt_param_max_values):
            gen.add(param, double_t, 0, f"{param} parameter", 0.0, min_val, max_val)

        cfg_file_content = gen.generate(package_name, package_name, 'rqt_settings')
        cfg_file_path = os.path.join(rqt_config_path(), "rqt_settings.cfg")
        
        with open(cfg_file_path, 'w') as cfg_file:
            cfg_file.write(cfg_file_content)

        print(f"Generated dynamic reconfigure cfg file at: {cfg_file_path}")


class AcadosParameters(Parameters):

    def __init__(self):
        super().__init__()

    def load_global_parameters(self):
        self._global_params = load_parameters("parameter_map_global", package="mpc_planner_solver")
        self._param_idx = self._global_params["num parameters"]
        self._params.update(self._global_params)

    def load_acados_parameters(self):
       
        self._p = []
        for param in self._params.keys():
            par = cd.SX.sym(param, 1)
            self._p.append(par)

        self.load(self._p)

    def get_acados_parameters(self):
        result = None
        for param in self._params.keys():
            if result is None:
                result = self.get(param)
            else:
                result = cd.vertcat(result, self.get(param))

        return result

    def get_acados_p(self):
        return self._p

class GlobalParameters(Parameters):

    def __init__(self):
        super().__init__()

    def save_map(self):
        file_path = parameter_map_path(parameter_map_name="parameter_map_global")

        map = self._params
        map["num parameters"] = self._param_idx
        write_to_yaml(file_path, self._params)