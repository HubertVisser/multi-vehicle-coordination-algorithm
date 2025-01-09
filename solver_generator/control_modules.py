

from util.logging import print_value, print_header


class ModuleManager:
    """
    Modules can bundle objectives and constraints
    """

    def __init__(self):
        self.modules = []

    def add_module(self, module):
        self.modules.append(module)
        return module

    def inequality_constraints(self, z, param, settings, model):
        return self.constraint_manager.inequality(z, param, settings, model)

    def number_of_constraints(self):
        return self.constraint_manager.nh

    def get_last_added_module(self):
        return self.modules[-1]

    def __str__(self):
        result = "--- MPC Modules ---\n"
        for module in self.modules:
            result += str(module) + "\n"
        return result

    def print(self):
        print_header("MPC Modules")
        for module in self.modules:
            print_value(module.module_name, str(module), tab=True)


class Module:

    def __init__(self):
        self.module_name = "UNDEFINED"
        self.description = ""

        self.submodules = []
        self.dependencies = []
        self.sources = []

    def write_to_solver_interface(self, header_file):
        return

    def __str__(self):
        result = self.description
        return result

    def add_definitions(self, header_file):
        pass


class ConstraintModule(Module):

    def __init__(self):
        super(ConstraintModule, self).__init__()
        self.type = "constraint"

        self.constraints = []

    def define_parameters(self, params):
        for constraint in self.constraints:
            constraint.define_parameters(params)


class ObjectiveModule(Module):

    def __init__(self):
        super(ObjectiveModule, self).__init__()
        self.type = "objective"
        self.objectives = []

    def define_parameters(self, params):
        for objective in self.objectives:
            objective.define_parameters(params)

    def get_value(self, model, params, settings, stage_idx):
        cost = 0.0
        for objective in self.objectives:
            cost += objective.get_value(model, params, settings, stage_idx)

        return cost


class Objective:

    def __init__(self) -> None:
        pass

    def define_parameters(self, params):
        raise IOError("Objective did not specify parameters")

    def get_value(self, model, params, settings, stage_idx) -> float:
        raise IOError("Objective did not return a cost")