import sys, os

sys.path.append(os.path.join(sys.path[0], "..", "solver_generator"))

from util.math import huber_loss
from control_modules import ObjectiveModule, Objective

class GoalObjective(Objective):

    """
        Objective to go to a 2D goal position
    """

    def __init__(self, settings):
        pass

    def define_parameters(self, params):
        params.add("goal", add_to_rqt_reconfigure=True, rqt_config_name=lambda p: f'["weights"]["goal"]')
        params.add("goal_x")
        params.add("goal_y")

    def get_value(self, model, params, settings, stage_idx):
        cost = 0.0

        pos_x = model.get("x")
        pos_y = model.get("y")
        # s = model.get("s")
        # goal_x = s
        # goal_y = 0

        goal_weight = params.get("goal")
        goal_x = params.get("goal_x")
        goal_y = params.get("goal_y")
        

        cost += goal_weight * ((pos_x - goal_x) ** 2 + (pos_y - goal_y) ** 2) / (goal_x**2 + goal_y**2 + 0.01)
        # cost += goal_weight * ((pos_x - goal_x) ** 2 ) / (goal_x**2 + 0.01)
        # dist_to_goal = (pos_x - goal_x) ** 2 + (pos_y - goal_y) ** 2
        # normalized_dist = dist_to_goal / (20.0)
        # cost += goal_weight * huber_loss(normalized_dist, quadratic_from=0.05)

        return cost


class GoalModule(ObjectiveModule):

    def __init__(self, settings):
        super().__init__()
        self.module_name = "GoalModule"  # Needs to correspond to the c++ name of the module
        self.import_name = "goal_module.h"
        self.description = "Tracks a goal in 2D"

        self.objectives.append(GoalObjective(settings))
