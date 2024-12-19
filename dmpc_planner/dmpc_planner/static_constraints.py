import math
import rospy

from mpc_planner_standalone.srv import decomp, decompRequest

from solver_generator.util.logging import print_success, print_value, TimeTracker
from .timer import Timer

class StaticConstraints:

    def __init__(self, settings) -> None:
        self._settings = settings
        self._N = self._settings["N"]
        self._max_constraints = self._settings["decomp"]["max_constraints"]

        self.timing = TimeTracker("static_constraints")

        self.get_decomp_constraints = rospy.ServiceProxy('/compute_constraints', decomp)
        print_value("StaticConstraints", 'waiting for service to start')
        rospy.wait_for_service('/compute_constraints')
        print_success("StaticConstraints ready")

    def call(self, state, model, params, is_feasible):

        # timer = Timer("static constraints")
        for k in [0]:#[0, self._N]:
            for i in range(self._max_constraints):
                params.set(k, f"disc_0_decomp_{i}_a1", 1.)
                params.set(k, f"disc_0_decomp_{i}_a2", 0.)
                params.set(k, f"disc_0_decomp_{i}_b", state[0] + 100.)

        all_x = []
        all_y = []
        request = decompRequest()
        for k in range(0, self._N):
            if is_feasible:
                all_x.append(model.get(k, "x"))
                all_y.append(model.get(k, "y"))

                # request.x = [model.get(k-1, "x"), model.get(k, "x")]
                # request.y = [model.get(k-1, "y"), model.get(k, "y")]
            else:
                all_x.append(state[0] + 0.01 * float(k) * math.cos(state[2]))
                all_y.append(state[1]+ 0.01 * float(k) * math.sin(state[2]))

                # request.x = [state[0], state[0] + 0.1]
                # request.y = [state[1], state[1] + 0.1]

        request.first = True
        request.last = True
        request.x = all_x
        request.y = all_y

        response = self.get_decomp_constraints(request.x, request.y, request.first, request.last)

        num_constraints = int(len(response.constraint_a1) / (self._N - 1))
        for k in range(1, self._N+1):
            start = (k-1) * num_constraints
            if k == self._N:
                start -= num_constraints # Reuse the final constraints
                
            for i in range(num_constraints):
                params.set(k, f"disc_0_decomp_{i}_a1", response.constraint_a1[start + i])
                params.set(k, f"disc_0_decomp_{i}_a2", response.constraint_a2[start + i])
                params.set(k, f"disc_0_decomp_{i}_b", response.constraint_b[start + i])
        # timer.stop_and_print()


    def print_stats(self):
        pass
        # self.timing.print_stats()