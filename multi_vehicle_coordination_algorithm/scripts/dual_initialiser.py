import casadi as cd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import minimize

from helpers import get_robot_pairs_one

class DualProgram:
    def __init__(self, length, width, d_min):
        self._l = length
        self._w = width
        self._d_min = d_min

    def update_parameters(self, x_i, x_j):
        self._xy_i = x_i[0:2]
        self._theta_i = x_i[2]

        self._xy_j = x_j[0:2]
        self._theta_j = x_j[2]

        self._A_i = np.array([
            [np.cos(self._theta_i), np.sin(self._theta_i)],
            [-np.sin(self._theta_i), np.cos(self._theta_i)],
            [-np.cos(self._theta_i), -np.sin(self._theta_i)],
            [np.sin(self._theta_i), -np.cos(self._theta_i)]
        ])
        self._A_j = np.array([
            [np.cos(self._theta_j), np.sin(self._theta_j)],
            [-np.sin(self._theta_j), np.cos(self._theta_j)],
            [-np.cos(self._theta_j), -np.sin(self._theta_j)],
            [np.sin(self._theta_j), -np.cos(self._theta_j)]
        ])

        self._lw = np.array([self._l/2, self._w/2, self._l/2, self._w/2])
        self._b_i = self._lw + self._A_i @ self._xy_i
        self._b_j = self._lw + self._A_j @ self._xy_j

    def objective(self, x):
        return 0
    
    # Constraint 6a
    def constraint_dmin(self, x):
        lambda_ij = x[0:4]
        lambda_ji = x[4:8]
        return -np.dot(self._b_i, lambda_ij) - np.dot(self._b_j, lambda_ji) - self._d_min

    # Constraint 6b
    def constraint_s_ij(self, x):
        lambda_ij = x[0:4]
        s_ij = x[8:]
        return np.dot(self._A_i.T, lambda_ij) + s_ij

    # Constraint s_ij norm
    def constraint_s_norm(self, x):
        s_ij = x[8:]
        return 1 - np.linalg.norm(s_ij, ord=2)

    def solve(self):
        bounds = [(0, None)] * 8 + [(None, None)] * 2
        x0 = np.zeros(10)
        constraints = [
            {'type': 'ineq', 'fun': self.constraint_dmin},
            {'type': 'eq', 'fun': self.constraint_s_ij},
            {'type': 'ineq', 'fun': self.constraint_s_norm}
        ]
        result = minimize(self.objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        if result.success:
            lambda_ij = result.x[:4]
            lambda_ji = result.x[4:8]
            s_ij = result.x[8:]
            return {
                "success": True,
                "value": result.x
            }
        else:
            return {
                "success": False,
                "message": result.message
            }
        
        
def set_initial_x_plan(settings, xinit):
    
    _N = settings["N"]
    reference_velocity = settings["weights"]["reference_velocity"]
    int_step = settings["integrator_step"]

    # assign initial guess for the states by forward euler integration on the reference path
    # refinement for first guess needs to be higher because the forward euler is a bit lame
    N_0 = _N

    s_0_vec = np.linspace(0, 0 + reference_velocity * 1.5, N_0+1)
    x_ref_0 = np.ones(N_0+1) * xinit[0]
    y_ref_0 = np.ones(N_0+1) * xinit[1]
    theta_ref_0 = np.ones(N_0+1) * xinit[2]

    for i in range(1,N_0+1):
        x_ref_0[i] = x_ref_0[i-1] + reference_velocity * int_step * np.cos(theta_ref_0[i-1])
        y_ref_0[i] = y_ref_0[i-1] + reference_velocity * int_step * np.sin(theta_ref_0[i-1])
        # theta_ref_0[i] = theta_ref_0[i-1] + k_0_vals[i-1] * reference_velocity * int_step
    
    # now down sample to the N points
    x = np.interp(np.linspace(0,1, _N), np.linspace(0,1,N_0+1), x_ref_0)
    y = np.interp(np.linspace(0,1, _N), np.linspace(0,1,N_0+1), y_ref_0)
    theta_ref_0 = np.interp(np.linspace(0,1, _N), np.linspace(0,1,N_0+1), theta_ref_0)
    return np.array([x, y, theta_ref_0])


def dual_initialiser(settings, i, j):
    _N = settings["N"]
    reference_velocity = settings["weights"]["reference_velocity"]
    int_step = settings["integrator_step"]
    length = settings["polytopic"]["length"]
    width = settings["polytopic"]["width"]
    d_min = settings["polytopic"]["d_min"]
    x_i = np.array([settings[f"robot_{i}"]["start_x"], 
                    settings[f"robot_{i}"]["start_y"],
                    settings[f"robot_{i}"]["start_theta"] * np.pi,
                    ])
    x_j = np.array([settings[f"robot_{j}"]["start_x"], 
                    settings[f"robot_{j}"]["start_y"],
                    settings[f"robot_{j}"]["start_theta"] * np.pi,
                    ])

    x_plan_i = set_initial_x_plan(settings, x_i)
    x_plan_j = set_initial_x_plan(settings, x_j)

    duals = np.zeros((10, _N))

    initial_guesser = DualProgram(length=length, width=width, d_min=d_min)

    for i in range(_N):
        x_i = x_plan_i[:, i]
        x_j = x_plan_j[:, i]
        initial_guesser.update_parameters(x_i, x_j)
        dual = initial_guesser.solve()
        if dual["success"]:
            duals[:, i] = dual["value"]
        else:
            print(f"No solution found for step {i}: {dual['message']}")
    
    return duals

def get_all_initial_duals(settings):
    """
    Returns a dictionary of duals for each unique robot pair (i, j) with i < j.
    The key is a string 'i_j'.
    """
    num_robots = settings["number_of_robots"]
    duals = get_robot_pairs_one(num_robots)
    for i in range(1, num_robots + 1):
        for j in range(i + 1, num_robots + 1):
            duals.get(f"{i}_{j}") = dual_initialiser(settings, i, j)
    return duals


def main():
    N=4
    xinit_1 = np.array([-5, 0, 0])
    xinit_2 = np.array([1, -4, np.pi/2])
    x_plan_1 = set_initial_x_plan(xinit_1)
    x_plan_2 = set_initial_x_plan(xinit_2)

    h = 0.2
    w = 0.2
    d_min = 0.1

    results = []

    initial_guesser = DualProgram(h, w, d_min)

    for i in range(N):
        x_i = x_plan_1[:3, i]
        x_j = x_plan_2[:3, i]
        initial_guesser.update_parameters(x_i, x_j)
        result = initial_guesser.solve()
        if result["success"]:
            results.append(result["value"])
        else:
            print(f"No solution found for step {i}: {result['message']}")
    
    return results

    
if __name__ == "__main__":
    main()
