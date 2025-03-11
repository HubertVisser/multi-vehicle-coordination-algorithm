import casadi as cd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import minimize

class InitialGuessDuals:
    def __init__(self, i, theta_1, theta_2, xy_1, xy_2, h, w, d_min):
        self.i = i
        self.theta_1 = theta_1
        self.theta_2 = theta_2
        self.xy_1 = xy_1
        self.xy_2 = xy_2
        self.h = h
        self.w = w
        self.d_min = d_min

        self.A_1 = np.array([
            [np.cos(self.theta_1), np.sin(self.theta_1)],
            [-np.sin(self.theta_1), np.cos(self.theta_1)],
            [-np.cos(self.theta_1), -np.sin(self.theta_1)],
            [np.sin(self.theta_1), -np.cos(self.theta_1)]
        ])
        self.A_2 = np.array([
            [np.cos(self.theta_2), np.sin(self.theta_2)],
            [-np.sin(self.theta_2), np.cos(self.theta_2)],
            [-np.cos(self.theta_2), -np.sin(self.theta_2)],
            [np.sin(self.theta_2), -np.cos(self.theta_2)]
        ])

        self.hw = np.array([self.h/2, self.w/2, self.h/2, self.w/2])
        self.b_1 = self.hw + self.A_1 @ self.xy_1
        self.b_2 = self.hw + self.A_2 @ self.xy_2

    def objective(self, x):
        return 0

    def constraint1(self, x):
        lambda_ij = x[0:4]
        lambda_ji = x[4:8]
        if self.i == 1:
            return -np.dot(self.b_1, lambda_ij) - np.dot(self.b_2, lambda_ji) - self.d_min
        else:
            return -np.dot(self.b_2, lambda_ij) - np.dot(self.b_1, lambda_ji) - self.d_min

    def constraint2(self, x):
        lambda_ij = x[0:4]
        s_ij = x[8:]
        if self.i == 1:
            return np.dot(self.A_1.T, lambda_ij) + s_ij
        else:
            return np.dot(self.A_2.T, lambda_ij) + s_ij

    def constraint3(self, x):
        s_ij = x[8:]
        return 1 - np.linalg.norm(s_ij, ord=2)

    def solve(self):
        bounds = [(0, None)] * 8 + [(None, None)] * 2
        x0 = np.zeros(10)
        constraints = [
            {'type': 'ineq', 'fun': self.constraint1},
            {'type': 'eq', 'fun': self.constraint2},
            {'type': 'ineq', 'fun': self.constraint3}
        ]
        result = minimize(self.objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        if result.success:
            lambda_ij = result.x[:4]
            lambda_ji = result.x[4:8]
            s_ij = result.x[8:]
            return {
                "success": True,
                "lambda_ij": lambda_ij,
                "lambda_ji": lambda_ji,
                "s_ij": s_ij
            }
        else:
            return {
                "success": False,
                "message": result.message
            }

# Example usage
initialGuesser = InitialGuessDuals(
    i=1,
    theta_1=0,
    theta_2=np.pi/2,
    xy_1=np.array([-5, 0]),
    xy_2=np.array([1, -4]),
    h=0.2,
    w=0.2,
    d_min=0.1
)

result = initialGuesser.solve()
if result["success"]:
    print("robot", initialGuesser.i)
    print("Solution found:")
    print("lambda_ij:", result["lambda_ij"][2])
    print("lambda_ji:", result["lambda_ji"])
    print("s_ij:", result["s_ij"])
    print("b_1:", initialGuesser.b_1)
    print("b_2:", initialGuesser.b_2)
else:
    print("No solution found:", result["message"])
