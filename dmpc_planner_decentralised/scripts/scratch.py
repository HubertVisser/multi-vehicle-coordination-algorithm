import casadi as cd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import minimize

# Example values for A and b

theta_1 = np.pi/2
theta_2 = 0

A_1 =   np.array([
    [np.cos(theta_1), np.sin(theta_1)],
    [-np.sin(theta_1), np.cos(theta_1)],
    [-np.cos(theta_1), -np.sin(theta_1)],
    [np.sin(theta_1), -np.cos(theta_1)]
])
A_2 =   np.array([
    [np.cos(theta_2), np.sin(theta_2)],
    [-np.sin(theta_2), np.cos(theta_2)],
    [-np.cos(theta_2), -np.sin(theta_2)],
    [np.sin(theta_2), -np.cos(theta_2)]
])

# Define the vector [x, y]
xy_2 = np.array([1, 5])  # Example values for x and y
xy_1 = np.array([6, 1])  # Example values for x and y

# Define the vector [h/2, w/2, h/2, w/2]
h = 1  # Example value for h
w = 1  # Example value for w
hw = np.array([h/2, w/2, h/2, w/2])

# Calculate b(z)
b_1 = hw + A_1 @ xy_1
print("b(z):", b_1)
b_2 = hw + A_2 @ xy_2
print("b(z):", b_2)

d_min = 0.5

# Objective function to minimize (we can use a dummy objective since we are interested in constraints)
def objective(x):
    return 0

# Constraints
def constraint1(x):
    lambda_ij = x[0:4]      #np.ones((4)) * x[0]
    lambda_ji = x[4:8]      #np.ones((4)) * x[0]
    # return -np.dot(b_1, lambda_ij) - np.dot(b_2, lambda_ji) - d_min
    return -b_1.reshape(1,-1) @ lambda_ij - b_2.reshape(1,-1) @ lambda_ji - d_min

def constraint2(x):
    lambda_ij = x[0:4]      #np.ones((4)) * x[0]
    s_ij = x[8:]            #np.ones((2)) * x[1]
    # return np.dot(A_1.T, lambda_ij) + s_ij
    return A_1.T @lambda_ij + s_ij

def constraint3(x):
    s_ij = x[8:] #np.ones((2)) * x[1]
    return 1 - np.linalg.norm(s_ij, ord=2)

# Bounds for lambda (non-negative)
bounds = [(0, None)] * 8 + [(None, None)] * 2

# Initial guess
x0 = np.zeros(10)

# Define constraints in the form required by scipy.optimize.minimize
constraints = [
    {'type': 'ineq', 'fun': constraint1},
    {'type': 'eq', 'fun': constraint2},
    {'type': 'ineq', 'fun': constraint3}
]

# Solve the optimization problem
result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)

if result.success:
    lambda_ij = result.x[:4] #result.x[0]
    lambda_ji = result.x[4:8] #result.x[0]
    s_ij = result.x[8:]
    print("Solution found:")
    print("lambda_ij:", lambda_ij)
    print("lambda_ji:", lambda_ji)
    print("s_ij:", s_ij)
else:
    print("No solution found:", result.message)
