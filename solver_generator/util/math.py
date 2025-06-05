import numpy as np
import casadi as cd
from itertools import combinations
import cvxpy as cp


def rotation_matrix_cd(angle):
    return cd.vertcat(cd.horzcat(cd.cos(angle), -cd.sin(angle)), cd.horzcat(cd.sin(angle), cd.cos(angle)))

def rotation_matrix_np(angle):
    return np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])

def haar_difference_without_abs(angle1, angle2):
    return cd.fmod(angle1 - angle2 + np.pi, 2 * np.pi) - np.pi


def huber_loss(value, quadratic_from, reference=0.0):
    """
    value: symbolic value
    reference: reference value
    delta: treshold to switch from quadratic to linear
    """
    # Calculate the residual
    residual = value - reference

    # Define the conditions for Huber loss
    is_small_error = cd.fabs(residual) <= quadratic_from
    small_error_loss = 0.5 * residual**2
    large_error_loss = quadratic_from * (cd.fabs(residual) - 0.5 * quadratic_from)

    # Use CasADi's conditional operation
    return cd.if_else(is_small_error, small_error_loss, large_error_loss)


def sigmoid(x, a, b):
    return 1 / (1 + np.exp(-a * (x - b)))


# `a` is the steepness of the descend ~10
# `b` is the transition point between the two functions
def blended_function(x, a, b, f1, f2):
    return (1 - sigmoid(x, a, b)) * f1(x) + sigmoid(x, a, b) * f2(x)


### --- Calulation polytopic constraints --- ###

def get_A(theta):
    if isinstance(theta, (int, float, np.number)):
        rot_mat = rotation_matrix_np(theta)
        return np.vstack([rot_mat.T, -rot_mat.T])
    elif isinstance(theta, (cd.SX, cd.MX)):
        rot_mat = rotation_matrix_cd(theta)
        return cd.vertcat(rot_mat.T, -rot_mat.T)
    else:
        raise ValueError("theta must be a number or a CasADi variable")

def get_b(position, theta, length, width):
    if isinstance(position, (np.ndarray, list)):
        assert len(position) == 2, "position must be a 2-element list or NumPy array [x, y]"
        dim_vector = np.array([length/2, width/2, length/2, width/2])
        _A = get_A(theta)
        return dim_vector + _A @ position
    elif isinstance(position, (cd.SX, cd.MX, cd.DM)):
        assert position.size1() == 2 and position.size2() == 1, "position must be a 2 by 1 CasADi vector"
        dim_vector = cd.DM([length/2, width/2, length/2, width/2])
        _A = get_A(theta)
        return dim_vector + _A @ position
    else:
        raise ValueError("position must be a NumPy array, list, or a CasADi variable")
    
def min_distance_polytopes(pose1, pose2, length, width):
    """
    Calculates the minimum distance between two polytopes defined by their position and orientation.

    args:
    poses: list of tuples (x, y, orientation) for each polytope
    """
    
    A1 = get_A(pose1[2])
    A2 = get_A(pose2[2])
    b1 = get_b([pose1[0],pose1[1]], pose1[2], length, width)
    b2 = get_b([pose2[0],pose2[1]], pose2[2], length, width)

    dim = 2
    # Define optimization variables
    x = cp.Variable(dim)
    y = cp.Variable(dim)

    # Define the objective function: minimize ||x - y||_2^2
    objective = cp.Minimize(cp.norm(x - y, 2))

    # Define the constraints
    constraints = [
        A1 @ x <= b1,
        A2 @ y <= b2
    ]

    # Define and solver the problem
    problem = cp.Problem(objective, constraints)
    problem.solve()

    # Check if the problem was solved successfully
    if problem.status not in ["infeasible", "unbounded"]:
        return x.value, y.value, problem.value  # Return x, y, and the minimum distance
    else:
        raise ValueError("Optimization problem is infeasible or unbounded.")

