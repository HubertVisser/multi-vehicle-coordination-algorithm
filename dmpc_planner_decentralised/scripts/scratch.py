import os, sys
sys.path.append(os.path.join(sys.path[0], "..", "..", "solver_generator"))

from visualization_msgs.msg import Marker
from geometry_msgs.msg import Pose, Quaternion
import tf.transformations as tf
import numpy as np
from scipy.optimize import linprog
from util.math import get_A, get_b

import numpy as np
from itertools import combinations
from scipy.optimize import linprog

def find_intersection_points(A, b):
    n = A.shape[0]
    intersection_points = []

    # Generate all combinations of half-spaces
    for comb in combinations(range(n), 2):
        A_comb = A[list(comb), :]
        b_comb = b[list(comb)]

        # Solve the system of linear equations A_comb x = b_comb
        try:
            x = np.linalg.solve(A_comb, b_comb)
        except np.linalg.LinAlgError:
            continue  # Skip if the system is singular

        # Check if the intersection point satisfies all inequalities
        if np.all(A @ x <= b):
            intersection_points.append(x)

    if len(intersection_points) > 1:
        intersection_points[-1], intersection_points[-2] = intersection_points[-2], intersection_points[-1]
    
    return np.array(intersection_points)

# Define the matrix A and vector b
theta = np.pi / 2
pos = [0, 0]
A = get_A(theta)
b = get_b(pos, theta, 2, 2)

# Find all intersection points
intersection_points = find_intersection_points(A, b)

print("Intersection points:")
print(intersection_points)