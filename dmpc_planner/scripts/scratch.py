import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
from util.math import rotation_matrix

x = 1
y = 0.5
pos = np.array([x,y])
assert pos.shape == (2,)

theta = 1/3*np.pi
rot_mat = rotation_matrix(theta)
A = np.vstack([rot_mat.T, -rot_mat.T])
assert A.shape == (4, 2)
dim_vector = np.array([1, 1, 1, 1])
assert dim_vector.shape == (4,)

b = dim_vector * 2
print(b)


