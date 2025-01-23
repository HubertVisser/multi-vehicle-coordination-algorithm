import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

# Example data points
s = np.array([0, 1, 2, 3, 4])
x = np.array([0, 1, 0, -1, 0])

# Create a cubic spline interpolation
cs_x = CubicSpline(s, x)

# Generate values for plotting the spline
s_vals = np.linspace(0, 4, 100)
x_vals = cs_x(s_vals)

# Plot the original data points and the spline
plt.plot(s, x, 'o', label='Data points')
plt.plot(s_vals, x_vals, label='Cubic spline')
plt.xlabel('s')
plt.ylabel('x')
plt.legend()
plt.title('Cubic Spline Interpolation')
plt.grid(True)
plt.show()