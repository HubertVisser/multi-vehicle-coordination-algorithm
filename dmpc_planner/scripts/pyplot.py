import os
os.environ["DISPLAY"] = "host.docker.internal:0"  # Set XQuartz display

import matplotlib 
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

def plot_x_traj(trajectory, N, dt):
        # Create a time array based on the number of steps and the time step
        time = np.linspace(0, (N-1) * dt, N)

        x_traj = trajectory
        # Plot the trajectory


        # Define labels for the states
        labels = ["Throttle", "Steering", "x", "y", "theta", "vx", "vy", "omega", "s"]

        # Clear the previous plot
        plt.close()

        # Plot the trajectory
        plt.figure()
        for i in range(x_traj.shape[0]):
            plt.plot(time, x_traj[i, :], label=labels[i])

        plt.xlabel('Time [s]')
        plt.ylabel('State Value')
        plt.title('State Trajectory over Time')
        plt.legend()
        plt.grid(True)
        plt.show()

def plot_splines(cs_x, cs_y, x, y):
    # Clear the previous plot
    plt.close()

    # Generate values for plotting the spline
    s_vals = np.linspace(0, 24, 1000)
    x_vals = cs_x(s_vals)
    y_vals = cs_y(s_vals)

    # Plot the spline path in the x-y plane
    plt.figure()
    plt.plot(x_vals, y_vals, linestyle='-', color='b')
    plt.plot(x, y, 'o', label='Data points')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Spline Path')
    plt.grid(True)
    plt.show()