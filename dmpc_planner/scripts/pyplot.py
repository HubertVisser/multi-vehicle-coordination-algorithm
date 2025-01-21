import matplotlib 
matplotlib.use('Agg')
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