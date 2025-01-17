import matplotlib.pyplot as plt
import numpy as np

def plot_x_traj_init(trajectory, N, dt):
        # Create a time array based on the number of steps and the time step
        time = np.arange(0, N * dt, dt)

        # Plot the trajectory
        plt.figure()
        for i in range(x_traj_init.shape[0]):
            plt.plot(time, x_traj_init[i, :], label=f'State {i}')

        plt.xlabel('Time [s]')
        plt.ylabel('State Value')
        plt.title('State Trajectory over Time')
        plt.legend()
        plt.grid(True)
        plt.show()