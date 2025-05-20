import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
sys.path.append(os.path.join(sys.path[0], "..", "..", "solver_generator"))

from geometry_msgs.msg import PoseStamped, Pose
from nav_msgs.msg import Odometry, Path
from util.math import min_distance_polytopes
from util.logging import print_value



def plot_warmstart(planner): 
    warmstart_u, warmstart_x = planner._planner.get_initial_guess() 
    cylinder = planner._debug_visuals_pub.get_cylinder()
    cylinder.set_color(10, alpha=1.0)
    cylinder.set_scale(0.65, 0.65, 0.05)
    pose = Pose()
    for k in range(1, planner._N):
        pose.position.x = float(warmstart_x[0, k])
        pose.position.y = float(warmstart_x[1, k])
        cylinder.add_marker(deepcopy(pose))

def plot_path(planner):
    dist = 0.2
    if planner._path_msg is not None:
        line = planner._path_visual.get_line()
        line.set_scale(0.05)
        line.set_color(1, alpha=1.0)
        points = planner._path_visual.get_cube()
        points.set_color(3)
        points.set_scale(0.1, 0.1, 0.1)
        s = 0.0
        for i in range(50):
            a = planner._spline_fitter.evaluate(s)
            b = planner._spline_fitter.evaluate(s + dist)
            pose_a = Pose()
            pose_a.position.x = float(a[0])
            pose_a.position.y = float(a[1])
            points.add_marker(pose_a)
            pose_b = Pose()
            pose_b.position.x = float(b[0])
            pose_b.position.y = float(b[1])
            s += dist
            line.add_line_from_poses(pose_a, pose_b)
    planner._path_visual.publish()
    

def print_contouring_ref(planner):     # Not adjusted for multi robot
    s = planner._state[6]
    x, y = planner._spline_fitter.evaluate(s)
    print(f"Path at s = {s}: ({x}, {y})")
    print(f"State: ({planner._spline_fitter._closest_x}, {planner._spline_fitter._closest_y})")

def plot_states(planner):
    state_labels = ["x", "y", "theta", "vx", "vy", "omega", "s"]
    output_labels = ["throttle", "steering"]
    plt.figure(figsize=(12, 6))
    
    # Plot states
    plt.subplot(1, 2, 1)
    num_states = len(state_labels)
    for i in range(num_states):
        state_values = [state[i] for state in planner._states_save]
        plt.plot(state_values, label=state_labels[i])
    plt.xlabel('Time Step')
    plt.ylabel('State Values')
    plt.legend()
    plt.grid(True)
    plt.title(f'Robot {planner._idx} States')

    # Plot outputs
    plt.subplot(1, 2, 2)
    for i in range(len(output_labels)):
        output_values = [output[i] for output in planner._outputs_save]
        plt.plot(output_values, label=output_labels[i])
    plt.xlabel('Time Step')
    plt.ylabel('Output Values')
    plt.legend()
    plt.grid(True)
    plt.title(f'Robot {planner._idx} States {planner._scheme}')

    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), f'{planner._scheme}-algorithm/plots', f'states_{planner._idx}_{planner._scheme}.png'))  # Save the plot to a file
    plt.close()

def plot_dual_dicts_over_time(data_list, ylabel, title, filename, scheme, idx):
    """
    Plots each element of each key in a list of dicts over time.
    """
    if not data_list:
        return
    keys = data_list[0].keys()
    time_steps = range(len(data_list))
    num_elements = len(next(iter(data_list[0].values())))

    plt.figure(figsize=(12, 6))
    for element_index in range(num_elements):
        for key in keys:
            values = [d[key][element_index] for d in data_list]
            plt.plot(time_steps, values, label=f'{key}[{element_index}]')
    plt.xlabel('Time Step')
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.title(f'{title} - {scheme}')
    plt.tight_layout()
    plot_dir = os.path.join(os.path.dirname(__file__), f'{scheme}-algorithm/plots')
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir, f'{filename}_{idx}_{scheme}.png'))
    plt.close()

def plot_duals(lams, s_duals, scheme, idx=1):
    plot_dual_dicts_over_time(
        lams,
        ylabel='Lam Values',
        title='Lam Values',
        filename='lambda',
        scheme=scheme,
        idx=idx
    )
    plot_dual_dicts_over_time(
        s_duals,
        ylabel='s Values',
        title='s Values',
        filename='s',
        scheme=scheme,
        idx=idx
    )

def plot_pred_traj(planner):
    state_labels = ["x", "y", "theta", "vx", "vy", "omega", "s"]
    time = np.linspace(0, (planner._N-1) * planner._integrator_step, planner._N)

    plt.figure(figsize=(6, 12))
    
    # Plot states
    num_states = len(state_labels)
    trajectory_i = getattr(planner, f'_trajectory_{planner._idx}')
    plt.plot(time, trajectory_i.T)
    plt.legend(state_labels)

    plt.xlabel('Time Steps')
    plt.ylabel('State Values')
    plt.legend()
    plt.grid(True)
    plt.title(f'Robot {planner._idx} Predictions')

    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), 'plots', 'prediction_plot.png'))  # Save the plot to a file
    plt.close()

def plot_distance(poses_1, poses_2, length, width, scheme='centralised'):

    # Find the minimum length of the two lists
    min_length = min(len(poses_1), len(poses_2))
    poses_1 = poses_1[:min_length]
    poses_2 = poses_2[:min_length]

    assert len(poses_1) == len(poses_2), "The two lists must have the same length."

    distances = []
    for pose_1, pose_2 in zip(poses_1, poses_2):
        _, _, distance = min_distance_polytopes(pose_1, pose_2, length=length, width=width)
        distances.append(distance)

    # Convert distances to a NumPy array for easier indexing
    distances = np.array(distances)

    # Find the minimum distance and its index
    min_index = np.argmin(distances)
    min_distance = distances[min_index]
    
    # Mark the minimum distance
    plt.figure()
    plt.plot(distances, label='Distances')
    plt.scatter(min_index, min_distance, color="red", label="Minimum Distance")

    # Annotate the minimum distance
    plt.annotate(f"Min: {min_distance:.2f}",
                 xy=(min_index, min_distance),
                 xytext=(min_index + 1, min_distance + 0.5),  # Offset the text for better visibility
                 arrowprops=dict(facecolor="black", arrowstyle="->"),
                 fontsize=10)
    
    plt.xlabel("Index")
    plt.ylabel("Distance")
    plt.title(f"Distances Between Points - {scheme}")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(os.path.dirname(__file__), f'{scheme}-algorithm/plots', f'distance_{scheme}.png'))  # Save the plot to a file
    print_value("Minimum Distance", min_distance, tab=True)

def plot_trajectory(poses_1, poses_2, reference_1, reference_2, track_choice, scheme='centralised'):
    plt.figure()
    plt.plot(poses_1[:, 0], poses_1[:, 1], label='Robot 1', color='blue')
    plt.plot(poses_2[:, 0], poses_2[:, 1], label='Robot 2', color='red')
    if track_choice == 'straight_line':
        plt.plot(reference_1[:, 0], reference_1[:, 1], label='Reference Path', color='grey', alpha=0.5, linestyle='--')
    else:
        plt.plot(reference_1[:, 0], reference_1[:, 1], label='Reference Path 1', color='blue', alpha=0.5, linestyle='--')
        plt.plot(reference_2[:, 0], reference_2[:, 1], label='Reference Path 2', color='red', alpha=0.5, linestyle='--')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    # plt.title(f'Trajectories - {scheme}')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(os.path.dirname(__file__), f'{scheme}-algorithm/plots', f'trajectories_{scheme}.png'))  # Save the plot to a file

def get_reference_from_path_msg(path_msg):
    reference = []
    for pose in path_msg.poses:
        reference.append([pose.pose.position.x, pose.pose.position.y])
    return np.array(reference)

def plot_slack_centralised(slack_tracker, scheme='centralised'):
    """
    Plot the slack values for the NMPC and CA solvers.
    Args:
        slack_tracker: The SlackTracker object for the NMPC solver.
    """
    plt.figure(figsize=(12, 6))
    
    # Plot NMPC slack values
    plt.subplot(1, 2, 1)
    plt.plot(slack_tracker.get_max_sl(), label='NMPC Lower Slack', color='blue')
    plt.plot(slack_tracker.get_max_su(), label='NMPC Upper Slack', color='red')
    plt.xlabel('Time Step')
    plt.ylabel('Slack Values')
    plt.title('Max NMPC Slack Values')
    plt.legend()
    plt.grid(True)

    plt.savefig(os.path.join(os.path.dirname(__file__), 'centralised-algorithm/plots', 'max_slack_values.png'))  # Save the plot to a file

def plot_slack_distributed(slack_tracker_nmpc, slack_tracker_ca, scheme='centralised'):
    """
    Plot the slack values for the NMPC and CA solvers.
    Args:
        slack_tracker_nmpc: The SlackTracker object for the NMPC solver.
        slack_tracker_ca: The SlackTracker object for the CA solver.
    """
    plt.figure(figsize=(12, 6))
    
    # Plot NMPC slack values
    plt.subplot(1, 2, 1)
    plt.plot(slack_tracker_nmpc.get_max_sl(), label='NMPC Lower Slack', color='blue')
    plt.plot(slack_tracker_nmpc.get_max_su(), label='NMPC Upper Slack', color='red')
    plt.xlabel('Time Step')
    plt.ylabel('Slack Values')
    plt.title('Max NMPC Slack Values')
    plt.legend()
    plt.grid(True)

    # Plot CA slack values
    plt.subplot(1, 2, 2)
    plt.plot(slack_tracker_ca.get_max_sl(), label='CA Lower Slack', color='blue')
    plt.plot(slack_tracker_ca.get_max_su(), label='CA Upper Slack', color='red')
    plt.xlabel('Time Step')
    plt.ylabel('Slack Values')
    plt.title('Max CA Slack Values')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), 'distributed-algorithm/plots', 'max_slack_values.png'))  # Save the plot to a file

