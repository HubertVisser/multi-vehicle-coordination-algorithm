import numpy as np


# Douglas Rachford projection
def project_trajectory_to_safety(p, delta, anchor, r, start_pose):
    """
        A way to project a trajectory to safety
        Anchor: An obstacle to function as the anchor
        delta: The obstacle to project from
        r: its radius
    """

    p[:] = (p + reflect(reflect(p, anchor, r, p), delta, r, start_pose)) / 2.0

def project(p, delta, r, start_pose):
    if np.linalg.norm(p - delta) < r:
        return delta - (delta - start_pose) / np.linalg.norm(start_pose - delta) * r
    else:
        return p

def reflect(p, delta, r, start_pose):
    return 2.0 * project(p, delta, r, start_pose) - p

# Example usage
if __name__ == "__main__":

    # Define the vectors as numpy arrays
    delta = np.array([0.5, 0.5])
    anchor = np.array([2.0, 2.0])
    start_pose = np.array([0.0, 0.0])
    r = 1.0

    p = start_pose #np.array([0.0, 0.0])
    # Apply the Douglas-Rachford projection
    project_trajectory_to_safety(p, delta, anchor, r, start_pose)
    
    print("Projected point:", p)
    print(f"Distance from obstacle after projection: {np.linalg.norm(delta - p):.3f}")