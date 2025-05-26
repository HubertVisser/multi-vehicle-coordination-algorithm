import os, sys
import pathlib

def get_solver_import_paths():
    path = pathlib.Path(__file__).parent.resolve()
    sys.path.append(os.path.join(path, "..", "solver_generator"))
    sys.path.append(os.path.join(path))

def get_robot_pairs_one(number_of_robots):
    """
    Returns a dictionary with unique robot pairs with i < j.
    The key is a string 'i_j'.
    """
    pairs = {f'{i}_{j}': None 
             for i in range(1, number_of_robots + 1) 
             for j in range(i + 1, number_of_robots + 1)}
    return pairs

def get_robot_pairs_both(number_of_robots):
    """
    Returns a dictionary with unique robot pairs.
    The key is a string 'i_j'.
    """
    pairs = {f'{i}_{j}': None
             for i in range(1, number_of_robots + 1)
             for j in range(1, number_of_robots + 1)
             if i != j}
    return pairs
