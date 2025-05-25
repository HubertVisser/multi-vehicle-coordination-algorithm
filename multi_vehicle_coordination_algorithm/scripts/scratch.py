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

trajectory_received = {n: False for n in range(1, 2 + 1) if n != 1}

trajectory_received[2] = True

for j in trajectory_received:
    trajectory_received[j] = False
    
print(trajectory_received)