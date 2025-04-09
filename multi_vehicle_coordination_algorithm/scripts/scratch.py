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

arr = np.array([0,1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

print(arr[:3])
print(arr[7:10])