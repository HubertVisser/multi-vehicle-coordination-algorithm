import rospy
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Pose, Quaternion
import tf.transformations as tf
import numpy as np

arr = np.zeros((2, 2))
arr[0, 0] = 1

if np.all(arr==0):
    print("empty")