#!/usr/bin/env python

import rospy
from nav_msgs.msg import Path
from scipy.interpolate import CubicSpline
import numpy as np

from pyplot import plot_x_traj, plot_splines
# from path_generator import generate_path_msg

class SplineFitter:
    def __init__(self, settings):
        self._splines = []
        self._num_segments = settings["contouring"]["num_segments"]
        self._closest_s = None
        # rospy.init_node('spline_fitter', anonymous=True)
        # rospy.Subscriber('/reference_path', Path, self.path_callback)
        # rospy.spin()

    def fit_path(self, path_msg):
        if not path_msg.poses:
            rospy.logwarn("Received empty path")
            return

        self.x = []
        self.y = []
        for pose in path_msg.poses:
            self.x.append(pose.pose.position.x)
            self.y.append(pose.pose.position.y)

        if len(self.x) < 2:
            rospy.logwarn("Not enough points to fit splines")
            return

        self.fit_splines(self.x, self.y)
        # print(self.evaluate(0.))
        # print(self.evaluate(4.))
        # print(self.evaluate(6.7))
        # print(self.find_closest_s(np.array([3., 0.])))
        # splines = self.get_active_splines(np.array([3., 0.]))
        # print(splines)

        # self.log_splines()

    def fit_splines(self, x, y):
        # Clear previous splines
        newsplines = []

        # Compute cumulative path lengths
        distances = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
        s = np.cumsum(distances)
        s = np.insert(s, 0, 0)  # Include the starting point

        # Fit cubic splines to the path
        self.cs_x = CubicSpline(s, x)
        self.cs_y = CubicSpline(s, y)
        
        # Store splines and their parameters
        for i in range(len(s) - 1):
            spline = {
                'a_x': self.cs_x.c[:, i][0],
                'b_x': self.cs_x.c[:, i][1],
                'c_x': self.cs_x.c[:, i][2],
                'd_x': self.cs_x.c[:, i][3],
                'a_y': self.cs_y.c[:, i][0],
                'b_y': self.cs_y.c[:, i][1],
                'c_y': self.cs_y.c[:, i][2],
                'd_y': self.cs_y.c[:, i][3],
                's': s[i]
            }
            newsplines.append(spline)
        self._splines = newsplines


    def log_splines(self):
        for i, spline in enumerate(self._splines):
            rospy.loginfo("Spline %d: s = %.2f, a_x = %.2f, b_x = %.2f, c_x = %.2f, d_x = %.2f, a_y = %.2f, b_y = %.2f, c_y = %.2f, d_y = %.2f" % (
                i, spline['s'], spline['a_x'], spline['b_x'], spline['c_x'], spline['d_x'], spline['a_y'], spline['b_y'], spline['c_y'], spline['d_y']))
            

    def ready(self):
        return len(self._splines) > 0

    def get_active_splines(self, pos):
        self._closest_s = self.find_closest_s(pos)
        self._closest_x, self._closest_y, index = self.evaluate_and_get_spline_index(self._closest_s)

        result = []
        for i in range(self._num_segments):
            if index + i < len(self._splines):
                result.append(self._splines[index + i])
            else:
                result.append(self._splines[-1])
        return result        

    def evaluate(self, s):
        x, y, _ = self.evaluate_and_get_spline_index(s)
        return x, y

    def evaluate_and_get_spline_index(self, s):
        if not self._splines:
            rospy.logwarn("Splines have not been computed yet")
            return None, None, None

        # Find the correct spline segment for the given s
        for i in range(len(self._splines) - 1):
            if s >= self._splines[i]['s'] and s < self._splines[i + 1]['s']:
                spline = self._splines[i]
                spline_index = i
                break
        else:
            # If s is beyond the last segment, use the last spline
            spline = self._splines[-1]
            spline_index = len(self._splines) - 1

        # Compute the parameter t for the cubic polynomial
        t = s - spline['s']

        # Evaluate the spline at parameter t
        x = spline['a_x'] * t**3 + spline['b_x'] * t**2 + spline['c_x'] * t + spline['d_x']
        y = spline['a_y'] * t**3 + spline['b_y'] * t**2 + spline['c_y'] * t + spline['d_y']

        return x, y, spline_index

    def distance_to_point(self, s, point):
        x, y = self.evaluate(s)
        return np.sqrt((x - point[0])**2 + (y - point[1])**2)

    def find_closest_s(self, point, tol=1e-5):
        if not self._splines:
            rospy.logwarn("Splines have not been computed yet")
            return None

        # Total path length
        s_start = self._splines[0]['s']
        s_end = self._splines[-1]['s'] + np.sqrt((self._splines[-1]['a_x'])**2 + (self._splines[-1]['a_y'])**2)
        #               23.8                               
        # Bisection method
        while s_end - s_start > tol:
            s_mid1 = s_start + (s_end - s_start) / 3
            s_mid2 = s_end - (s_end - s_start) / 3

            d_mid1 = self.distance_to_point(s_mid1, point)
            d_mid2 = self.distance_to_point(s_mid2, point)

            if d_mid1 < d_mid2:
                s_end = s_mid2
            else:
                s_start = s_mid1

        s_closest = (s_start + s_end) / 2
        return s_closest
    
if __name__ == "__main__":
    settings = {}
    settings["contouring"] = {}
    settings["contouring"]["num_segments"] = 5
    rospy.init_node('contouring_spline')
    path_msg = generate_path_msg()
    spline_fitter = SplineFitter(settings)
    spline_fitter.fit_path(path_msg)
    x, y = spline_fitter.evaluate(15.)
    print(x, y)
    plot_splines(spline_fitter.cs_x, spline_fitter.cs_y, spline_fitter.x, spline_fitter.y)
    
    
