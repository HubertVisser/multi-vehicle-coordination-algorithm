import casadi as cd
import numpy as np

from solver_generator.util.files import model_map_path, write_to_yaml

from solver_generator.spline import Spline2D


class DynamicsModel:

    def __init__(self):
        self.nu = 0  # number of control variables
        self.nx = 0  # number of states

        self.states = []
        self.inputs = []

        self.lower_bound = []
        self.upper_bound = []

        self.params = None
        self.nx_integrate = None

    # def discrete_dynamics(self, z, p, settings, **kwargs):
    #     params = settings["params"]
    #     params.load(p)
    #     self.load(z)
    #     self.load_settings(settings)

    #     nx_integrate = self.nx if self.nx_integrate is None else self.nx_integrate
    #     # integrated_states = forces_discrete_dynamics(z, p, self, settings, nx=nx_integrate, **kwargs)

    #     integrated_states = self.model_discrete_dynamics(z, integrated_states, **kwargs)
    #     return integrated_states

    def model_discrete_dynamics(self, z, integrated_states, **kwargs):
        return integrated_states

    def get_nvar(self):
        return self.nu + self.nx

    def get_xinit(self):
        return range(self.nu, self.get_nvar())

    def acados_symbolics(self):
        x = cd.SX.sym("x", self.nx)  # [x, y, omega, vx, vy, w]
        u = cd.SX.sym("u", self.nu)  # [th, delta]
        z = cd.vertcat(u, x)
        self.load(z)
        return z

    def get_acados_dynamics(self):
        f_expl = self.continuous_model(self._z[self.nu :], self._z[: self.nu])
        return f_expl

    def get_x(self):
        return self._z[self.nu :]

    def get_u(self):
        return self._z[: self.nu]

    def get_acados_x_dot(self):
        return self._x_dot

    def get_acados_u(self):
        return self._z[: self.nu]

    def load(self, z):
        self._z = z

    def load_settings(self, settings):
        self.params = settings["params"]
        self.settings = settings

    def save_map(self):
        file_path = model_map_path()

        map = dict()
        for idx, state in enumerate(self.states):
            map[state] = ["x", idx + self.nu, self.get_bounds(state)[0], self.get_bounds(state)[1]]

        for idx, input in enumerate(self.inputs):
            map[input] = ["u", idx, self.get_bounds(input)[0], self.get_bounds(input)[1]]

        write_to_yaml(file_path, map)

    def integrate(self, z, settings, integration_step):
        return self.discrete_dynamics(z, settings["params"].get_p(), settings, integration_step=integration_step)

    def do_not_use_integration_for_last_n_states(self, n):
        self.nx_integrate = self.nx - n

    def get(self, state_or_input):
        if state_or_input in self.states:
            i = self.states.index(state_or_input)
            return self._z[self.nu + i]
        elif state_or_input in self.inputs:
            i = self.inputs.index(state_or_input)
            return self._z[i]
        else:
            raise IOError(f"Requested a state or input `{state_or_input}' that was neither a state nor an input for the selected model")

    def set_bounds(self, lower_bound, upper_bound):
        assert len(lower_bound) == len(upper_bound) == len(self.lower_bound)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def get_bounds(self, state_or_input):
        if state_or_input in self.states:
            i = self.states.index(state_or_input)
            return (
                self.lower_bound[self.nu + i],
                self.upper_bound[self.nu + i],
                self.upper_bound[self.nu + i] - self.lower_bound[self.nu + i],
            )
        elif state_or_input in self.inputs:
            i = self.inputs.index(state_or_input)
            return (
                self.lower_bound[i],
                self.upper_bound[i],
                self.upper_bound[i] - self.lower_bound[i],
            )
        else:
            raise IOError(f"Requested a state or input `{state_or_input}' that was neither a state nor an input for the selected model")


    def get_cost(self, state_or_input):
        var = self.get(state_or_input)
        lower_bound, upper_bound, range = self.get_bounds(state_or_input)

        return (var / range) ** 2

    def get_tracking_cost(self, state_or_input, tracking_value):
        var = self.get(state_or_input)
        lower_bound, upper_bound, range = self.get_bounds(state_or_input)

        return ((var - tracking_value) / range) ** 2
    
# Bicycle model
class BicycleModel2ndOrder(DynamicsModel):

    def __init__(self):
        super().__init__()
        self.nu = 3
        self.nx = 7

        self.states = ["x", "y", "theta", "vx", "vy" "w", "s"]
        self.inputs = ["throttle", "steering", "slack"]

        self.lower_bound = [-1000.0, -1000.0, -1000.0, -1000.0, -1000.0, -1000.0, -1000.0, -1000.0, -1000.0, -1000.0] # [u, x]
        self.upper_bound = [1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0] # [u, x]

    def motor_force(self, throttle, v):
        a =  28.887779235839844
        b =  5.986172199249268
        c =  -0.15045104920864105
        w = 0.5 * (cd.tanh(100*(throttle+c))+1)
        Fm =  (a - v * b) * w * (throttle+c)
        return Fm

    def friction(self, v):
        a =  1.7194761037826538
        b =  13.312559127807617
        c =  0.289848655462265
        Ff = - a * cd.tanh(b  * v) - v * c
        return Ff
    
    def steering_angle(self, steer_command):
        a = -1.2053807
        b = 0.38302866
        c = 0.08739186
        steer_angle = b * cd.tanh(a * (steer_command-c))
        return steer_angle
    
    def continuous_model(self, x, u):
        
        th = u[2]
        st = u[1]
        theta = x[2]
        vx = x[3]

        # Define constants for Jetracer
        m = 1.6759806
        l = 0.175
        l_r = 0.54*l

        Fx_wheels = self.motor_force(th, vx) + self.friction(vx)

        # Evaluate to acceleration TODO: Obtain mass parameter from somewhere
        acc_x =  Fx_wheels / m
        
        # convert steering command to steering angle
        steering_angle = self.steering_angle(st)

        # evaluate lateral velocity and yaw rate
        w = vx * cd.tan(steering_angle) / l # angular velocity
        vy = l_r * w

        xdot1 = vx * np.cos(theta) - vy * np.sin(theta)
        xdot2 = vx * np.sin(theta) + vy * np.cos(theta)
        xdot3 = w
        xdot4 = acc_x  
        xdot5 = 0 # vy dot is not used
        xdot6 = 0 # w dot is not used

        xdot = [xdot1,xdot2,xdot3,xdot4,xdot5,xdot6]

        # v Simple s_dot approx taken from standard MPCC formulation
        s_dot = vx

        return cd.vertcat(*xdot, s_dot)