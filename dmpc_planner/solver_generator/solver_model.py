import casadi as cd
import numpy as np

from util.files import model_map_path, write_to_yaml

from spline import Spline2D


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

    def discrete_dynamics(self, z, p, settings, **kwargs):
        params = settings["params"]
        params.load(p)
        self.load(z)
        self.load_settings(settings)

        nx_integrate = self.nx if self.nx_integrate is None else self.nx_integrate
        # integrated_states = forces_discrete_dynamics(z, p, self, settings, nx=nx_integrate, **kwargs)

        integrated_states = self.model_discrete_dynamics(z, integrated_states, **kwargs)
        return integrated_states

    def model_discrete_dynamics(self, z, integrated_states, **kwargs):
        return integrated_states

    def get_nvar(self):
        return self.nu + self.nx

    def get_xinit(self):
        return range(self.nu, self.get_nvar())

    def acados_symbolics(self):
        x = cd.SX.sym("x", self.nx)  # [px, py, vx, vy]
        u = cd.SX.sym("u", self.nu)  # [ax, ay]
        z = cd.vertcat(u, x)
        self.load(z)
        return z

    def get_acados_dynamics(self):
        self._x_dot = cd.SX.sym("x_dot", self.nx)

        f_expl = numpy_to_casadi(self.continuous_model(self._z[self.nu :], self._z[: self.nu]))
        f_impl = self._x_dot - f_expl
        return f_expl, f_impl

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
        self.nx = 6

        self.states = ["x", "y", "psi", "v", "delta", "spline"]
        self.inputs = ["a", "w", "slack"]

        # Prius limits: https://github.com/oscardegroot/lmpcc/blob/prius/lmpcc_solver/scripts/systems.py
        # w [-0.2, 0.2] | a [-1.0 1.0]
        # w was 0.5
        # delta was 0.45

        # NOTE: the angle of the vehicle should not be limited to -pi, pi, as the solution will not shift when it is at the border!
        # a was 3.0
        self.lower_bound = [-3.0, -1.5, 0.0, -1.0e6, -1.0e6, -np.pi * 4, -0.01, -0.55, -1.0]
        self.upper_bound = [3.0, 1.5, 1.0e2, 1.0e6, 1.0e6, np.pi * 4, 5.0, 0.55, 5000.0]

    def motor_force(th,v):
        a =  28.887779235839844
        b =  5.986172199249268
        c =  -0.15045104920864105
        w = 0.5 * (np.tanh(100*(th+c))+1)
        Fm =  (a - v * b) * w * (th+c)
        return Fm

    def friction(v):
        a =  1.7194761037826538
        b =  13.312559127807617
        c =  0.289848655462265
        Ff = - a * np.tanh(b  * v) - v * c
        return Ff
    
    def continuous_model(self, x, u):
        
        th = u[2]
        st = u[1]
        yaw = x[2]
        vx = x[3]

        Fx_wheels = self.motor_force(th,vx) + self.friction(vx)

        acc_x =  Fx_wheels / self.m_self # evaluate to acceleration

        # convert steering to steering angle
        steering_angle = self.steering_2_steering_angle(st,self.a_s_self,self.b_s_self,self.c_s_self,self.d_s_self,self.e_s_self)
        # evaluate lateral velocity and yaw rate
        w = vx * np.tan(steering_angle) / (self.lr_self + self.lf_self) # angular velocity
        vy = self.l_COM_self * w

        # assemble derivatives
        xdot = self.produce_xdot(yaw,vx,vy,w,acc_x,0,0) # vy, acc y and acc w are 0 in this case
        return xdot