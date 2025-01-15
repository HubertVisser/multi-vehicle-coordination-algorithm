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
        u = cd.SX.sym("u", self.nu)  # [throttle, steering]
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
        self.nu = 2
        self.nx = 7

        self.states = ["x", "y", "theta", "vx", "vy", "w", "s"]
        self.inputs = ["throttle", "steering"] #, "slack"]

        self.lower_bound = [-1000.0, -1000.0, -1000.0, -1000.0, -1000.0, -1000.0, -1000.0, -1000.0, -1000.0, -1000.0] # [u, x]
        self.upper_bound = [1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0] # [u, x]
    
    def evaluate_Fx_2(self, vx, th):
        #fitted from same data as GP for ICRA 2024
        v_friction = 1.0683593
        v_friction_static = 1.1530068
        v_friction_static_tanh_mult = 23.637709
        v_friction_quad = 0.09363517

        tau_offset = 0.16150239
        tau_offset_reverse = 0.16150239
        tau_steepness = 10.7796755
        tau_steepness_reverse = 90
        tau_sat_high = 2.496312
        tau_sat_high_reverse = 5.0

        #friction model
        static_friction = cd.tanh(v_friction_static_tanh_mult  * vx) * v_friction_static
        v_contribution = - static_friction - vx * v_friction - cd.sign(vx) * vx ** 2 * v_friction_quad 

        #for positive throttle
        th_activation1 = (cd.tanh((th - tau_offset) * tau_steepness) + 1) * tau_sat_high
        #for negative throttle
        th_activation2 = (cd.tanh((th + tau_offset_reverse) * tau_steepness_reverse)-1) * tau_sat_high_reverse

        throttle_contribution = (th_activation1 + th_activation2) 

        # --------

        Fx = throttle_contribution + v_contribution
        Fx_r = Fx * 0.5
        Fx_f = Fx * 0.5
        return Fx_r, Fx_f
    
    def evaluate_vy_w_kin_bike(self, steering_angle,vx,L):
        
        w = vx * cd.tan(steering_angle) / L

        # lateral velocity of centre of mass
        R = L / cd.tan(steering_angle)
        beta = cd.atan2(L/2*cd.tan(steering_angle),L)
        R_star =  R / cd.cos(beta)
        vy = w * R_star * cd.sin(beta)

        return vy, w

    def steering_angle(self, steering_command):
        a =  1.6379064321517944
        b =  0.3301370143890381
        c =  0.019644200801849365
        d =  0.37879398465156555
        e =  1.6578725576400757

        w = 0.5 * (cd.tanh(30*(steering_command+c))+1)
        steering_angle1 = b * cd.tanh(a * (steering_command + c)) 
        steering_angle2 = d * cd.tanh(e * (steering_command + c))
        steering_angle = (w)*steering_angle1+(1-w)*steering_angle2 

        return steering_angle
    
    def motor_force_dart(self, vx, th):
        a =  28.887779235839844
        b =  5.986172199249268
        c =  -0.15045104920864105
        w = 0.5 * (cd.tanh(100*(th+c))+1)
        Fm =  (a - vx * b) * w * (th+c)
        return Fm

    def friction_dart(self, vx):
        a =  1.7194761037826538
        b =  13.312559127807617
        c =  0.289848655462265
        Ff = - a * cd.tanh(b  * vx) - vx * c
        return Ff

    def continuous_model(self, x, u):

        th = u[0]
        st = u[1]
        theta = x[2]
        vx = x[3]

        # Define constants for Jetracer
        m = 1.6759806
        l = 0.175

        #------- Kinematic Bike - Jetracer repo -------

        # convert steering command to steering angle
        steering_angle = self.steering_angle(st)
        
        Fx_r, Fx_f = self.evaluate_Fx_2(vx, th)

        # Evaluate to acceleration
        acc_x = (Fx_r + Fx_f * cd.cos(steering_angle)) / m
        
        vy, w = self.evaluate_vy_w_kin_bike(steering_angle, vx, l)

        xdot1 = vx * cd.cos(theta) - vy * cd.sin(theta)
        xdot2 = vx * cd.sin(theta) + vy * cd.cos(theta)
        xdot3 = w
        xdot4 = acc_x  
        xdot5 = 0
        xdot6 = 0

        xdot = [xdot1,xdot2,xdot3,xdot4,xdot5,xdot6]

        # v Simple s_dot approx taken from standard MPCC formulation
        s_dot = vx
        return cd.vertcat(*xdot, s_dot)

if __name__ == "__main__":

    model = BicycleModel2ndOrder()
    th = 1.0
    vx = -2.0
    Fx_wheels = model.motor_force(th, vx) + model.friction(vx)
    print(Fx_wheels)

    tau_offset = 0.16150239
    tau_steepness = 10.7796755
    tau_sat_high = 2.496312
    v_friction = 1.0683593
    v_friction_static = 1.1530068
    v_friction_static_tanh_mult = 23.637709
    v_friction_quad = 0.09363517

    th_activation1 = (np.tanh((th - tau_offset) * tau_steepness) + 1) * tau_sat_high
    static_friction = np.tanh(v_friction_static_tanh_mult  * vx) * v_friction_static
    v_contribution = - static_friction - vx * v_friction - np.sign(vx) * vx ** 2 * v_friction_quad 
    print(th_activation1 + v_contribution)