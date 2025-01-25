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

        self.inputs = ["throttle", "steering"] #, "slack"]
        self.states = ["x", "y", "theta", "vx", "vy", "w", "s"]

        self.lower_bound = [0.0, -1.0, -5.1, -1000.0, -1000.0, -1000.0, -1000.0, -1000.0, -1000.0, 0.0] # [u, x]
        self.upper_bound = [0.6, 1.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0] # [u, x]
    
    def model_parameters(self):
        lr_reference = 0.115  #0.11650    # (measureing it wit a tape measure it's 0.1150) reference point location taken by the vicon system measured from the rear wheel
        #COM_positon = 0.084 #0.09375 #centre of mass position measured from the rear wheel

        # car parameters
        l = 0.1735 # [m]length of the car (from wheel to wheel)
        m = 1.580 # mass [kg]
        m_front_wheel = 0.847 #[kg] mass pushing down on the front wheel
        m_rear_wheel = 0.733 #[kg] mass pushing down on the rear wheel

        COM_positon = l / (1+m_rear_wheel/m_front_wheel)
        lr = COM_positon
        lf = l-lr
        # Automatically adjust following parameters according to tweaked values
        l_COM = lr_reference - COM_positon
        return l, m, lr, lf, l_COM
    
    def motor_force(self, throttle_cmd, vx):
        # motor parameters
        a_m =  25.35849952697754    
        b_m =  4.815326690673828    
        c_m =  -0.16377617418766022 

        w_m = 0.5 * (cd.tanh(100*(throttle_cmd+c_m))+1)
        motor_force =  (a_m - b_m * vx) * w_m * (throttle_cmd+c_m)
        return motor_force
    
    def rolling_friction(self,vx):
        # friction parameters
        a_f =  1.2659882307052612
        b_f =  7.666370391845703
        c_f =  0.7393041849136353
        d_f =  -0.11231517791748047

        force_rolling_friction = - ( a_f * cd.tanh(b_f  * vx) + c_f * vx + d_f * vx**2 )
        return force_rolling_friction
    
    def fx_wheels(self,throttle_cmd, vx):
        fx_wheels = self.motor_force(throttle_cmd, vx) + self.rolling_friction(vx)
        return fx_wheels
    
    def steering_2_steering_angle(self,steering_cmd):
        # steering angle curve --from fitting on vicon data
        a_s =  1.392930030822754
        b_s =  0.36576229333877563
        c_s =  0.0029959678649902344 - 0.03 # littel adjustment to allign the tire curves
        d_s =  0.5147881507873535
        e_s =  1.0230425596237183

        w_s = 0.5 * (cd.tanh(30*(steering_cmd+c_s))+1)
        steering_angle1 = b_s * cd.tanh(a_s * (steering_cmd + c_s))
        steering_angle2 = d_s * cd.tanh(e_s * (steering_cmd + c_s))
        steering_angle = (w_s)*steering_angle1+(1-w_s)*steering_angle2
        return steering_angle
    
    def continuous_model(self, x, u): 


        th = u[0]
        st = u[1]
        theta = x[2]
        vx = x[3]

        # Define model parameters
        l, m, lr, lf, l_COM = self.model_parameters()

        # convert steering command to steering angle
        steering_angle = self.steering_2_steering_angle(st)
        
        # convert throttle to force on the wheels
        Fx_wheels = self.fx_wheels(th, vx)
        
        # Evaluate to acceleration
        acc_x = Fx_wheels / m
        
        w = vx * cd.tan(steering_angle) / (lr + lf)# angular velocity
        vy = l_COM * w

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
    
    def get_mass(self):
        return self.model_parameters()[1]

    #def continuous_model(self, x, u):
        """Dynamics model with states: x, vx and input: throttle """

        th = u[0]
        vx = x[1]

        # Define model parameters
        l, m, lr, lf, l_COM_self = self.model_parameters()

        # motor parameters
        a_m =  25.35849952697754    
        b_m =  4.815326690673828    
        c_m =  -0.16377617418766022 

        a_f =  1.2659882307052612
        b_f =  7.666370391845703
        c_f =  0.7393041849136353
        d_f =  -0.11231517791748047

        Fx_wheels = self.motor_force(th, vx, a_m, b_m, c_m)\
                + self.rolling_friction(vx, a_f, b_f, c_f, d_f)

        # Evaluate to acceleration
        acc_x = Fx_wheels / m

        xdot1 = vx
        xdot2 = acc_x
        sdot = vx
        xdot = [xdot1,xdot2]

        return cd.vertcat(*xdot, sdot)
    
    #def continuous_model(self, x, u):
        """Dynamics model with only input: steering """

        st = u[0]
        theta = x[2]
        vx = 1.0
        vy = x[4]
        w = x[5]

        # Define model parameters
        l, m, lr, lf, l_COM = self.model_parameters()

         # steering angle curve --from fitting on vicon data
        a_s =  1.392930030822754
        b_s =  0.36576229333877563
        c_s =  0.0029959678649902344 - 0.03 # littel adjustment to allign the tire curves
        d_s =  0.5147881507873535
        e_s =  1.0230425596237183


        # convert steering command to steering angle
        steering_angle = self.steering_2_steering_angle(st, a_s, b_s, c_s, d_s, e_s)
        w = vx * cd.tan(steering_angle) / (lr + lf)# angular velocity
        vy = l_COM * w

        xdot1 = vx * cd.cos(theta) - vy * cd.sin(theta)
        xdot2 = vx * cd.sin(theta) + vy * cd.cos(theta)
        xdot3 = w
        xdot4 = 0
        xdot5 = 0
        xdot6 = 0
        xdot = [xdot1,xdot2,xdot3, xdot4, xdot5, xdot6]

        return cd.vertcat(*xdot)

if __name__ == "__main__":

    model = BicycleModel2ndOrder()
    th = 1.0
    vx = 0.0
    model.continuous_model([0, vx, 0], [th])