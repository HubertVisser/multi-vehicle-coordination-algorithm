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
        self.nu = 1
        self.nx = 3

        self.states = ["x", "vx", "s"] #["x", "y", "theta", "vx", "vy", "w", "s"]
        self.inputs = ["throttle"]#, "steering"] #, "slack"]

        self.lower_bound = [-2.0, -1000.0, -1000.0, -1000.0, -1000.0, -1000.0, -1000.0, -1000.0, -1000.0, -1000.0] # [u, x]
        self.upper_bound = [2.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0] # [u, x]
    
    def motor_force(self,throttle_filtered,v,a_m,b_m,c_m):
        w_m = 0.5 * (cd.tanh(100*(throttle_filtered+c_m))+1)
        Fx =  (a_m - b_m * v) * w_m * (throttle_filtered+c_m)
        return Fx
    
    def rolling_friction(self,vx,a_f,b_f,c_f,d_f):

        F_rolling = - ( a_f * cd.tanh(b_f  * vx) + c_f * vx + d_f * vx**2 )
        return F_rolling
    
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
    
    def steering_2_steering_angle(self,steering_command,a_s,b_s,c_s,d_s,e_s):
        w_s = 0.5 * (cd.tanh(30*(steering_command+c_s))+1)
        steering_angle1 = b_s * cd.tanh(a_s * (steering_command + c_s))
        steering_angle2 = d_s * cd.tanh(e_s * (steering_command + c_s))
        steering_angle = (w_s)*steering_angle1+(1-w_s)*steering_angle2
        return steering_angle
    

    #def continuous_model(self, x, u):

        th = u[0]
        st = u[1]
        theta = x[2]
        vx = x[3]

        # Define model parameters
        l, m, lr, lf, l_COM = self.model_parameters()

        # motor parameters
        a_m =  25.35849952697754    
        b_m =  4.815326690673828    
        c_m =  -0.16377617418766022 

        a_f =  1.2659882307052612
        b_f =  7.666370391845703
        c_f =  0.7393041849136353
        d_f =  -0.11231517791748047

        # steering angle curve --from fitting on vicon data
        a_s =  1.392930030822754
        b_s =  0.36576229333877563
        c_s =  0.0029959678649902344 - 0.03 # littel adjustment to allign the tire curves
        d_s =  0.5147881507873535
        e_s =  1.0230425596237183


        # convert steering command to steering angle
        steering_angle = self.steering_2_steering_angle(st, a_s, b_s, c_s, d_s, e_s)
        
        Fx_wheels = self.motor_force(th, vx, a_m, b_m, c_m)\
                + self.rolling_friction(vx, a_f, b_f, c_f, d_f)
        
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

    def continuous_model(self, x, u):
        """Dynamics model with only x, vx and throttle """

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

    th_activation1 = (cd.tanh((th - tau_offset) * tau_steepness) + 1) * tau_sat_high
    static_friction = np.tanh(v_friction_static_tanh_mult  * vx) * v_friction_static
    v_contribution = - static_friction - vx * v_friction - np.sign(vx) * vx ** 2 * v_friction_quad 
    print(th_activation1 + v_contribution)