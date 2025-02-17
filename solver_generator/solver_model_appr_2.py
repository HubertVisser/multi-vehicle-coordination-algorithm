"""Approach 2 exists because the dual variables might be vectors with different entries instead of one value repeated multiple times"""

import casadi as cd
import numpy as np
import math

from util.files import model_map_path, write_to_yaml
from util.logging import print_value
from spline import Spline2D


class DynamicsModel:

    def __init__(self):
        self.nu = 0  # number of control variables
        self.nx = 0  # number of states

        self.states = []
        self.inputs = []
        self.dual_vars = []

        self.lower_bound = []
        self.upper_bound = []

        self.params = None
        self.nx_integrate = None

    def model_discrete_dynamics(self, z, integrated_states, **kwargs):
        return integrated_states

    def get_nvar(self):
        return self.nu + self.nx

    def get_xinit(self):
        return range(self.nu, self.get_nvar())

    def acados_symbolics_z(self):
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

class MultiRobotDynamicsModel():
    def __init__(self):
        self.n = 0  # number of robots
        self.nx = 0
        self.nu = 0
        self.nlam = 0
        self.ns = 0 

        self.states = []
        self.inputs = []
        self.lams = []
        self.s = []

        self.lower_bound_states = []
        self.upper_bound_states = []

        self.lower_bound_inputs = []
        self.upper_bound_inputs = []

        # Bounds for dual variables in constraint form

    def get_nvar(self):
        return (self.nx + self.nu + self.nlam + self.s)

    def acados_symbolics_z(self):
        x = cd.SX.sym("x", self.nx, self.n)               # [x, y, omega, vx, vy, w, s]
        u = cd.SX.sym("u", self.nu, self.n)               # [throttle, steering]
        z = cd.vertcat(x, u)
        self.load_z(z)
        return z
    
    def acados_symbolics_d(self):
        lam = cd.MX.sym("lam", self.n, 4, self.n)               # [x, y, omega, vx, vy, w, s]
        s = cd.MX.sym("s", self.n, self.n)               # [throttle, steering]
        d = [lam, s]    
        self.load_d(d)
        return d

    def get_acados_dynamics(self):
        f_expl = self.continuous_model(self._z[ : self.nx, :], self._z[self.nx : self.nx + self.nu, :])    #(x, u)
        return f_expl

    def get_x(self):
        return self._z[ : self.nx, :]
    
    def get_u(self):
        return self._z[self.nx : self.nx + self.nu, :]
    
    def get_lam(self):
        return self._d[:-1]
    
    def get_s(self):
        return self._d[-1]
    
    def get_acados_x(self):
        return cd.reshape(self.get_x(),-1, 1)
    
    def get_acados_u(self):    
        return cd.reshape(self.get_u(),-1, 1)
    
    def get_acados_d(self):    
        return cd.reshape(self.get_d(),-1, 1)

    def load_z(self, z):
        self._z = z
    
    def load_d(self, d):
        self._d = d

    def load_settings(self, settings):
        self.params = settings["params"]
        self.settings = settings

    def save_map(self):
        file_path = model_map_path()

        map = dict()
        for idx, state in np.ndenumerate(self.states):
            map[str(state)] = ["x", self.idx_states[idx].item() , self.get_bounds(state)[0], self.get_bounds(state)[1]]

        for idx, input in np.ndenumerate(self.inputs):      
            map[str(input)] = ["u", self.idx_inputs[idx].item(), self.get_bounds(input)[0], self.get_bounds(input)[1]]

        write_to_yaml(file_path, map)

    def do_not_use_integration_for_last_n_states(self, n):
        self.nx_integrate = self.nx - n

    def get(self, input_state_or_dual):
        if np.any(self.states == input_state_or_dual):
            i = np.where(self.states == input_state_or_dual)
            return self.get_x()[i]
        elif np.any(self.inputs == input_state_or_dual):
            i = np.where(self.inputs == input_state_or_dual)
            return self.get_u()[i]
        elif np.any(self.lams == input_state_or_dual):
            i = np.where(self.lams == input_state_or_dual)
            return self.get_lam()[i[0][0]][i[1][0], :].T      # This slice takes a copy from the original array
        elif np.any(self.s == input_state_or_dual):
            i = np.where(self.s == input_state_or_dual)
            return cd.vertcat(self.get_s()[i], self.get_s()[i[1][0], i[0][0]]).T
        else:
            raise IOError(f"Requested a state or input `{input_state_or_dual}' that was neither a state nor an input for the selected model")

    def get_bounds(self, input_state_or_dual): 
        
        if np.any(self.states == input_state_or_dual):
            i = np.where(self.states == input_state_or_dual)
            return (
                self.lower_bound_states.item((i[0][0], i[1][0])),
                self.upper_bound_states.item((i[0][0], i[1][0])),
                (self.upper_bound_states.item((i[0][0], i[1][0])) - self.lower_bound_states.item((i[0][0], i[1][0]))),
            )
        elif np.any(self.inputs == input_state_or_dual):
            i = np.where(self.inputs == input_state_or_dual)
            return (
                self.lower_bound_inputs.item((i[0][0],i[1][0])),
                self.upper_bound_inputs.item((i[0][0],i[1][0])),
                self.upper_bound_inputs.item((i[0][0],i[1][0])) - self.lower_bound_inputs.item((i[0][0],i[1][0])),
            )
        else:
            raise IOError(f"Requested a state or input `{input_state_or_dual}' that was neither a state nor an input for the selected model")


    def get_cost(self, input_state_or_dual):
        var = self.get(input_state_or_dual)
        lower_bound, upper_bound, range = self.get_bounds(input_state_or_dual)

        return (var / range) ** 2

    def get_tracking_cost(self, input_state_or_dual, tracking_value):
        var = self.get(input_state_or_dual)
        lower_bound, upper_bound, range = self.get_bounds(input_state_or_dual)

        return ((var - tracking_value) / range) ** 2
        
# Bicycle model
class BicycleModel2ndOrder(DynamicsModel):

    def __init__(self):
        super().__init__()
        self.nu = 2
        self.nx = 7     

        self.inputs = ["throttle", "steering"] #, "slack"]
        self.states = ["x", "y", "theta", "vx", "vy", "w", "s"]

        self.lower_bound = [0.0, -1.0, -5.1, -10.0, -1000.0, -1000.0, -1000.0, -1000.0, -1000.0, 0.0] # [u, x]
        self.upper_bound = [1.0, 1.0, 1000.0, 10.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0] # [u, x]
    
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

# Bicycle model multiple robots
class BicycleModel2ndOrderMultiRobot(MultiRobotDynamicsModel):

    def __init__(self, n):
        super().__init__()
        self.n = n

        # For one robot:
        self.nx = 7
        self.nu = 2
        self.nlam = (n-1) *n *4 # lambda variables
        self.ns = (n*n) -n # s variables

        # Define states and inputs
        self.lams = np.empty((n, n), dtype=object)
        self.s = np.empty((n, n), dtype=object)

        for i in range(n):
            sublist_states = np.array([f"{state}_{i+1}" for state in ["x", "y", "theta", "vx", "vy", "w", "s"]])
            sublist_inputs = np.array([f"{input}_{i+1}" for input in ["throttle", "steering"]])
            for j in range(n):
                if i != j:
                    self.lams[i,j] = f"lam_{i+1}_{j+1}"                
                    self.s[i,j] = f"s_{i+1}_{j+1}"        

            if len(self.states) == 0:
                self.states = sublist_states.reshape(-1, 1)
            else:
                self.states = np.hstack((self.states, sublist_states.reshape(-1, 1)))
            
            if len(self.inputs) == 0:
                self.inputs = sublist_inputs.reshape(-1, 1)
            else:
                self.inputs = np.hstack((self.inputs, sublist_inputs.reshape(-1, 1)))
 
        
        # Bounds on states and inputs
        self.lower_bound_states = np.tile(np.array([[-1000.0, -1000.0, -1000.0, -1000.0, -1000.0, -1000.0, -1000.0]]).T, (1, n))
        self.upper_bound_states = np.tile(np.array([[1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000]]).T, (1, n))
        
        self.lower_bound_inputs = np.tile([[0.0],[-1.0]], (1, n)) 
        self.upper_bound_inputs = np.tile([[1.0], [1.0]], (1, n))

        lower_bound_duals = np.tile([0.0], (n-1, 1))  # lambda
        upper_bound_duals = np.tile([1000.0], (n-1, 1))  # lambda
        lower_bound_duals = np.concatenate((lower_bound_duals, np.tile([-1.0], (n-1, 1))))  # s dual
        upper_bound_duals = np.concatenate((upper_bound_duals, np.tile([1.0], (n-1, 1))))  # s dual
                        
        self.lower_bound_duals = np.tile(lower_bound_duals, (1, n))
        self.upper_bound_duals = np.tile(upper_bound_duals, (1, n))  

        # Define index array
        self.idx_states = np.arange(0, self.states.size).reshape(self.n, self.nx)
        self.idx_states = self.idx_states.T
        self.idx_inputs = np.arange(self.states.size, self.states.size + self.inputs.size).reshape(self.n, self.nu)
        self.idx_inputs = self.idx_inputs.T

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
    
    def kinematic_bicycle_model(self, x, u): 

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


        # v Simple s_dot approx taken from standard MPCC formulation
        s_dot = vx

        xdot = [xdot1,xdot2,xdot3,xdot4,xdot5,xdot6,s_dot]

        return cd.vertcat(*xdot)
    
    def continuous_model(self, x, u):
        xdot = []
        for i in range(self.n):
            xi = x[: , i]
            ui = u[: , i]
            xdot_i = self.kinematic_bicycle_model(xi, ui)
            xdot.extend(xdot_i.elements())
        return cd.vertcat(*xdot)
    
    def get_mass(self):
        return self.model_parameters()[1]




if __name__ == "__main__":

    model = BicycleModel2ndOrderMultiRobot(2)
    model.acados_symbolics_z()
    model.acados_symbolics_d()
    model.get_acados_dynamics()
    model.save_map()
    # print("inputs",model.inputs)
    # print("states",model.states)
    # print("duals",model.duals)
    print("xdot", model.get_acados_dynamics().shape)
    
    
    print("lam",model.get_lam())
    print("s",model.get_s())
    print("bounds", model.get_bounds("lam_1_2"))
    # print("lower_bound_u_acados", model.lower_bound_inputs)
    # print("upper_bound_inputs flatten", model.upper_bound_u.flatten())
    # print("upper_bound_inputs", model.upper_bound_inputs)
    # print("upper_bound_inputs flatten", model.upper_bound_u.T.flatten())
    # print("lower_bound_inputs flatten", model.lower_bound_u.T.flatten())
    # print("acados_x",model.get_acados_x())
    # print("lower_bound_states", model.lower_bound_states)
    # print("lower_bound_states flatten", model.lower_bound_states.T.flatten())
    # print("upper_bound_states flatten", model.upper_bound_states.T.flatten())
    # print("acados_d",model.get_acados_d())
    # print("lower_bound_duals", model.lower_bound_duals)
    # print("lower_bound_duals flatten", model.lower_bound_duals.T.flatten())
    # print("upper_bound_duals flatten", model.upper_bound_duals.T.flatten())
    
    # print(model.get('steering_2'))

