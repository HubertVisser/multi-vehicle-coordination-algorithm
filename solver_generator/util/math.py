import numpy as np
import casadi as cd


def rotation_matrix(angle):
    return cd.vertcat(cd.horzcat(cd.cos(angle), -cd.sin(angle)), cd.horzcat(cd.sin(angle), cd.cos(angle)))


def haar_difference_without_abs(angle1, angle2):
    return cd.fmod(angle1 - angle2 + np.pi, 2 * np.pi) - np.pi


def huber_loss(value, quadratic_from, reference=0.0):
    """
    value: symbolic value
    reference: reference value
    delta: treshold to switch from quadratic to linear
    """
    # Calculate the residual
    residual = value - reference

    # Define the conditions for Huber loss
    is_small_error = cd.fabs(residual) <= quadratic_from
    small_error_loss = 0.5 * residual**2
    large_error_loss = quadratic_from * (cd.fabs(residual) - 0.5 * quadratic_from)

    # Use CasADi's conditional operation
    return cd.if_else(is_small_error, small_error_loss, large_error_loss)

def sigmoid(x, a, b):
    return 1 / (1 + np.exp(-a * (x - b)))

# `a` is the steepness of the descend ~10
# `b` is the transition point between the two functions
def blended_function(x, a, b, f1, f2):
    return (1 - sigmoid(x, a, b)) * f1(x) + sigmoid(x, a, b) * f2(x)

### --- Calulation polytopic constraints --- ###
def get_A(theta):
    assert theta.size1() == 1 and theta.size2() == 1, "theta must be a scalar CasADi symbol"
    
    rot_mat = rotation_matrix(theta)
    return cd.vertcat(rot_mat.T, -rot_mat.T)

def get_b(pos, theta, length, width):
    assert pos.size1() == 2 and pos.size2() == 1, "pos must be a 2 by 1 CasADi vector"
    
    dim_vector = cd.DM([length/2, width/2, length/2, width/2])
    _A = get_A(theta)
    return dim_vector + _A @ pos