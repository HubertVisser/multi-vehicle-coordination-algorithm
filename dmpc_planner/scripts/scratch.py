import casadi as cd
import numpy as np

n = 8
num_lam = (n-1)


lam = cd.MX.sym("lam", n, 4, n)               # [x, y, omega, vx, vy, w, s]
s = cd.MX.sym("s", n, n)               # [throttle, steering]
d = [lam, s]
print("d: ", d)
