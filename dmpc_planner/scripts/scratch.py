import casadi as cd
import numpy as np

n = 2
num_lam = (n-1)
s = cd.SX.sym("s", n)

print(cd.norm_2(s))