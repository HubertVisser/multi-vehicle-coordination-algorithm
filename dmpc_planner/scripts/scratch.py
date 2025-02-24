import casadi as cd
import numpy as np

n = 2
num_lam = (n-1)


for i in range(1, n+1):
    for j in range(i, n+1):
        if i != j:
            print(f"lam_{i}_{j}_0")
            print(f"s_{i}_{j}")