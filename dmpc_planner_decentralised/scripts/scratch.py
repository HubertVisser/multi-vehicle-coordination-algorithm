import casadi as cd
import numpy as np

dict1 = dict()
dict2 = {'c': 3, 'd': 4}

# Using dictionary unpacking
dict1.update(dict2)

print(dict1)

s = cd.SX.sym('s', 0)
print(s)