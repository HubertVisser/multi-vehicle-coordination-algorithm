import casadi as cd
import numpy as np

list = []
arr = np.array([1,2,3])
a = 2

dict1 = {}
dict2 = {}

dict1["s"] = [1,2,3]
# list.append(dict1)
dict2["s"] = [5,4,3]
# list.append(dict2)

arr[:2] += 5 if a == 1 else 0

print(arr)