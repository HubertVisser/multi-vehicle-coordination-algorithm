import casadi as cd
import numpy as np

n = 3
num_lam = (n-1)


lam = cd.MX.sym("lam", n, n*4)            
s = cd.MX.sym("s", n, n)            
d = [lam, s]
print("d: ", d)
print("s: ", s) 


# Print the entire matrix s
# print("Matrix s:")
# for i in range(n):
#     for j in range(n):
#         print(f"s[{i}, {j}] = {s[i, j]}")

# Print the entire matrix lam
print("Matrix lam:")
for i in range(n):
    for j in range(n*4):
        print(f"lam[{i}, {j}] = {lam[i, j]}")

# s_vec = cd.reshape(s,-1,1)
# print("s_vec: ", s_vec.shape)
# # Print the entire matrix s
# print("Matrix s:")
# for i in range(n*n):
#     print(f"s[{i}, 0] = {s_vec[i, 0]}")

lam_vec = lam.reshape((-1,1))
print("lam_vec: ", lam_vec.shape)
# Print the entire matrix s
print("Matrix lam:")
for i in range(n*n):
    print(f"lam[{i}, 0] = {lam_vec[i, 0]}")

# matrix = np.arange(0, n*n).reshape(n,n).T
# print("matrix: ", matrix)
# print("matrix: ", matrix.reshape(-1,1))
# print("matrix[1, 0]: ", matrix[1, 0])
# print("matrix[0, 1]: ", matrix[0, 1])

matrix_lam = np.arange(0, n*n).reshape(n,n).T
print("matrix_lam: ", matrix_lam)
print("matrix_lam: ", matrix_lam.reshape(-1,1))
print("matrix_lam[1, 0]: ", matrix_lam[1, 0])
print("matrix_lam[0, 1]: ", matrix_lam[0, 1])
