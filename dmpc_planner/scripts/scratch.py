import casadi as cd

n = 8
num_lam = (n-1)


lam1 = cd.MX.sym("lam",num_lam, 4, num_lam)
lam2 = cd.MX.sym("lam",num_lam, 4, num_lam)

s = cd.MX.sym("s", n, n)


z = [lam1, s]
print(lam1)
