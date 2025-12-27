from juliacall import Main as jl

# # 第一次运行时安装包（后续可注释掉）
# jl.seval('using Pkg; Pkg.add("SpinWeightedSpheroidalHarmonics")')

jl.seval("using SpinWeightedSpheroidalHarmonics")

s, l, m = -2, 2, 2
c = 0.9*0.3  # a*omega

# Julia 返回的是 Radial Lambda
lambda_radial = jl.spin_weighted_spheroidal_eigenvalue(s, l, m, c)
print("Radial Lambda (Teukolsky 1973) =", lambda_radial)



swsh = jl.spin_weighted_spheroidal_harmonic(s, l, m, c)

import numpy as np
theta = np.pi/6
phi   = np.pi/3

val0 = swsh(theta, phi)  # S(theta,phi)
val_dtheta = swsh(theta, phi, theta_derivative=1, phi_derivative=0)
val_dphi   = swsh(theta, phi, theta_derivative=0, phi_derivative=1)

print(val0, val_dtheta, val_dphi)
