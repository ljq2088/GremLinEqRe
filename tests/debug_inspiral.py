import sys
import os
import math
import cmath
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from GremLinEqRe import _core
except ImportError:
    print("❌ Error: 无法导入 C++ 模块")
    sys.exit(1)

# ==========================================
# 能量导数逻辑检查
# ==========================================
def debug_energy_deriv(a, r):
    print(f"\n--- Debugging dE/dr at r={r}, a={a} ---")
    try:
        v = r**(-0.5)
        v2 = v*v
        v3 = v2*v
        
        numer = 1.0 - 2.0*v2 + a*v3
        denom_sq = 1.0 - 3.0*v2 + 2.0*a*v3
        
        print(f"  v       = {v:.6f}")
        print(f"  denom_sq= {denom_sq:.6f}")
        
        if denom_sq <= 0:
            print("  ❌ Error: denom_sq <= 0, orbit unstable/invalid!")
            return None
            
        denom = math.sqrt(denom_sq)
        E = numer / denom
        print(f"  Energy E= {E:.6f}")
        
        # d(numer)/dv = -4v + 3av^2
        d_num_dv = -4.0*v + 3.0*a*v2
        
        # d(denom)/dv = 1/(2*sqrt) * (-6v + 6av^2)
        d_den_dv = 0.5 / denom * (-6.0*v + 6.0*a*v2)
        
        dE_dv = (d_num_dv * denom - numer * d_den_dv) / denom_sq
        dv_dr = -0.5 * v3
        dE_dr = dE_dv * dv_dr
        
        print(f"  dE/dv   = {dE_dv:.6f}")
        print(f"  dv/dr   = {dv_dr:.6f}")
        print(f"  dE/dr   = {dE_dr:.6e}")
        
        return dE_dr
    except Exception as e:
        print(f"  ❌ Exception in dE/dr: {e}")
        return None

# ==========================================
# 振幅与通量逻辑检查
# ==========================================
def debug_flux_calc(a, r):
    print(f"\n--- Debugging Flux at r={r}, a={a} ---")
    M = 1.0
    l, m = 2, 2
    s = -2
    
    # 1. 几何
    geo = _core.KerrGeo.from_circular_equatorial(a, r, True)
    state = _core.KerrGeo.State()
    state.x = [0.0, r, math.pi/2.0, 0.0]
    geo.update_kinematics(state)
    
    Omega_phi = state.u[3] / state.u[0]
    omega = m * Omega_phi
    print(f"  Omega   = {Omega_phi:.6f}")
    print(f"  omega   = {omega:.6f}")
    
    if math.isnan(omega):
        print("  ❌ Error: omega is NaN (KerrGeo issue?)")
        return
    
    # 2. SWSH
    swsh = _core.SWSH(s, l, m, a * omega)
    try: lam = swsh.m_lambda
    except: lam = swsh.E - 2*m*a*omega + (a*omega)**2 - s*(s+1)
    print(f"  Lambda  = {lam:.6f}")
    
    # 3. Radial Solver
    tr = _core.TeukolskyRadial(M, a, omega, s, l, m, lam)
    nu = tr.solve_nu(complex(float(l), 0.0))
    print(f"  nu      = {nu}")
    
    # 4. Coefficients
    print("  Computing Coeffs...", end="")
    n_max=50
    try:
        coeffs_pos = tr.ComputeSeriesCoefficients(nu, n_max)
        coeffs_neg = tr.ComputeSeriesCoefficients(-nu-1, n_max)
        K_pos = tr.k_factor(nu)
        K_neg = tr.k_factor(-nu-1)
        print(" Done.")
    except Exception as e:
        print(f"\n  ❌ Error in Coeffs: {e}")
        return

    # 5. R Functions
    print("  Evaluating R_in...", end="")
    r_plus= 1.0 + np.sqrt(1.0 - a**2)
    r_match=r_plus + 1.8 * np.sqrt(1.0 - a**2)
    r_match_test=20.0
    print(f"\n  r_match = {r_match:.4e}, r_match_test = {r_match_test:.4e}")
    try:
        R, dR = tr.Evaluate_R_in(r, nu, K_pos, K_neg, coeffs_pos, coeffs_neg, r_match_test)
        ddR = tr.evaluate_ddR(r, R, dR)
        print(f" Done.\n  R       = {R:.4e}\n  dR      = {dR:.4e}\n  ddR     = {ddR:.4e}")
    except Exception as e:
        print(f"\n  ❌ Error in Evaluate_R_in: {e}")
        return
        
    # 6. Source
    ts = _core.TeukolskySource(a, omega,-2,l, m)
    proj = ts.ComputeProjections(state, geo, swsh)
    print(f"  A_nn0   = {proj.A_nn0:.4e}")
    
    # 7. Amplitudes (Suspected Area)
    print("  Computing B_inc...", end="")
    try:
        amps_pos = tr.ComputeAmplitudes(nu, coeffs_pos)
        phys = tr.ComputePhysicalAmplitudes(nu, coeffs_pos, amps_pos)
        B_inc = phys.B_inc
        print(f" Done.\n  B_inc   = {B_inc:.4e}")
    except Exception as e:
        print(f"\n  ❌ Error in B_inc: {e}")
        return
        
    if abs(B_inc) < 1e-20:
        print("  ⚠️ Warning: B_inc is extremely small!")

    # 8. Wronskian
    W = R * (proj.A_nn0 + proj.A_mbarn0 + proj.A_mbarmbar0) \
      - dR * (proj.A_mbarn1 + proj.A_mbarmbar1) \
      + ddR * (proj.A_mbarmbar2)
    print(f"  W       = {W:.4e}")

    # 9. Z_inf
    Z_inf = W / (2.0j * omega * B_inc)
    print(f"  Z_inf   = {Z_inf:.4e}")
    
    if cmath.isnan(Z_inf):
        print("  ❌ CRITICAL: Z_inf is NaN!")
    
    flux = 2.0 * (abs(Z_inf)**2) / (4.0 * math.pi * omega**2)
    print(f"  Flux    = {flux:.4e}")
    
    return flux

def run_debug():
    # 测试导致崩溃的参数
    # Case 1: r=6.0, a=0.9
    print("================ DEBUG CASE 1 ================")
    a = 0.5
    r = 6.0
    dE_dr = debug_energy_deriv(a, r)
    flux = debug_flux_calc(a, r)
    
    if dE_dr is not None and flux is not None:
        dr_dt = - (flux * 50.0) / dE_dr
        print(f"\nResulting dr/dt = {dr_dt:.6e}")
        print(f"Next r would be = {r + dr_dt * 10.0}")
    else:
        print("\n❌ Cannot compute dr/dt due to NaNs.")

if __name__ == "__main__":
    run_debug()