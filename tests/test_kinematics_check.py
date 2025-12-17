import sys
import os
import math
import numpy as np

# 添加路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from GremLinEqRe import _core
except ImportError:
    print("❌ Error: 无法导入 C++ 模块")
    sys.exit(1)

def check_normalization(a, r, u):
    """
    验证 u.u = -1
    Metric (Boyer-Lindquist):
    ds^2 = - (1 - 2Mr/Sigma) dt^2 
           - (4Mar sin^2/Sigma) dt dphi 
           + (Sigma/Delta) dr^2 
           + Sigma dtheta^2 
           + ( (r^2+a^2)^2 - Delta a^2 sin^2 )/Sigma * sin^2 dphi^2
    """
    M = 1.0
    theta = math.pi / 2.0 # 赤道面
    Sigma = r**2
    Delta = r**2 - 2*M*r + a**2
    
    g_tt = -(1.0 - 2.0*M*r/Sigma)
    g_tp = -2.0*M*a*r/Sigma # sin=1
    g_rr = Sigma / Delta
    g_thth = Sigma
    g_pp = ((r**2 + a**2)**2 - Delta * a**2) / Sigma # sin=1
    
    ut, ur, uth, up = u
    
    norm = g_tt*ut**2 + 2*g_tp*ut*up + g_rr*ur**2 + g_thth*uth**2 + g_pp*up**2
    return norm

def test_kinematics_and_waveform():
    print("=========================================================")
    print("   Kinematics Validation & Waveform Test                 ")
    print("=========================================================")
    
    # 1. 验证 update_kinematics
    a = 0.9
    r_orbit = 5.0
    
    print(f"[Check] Verifying 4-velocity for r={r_orbit}, a={a}...")
    geo = _core.KerrGeo.from_circular_equatorial(a, r_orbit, True)
    
    state = _core.KerrGeo.State()
    state.x = [0.0, r_orbit, math.pi/2.0, 0.0]
    
    # C++ 计算四速度
    geo.update_kinematics(state, 0.0, 0.0)
    u_cpp = state.u
    
    print(f"  u^t   = {u_cpp[0]:.6f}")
    print(f"  u^phi = {u_cpp[3]:.6f}")
    print(f"  u^r   = {u_cpp[1]:.6f}")
    
    # 归一化检查
    norm = check_normalization(a, r_orbit, u_cpp)
    residual = abs(norm + 1.0)
    print(f"  u.u   = {norm:.8f}")
    print(f"  Error = {residual:.2e}")
    
    if residual > 1e-12:
        print("❌ Kinematics Check Failed!")
        sys.exit(1)
    else:
        print("✅ Kinematics Check Passed.")

    # 2. 运行完整波形流程 (单点测试)
    print("\n[Run] Computing Waveform Amplitude (l=2, m=2)...")
    
    omega_phi = u_cpp[3] / u_cpp[0]
    m = 2
    omega_wave = m * omega_phi
    s = -2
    l=2.0
    # SWSH
    swsh = _core.SWSH(s, int(l), m, a * omega_wave)
    # 使用新接口获取 lambda
    lam = swsh.m_lambda 
    
    # Radial
    tr = _core.TeukolskyRadial(1.0, a, omega_wave, s, int(l), m, lam)
    nu = tr.solve_nu(complex(l, 0.0))
    print(f"  nu = {nu}")
    
    # Coeffs & Function
    coeffs_pos = tr.ComputeSeriesCoefficients(nu, 40)
    coeffs_neg = tr.ComputeSeriesCoefficients(-nu-1, 40)
    K_pos = tr.k_factor(nu)
    K_neg = tr.k_factor(-nu-1)
    
    r_plus=1.0 + np.sqrt(1.0 - a**2)
    r_match=r_plus + 1.0 * np.sqrt(1.0 - a**2)
    
    
    R, dR = tr.Evaluate_R_in(r_orbit, nu, K_pos, K_neg, coeffs_pos, coeffs_neg,r_match)
    ddR = tr.evaluate_ddR(r_orbit, R, dR)
    
    # Source (使用自动计算好 u 的 state)
    ts = _core.TeukolskySource(a, omega_wave,s,int(l), m)
    proj = ts.ComputeProjections(state, geo, swsh)
    
    # Amplitudes
    amps_pos = tr.ComputeAmplitudes(nu, coeffs_pos)
    phys = tr.ComputePhysicalAmplitudes(nu, coeffs_pos, amps_pos)
    
    W = R * (proj.A_nn0 + proj.A_mbarn0 + proj.A_mbarmbar0) \
      - dR * (proj.A_mbarn1 + proj.A_mbarmbar1) \
      + ddR * (proj.A_mbarmbar2)
      
    Z_inf = W / (2.0j * omega_wave * phys.B_inc)
    
    print(f"  Z_inf = {Z_inf}")
    print(f"  |Z|   = {abs(Z_inf):.4e}")
    
    if abs(Z_inf) > 1e-10:
        print("✅ Waveform Pipeline Verified.")
    else:
        print("⚠️ Warning: Z_inf is very small/zero.")

if __name__ == "__main__":
    test_kinematics_and_waveform()