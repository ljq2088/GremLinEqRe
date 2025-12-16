import sys
import os
import math
import cmath
import numpy as np

# 添加项目根目录到路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from GremLinEqRe import _core
except ImportError:
    print("❌ Error: 无法导入 C++ 模块 GremLinEqRe._core")
    sys.exit(1)

def test_circular_orbit_amplitudes():
    print("=========================================================")
    print("   Full Amplitude & Waveform Test (Corrected)            ")
    print("=========================================================")

    # 1. 物理参数设定
    M = 1.0
    a = 0.9          
    r_orbit = 5.0    
    s = -2           
    l = 2
    m = 2
    
    print(f"[Params] M={M}, a={a}, r0={r_orbit}, s={s}, l={l}, m={m}")

    # 2. 几何与轨道
    geo = _core.KerrGeo.from_circular_equatorial(a, r_orbit, True) 
    E = geo.energy
    Lz = geo.angular_momentum
    omega_phi = 1.0 / (a + math.pow(r_orbit, 1.5))
    omega = m * omega_phi # 波频率
    
    print(f"[Orbit]  E={E:.8f}, Lz={Lz:.8f}")
    print(f"[Freq]   Omega_phi={omega_phi:.8f}, omega_wave={omega:.8f}")

    # 3. 径向求解 (TeukolskyRadial)
    # ---------------------------------------------------------
    print("\n[Radial] Solving Radial Teukolsky Equation...")
    
    swsh = _core.SWSH(s, l, m, a * omega)
    
    # [修正 1] 使用 swsh.lambda 而不是 swsh.E
    # Teukolsky 方程中的 lambda = E_lm - 2amw + (aw)^2 - s(s+1)
    lam_val = swsh.lambda 
    print(f"         Eigenvalue (swsh.E)      = {swsh.E:.8f}")
    print(f"         Separation Const (lambda)= {lam_val:.8f} [CORRECTED]")

    tr = _core.TeukolskyRadial(M, a, omega, s, l, m, lam_val)
    
    # 求解 nu
    nu_guess = complex(float(l), 0.0)
    nu = tr.solve_nu(nu_guess)
    
    # 检查残差
    residual = tr.calc_g(nu)
    print(f"         Renormalized nu          = {nu:.8f}")
    print(f"         Residual |g(nu)|         = {abs(residual):.4e}")
    
    if abs(residual) > 1e-5:
        print("❌ Error: nu did not converge! Check lambda or initial guess.")
        return

    # 计算系数
    n_max = 40 # 提高一点精度
    coeffs_pos = tr.ComputeSeriesCoefficients(nu, n_max)
    coeffs_neg = tr.ComputeSeriesCoefficients(-nu - 1.0, n_max)
    
    K_pos = tr.k_factor(nu)
    K_neg = tr.k_factor(-nu - 1.0)
    
    # 计算波函数
    R_val, dR_val = tr.Evaluate_R_in(r_orbit, nu, K_pos, K_neg, coeffs_pos, coeffs_neg)
    ddR_val = tr.evaluate_ddR(r_orbit, R_val, dR_val)
    
    print(f"         R_in({r_orbit})           = {R_val:.4e}")

    # 4. 源项投影 (TeukolskySource)
    # ---------------------------------------------------------
    print("\n[Source] Computing Source Projections...")
    ts = _core.TeukolskySource(a, omega, m)
    
    # 构造状态
    state = _core.KerrGeo.State()
    state.x = [0.0, r_orbit, math.pi/2.0, 0.0]
    
    # 计算四速度
    Sigma = r_orbit**2
    Delta = r_orbit**2 - 2.0*M*r_orbit + a**2
    g_tt = -(1.0 - 2.0*M*r_orbit/Sigma)
    g_tphi = -2.0*M*a*r_orbit / Sigma
    term1 = (r_orbit**2 + a**2)**2
    g_phiphi = (term1 - Delta * a**2) / Sigma
    Gamma_sq = -(g_tt + 2.0*omega_phi*g_tphi + omega_phi**2 * g_phiphi)
    ut = 1.0 / math.sqrt(Gamma_sq)
    uphi = omega_phi * ut
    state.u = [ut, 0.0, 0.0, uphi]
    
    proj = ts.ComputeProjections(state, geo, swsh)

    # 5. 振幅计算
    # ---------------------------------------------------------
    print("\n[Final] Calculating Amplitudes...")
    
    amps_pos = tr.ComputeAmplitudes(nu, coeffs_pos)
    amps_neg = tr.ComputeAmplitudes(-nu - 1.0, coeffs_neg)
    
    # [修正 2] 传入 amps_pos (对应 nu 分支) 而不是 amps_neg
    phys_amps = tr.ComputePhysicalAmplitudes(nu, coeffs_pos, amps_pos)
    B_inc = phys_amps.B_inc
    print(f"         B_inc = {B_inc:.4e}")
    
    # Wronskian 源项积分核
    term0 = proj.A_nn0 + proj.A_mbarn0 + proj.A_mbarmbar0
    term1 = proj.A_mbarn1 + proj.A_mbarmbar1
    term2 = proj.A_mbarmbar2
    W_val = R_val * term0 - dR_val * term1 + ddR_val * term2
    
    # 最终复振幅 Z_inf
    Z_inf = W_val / (2.0j * omega * B_inc)
    
    print("\n---------------------------------------------------------")
    print(f"RESULTS: Z_infinity (l={l}, m={m})")
    print(f"Z_inf      = {Z_inf:.6e}")
    print(f"|Z_inf|    = {abs(Z_inf):.6e}")
    print("---------------------------------------------------------")

    # 6. 生成波形 Snapshot
    # ---------------------------------------------------------
    print("\n[Waveform] Generating Waveform Snapshot...")
    
    # 观察者位置
    r_obs = 1000.0 * M
    theta_obs = math.pi / 4.0 # 45度角观测
    phi_obs = 0.0
    t_obs = 0.0
    
    # 1. 获取 r_star (龟坐标) 近似: r* approx r + 2M ln(r/2M - 1)
    r_star = r_obs + 2*M * math.log(r_obs/(2*M) - 1.0)
    
    # 2. 计算 S_lm(theta)
    S_val = swsh.evaluate_S(math.cos(theta_obs))
    
    # 3. 计算 h+ - i hx
    # factor = -2 / (omega^2 * r)
    # phase = exp(i * (omega*(r* - t) + m*phi))
    # psi4 ~ Z_inf * factor_radial * phase * S_val
    # 注意: LRR 定义 Z_inf 已经是 psi4 的系数 (除去 1/r exp... S_lm)
    
    prefactor = -2.0 / (omega**2 * r_obs)
    phase_val = cmath.exp(1.0j * (omega*(r_star - t_obs) + m*phi_obs))
    
    h_complex = prefactor * Z_inf * phase_val * S_val
    
    h_plus = h_complex.real
    h_cross = -h_complex.imag
    
    print(f"Observer : r={r_obs}, theta={theta_obs:.2f}, phi={phi_obs}")
    print(f"Strain h : {h_complex:.4e}")
    print(f"   h_plus  = {h_plus:.4e}")
    print(f"   h_cross = {h_cross:.4e}")
    
    if abs(Z_inf) > 1e-10 and abs(residual) < 1e-5:
        print("\n✅ Test Passed: Consistent calculation.")
    else:
        print("\n❌ Test Failed: Results suspicious.")

if __name__ == "__main__":
    test_circular_orbit_amplitudes()