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
    print("   Full Amplitude Calculation Test (Circular Orbit)      ")
    print("=========================================================")

    # 1. 物理参数设定
    # ---------------------------------------------------------
    M = 1.0
    a = 0.9          # 高自旋
    r_orbit = 5.0    # 轨道半径 (M)
    s = -2           # 引力波
    l = 2
    m = 2
    
    print(f"[Params] M={M}, a={a}, r0={r_orbit}, s={s}, l={l}, m={m}")

    # 2. 几何与轨道 (KerrGeo)
    # ---------------------------------------------------------
    # 使用工厂函数获取精确守恒量
    geo = _core.KerrGeo.from_circular_equatorial(a, r_orbit, True) # True=Prograde
    E = geo.energy
    Lz = geo.angular_momentum
    Q = geo.carter_constant # 应该为 0
    
    # 计算轨道角频率 Omega_phi = 1 / (a + r^(3/2))
    # 注意：这是几何单位制下的精确公式
    omega_phi = 1.0 / (a + math.pow(r_orbit, 1.5))
    
    # 引力波频率 (对于圆形轨道，是由 m 决定的单频)
    omega = m * omega_phi
    
    print(f"[Orbit]  E={E:.8f}, Lz={Lz:.8f}, Q={Q:.8f}")
    print(f"[Freq]   Omega_phi={omega_phi:.8f}, omega_wave={omega:.8f}")

    # 3. 径向波函数求解 (TeukolskyRadial)
    # ---------------------------------------------------------
    print("\n[Radial] Solving Radial Teukolsky Equation...")
    
    # 3.1 计算角向特征值 Lambda (使用 SWSH 模块)
    swsh = _core.SWSH(s, l, m, a * omega)
    lam = swsh.m_lambda # 获取本征值 E_lm (注意区分代码中的 lambda 和 E)
    print(f"         Angular Eigenvalue E_lm = {lam:.8f}")

    # 3.2 初始化径向求解器
    tr = _core.TeukolskyRadial(M, a, omega, s, l, m, lam)
    
    # 3.3 求解重整化角动量 nu
    nu_guess = complex(float(l), 0.0)
    nu = tr.solve_nu(nu_guess)
    print(f"         Renormalized Angular Momentum nu = {nu}")

    # 3.4 计算 MST 级数系数
    n_max = 50
    coeffs_pos = tr.ComputeSeriesCoefficients(nu, n_max)
    coeffs_neg = tr.ComputeSeriesCoefficients(-nu - 1.0, n_max)
    
    # 3.5 计算 K 因子
    K_pos = tr.k_factor(nu)
    K_neg = tr.k_factor(-nu - 1.0)
    r_plus= 1.0 + np.sqrt(1.0 - a**2)
    kappa= np.sqrt(1.0 - a**2)
    # 3.6 计算轨道位置处的波函数值 R, dR, ddR
    # Evaluate_R_in 会自动处理近场/远场匹配
    R_val, dR_val = tr.Evaluate_R_in(r_orbit, nu, K_pos, K_neg, coeffs_pos, coeffs_neg,r_plus+kappa)
    
    # 使用径向方程精确计算二阶导
    ddR_val = tr.evaluate_ddR(r_orbit, R_val, dR_val)
    
    print(f"         R_in({r_orbit})  = {R_val:.4e}")
    print(f"         R'_in({r_orbit}) = {dR_val:.4e}")
    print(f"         R''_in({r_orbit})= {ddR_val:.4e}")

    # 4. 源项投影计算 (TeukolskySource)
    # ---------------------------------------------------------
    print("\n[Source] Computing Source Projections T_lm...")
    
    ts = _core.TeukolskySource(a, omega,s, l, m)
    
    # 构造粒子状态 (State)
    state = _core.KerrGeo.State()
    
    # 位置: t=0 (平稳过程不依赖 t), r=r0, theta=pi/2, phi=0
    state.x = [0.0, r_orbit, math.pi/2.0, 0.0]
    
    # 速度: u^r=0, u^theta=0. 需要计算 u^t 和 u^phi
    # 利用归一化条件 g_uv u^u u^v = -1 和 u^phi = Omega * u^t
    # g_tt (ut)^2 + 2 g_tphi (ut)(uphi) + g_phiphi (uphi)^2 = -1
    # => (ut)^2 [ g_tt + 2 Omega g_tphi + Omega^2 g_phiphi ] = -1
    geo.update_kinematics(state, 0.0, 0.0)
    # 赤道面度规分量 (BL coords)
    Sigma = r_orbit**2
    Delta = r_orbit**2 - 2.0*M*r_orbit + a**2
    
    # g_tt = -(1 - 2Mr/Sigma)
    g_tt = -(1.0 - 2.0*M*r_orbit/Sigma)
    # g_tphi = -2Mar sin^2 / Sigma
    g_tphi = -2.0*M*a*r_orbit / Sigma
    # g_phiphi = ( (r^2+a^2)^2 - Delta a^2 sin^2 ) / Sigma * sin^2
    # at theta=pi/2, sin=1
    term1 = (r_orbit**2 + a**2)**2
    g_phiphi = (term1 - Delta * a**2) / Sigma
    
    Gamma_sq = -(g_tt + 2.0*omega_phi*g_tphi + omega_phi**2 * g_phiphi)
    ut = 1.0 / math.sqrt(Gamma_sq)
    uphi = omega_phi * ut
    
    state.u = [ut, 0.0, 0.0, uphi]
    print(f"         4-Velocity u^t={ut:.4f}, u^phi={uphi:.4f}")
    
    # 计算投影系数 A
    proj = ts.ComputeProjections(state, geo, swsh)
    
    print(f"         A_nn0       = {proj.A_nn0:.4e}")
    print(f"         A_mbarn0    = {proj.A_mbarn0:.4e}")
    print(f"         A_mbarmbar0 = {proj.A_mbarmbar0:.4e}")

    # 5. 综合计算振幅 Z (Sasaki-Tagoshi 公式)
    # ---------------------------------------------------------
    print("\n[Final] Calculating Amplitudes Z_infinity...")
    
    # 计算渐近系数 B_inc (入射波系数)
    # 需要先计算渐近振幅 AsymptoticAmplitudes
    amps_pos = tr.ComputeAmplitudes(nu, coeffs_pos)
    amps_neg = tr.ComputeAmplitudes(-nu - 1.0, coeffs_neg)
    
    phys_amps = tr.ComputePhysicalAmplitudes(nu, coeffs_pos, amps_pos)
    B_inc = phys_amps.B_inc
    print(f"         B_inc = {B_inc:.4e}")
    
    # 计算 Wronskian 源项 W (LRR Eq. 44)
    # W = R_in * [A_nn0 + A_mbarn0 + A_mbarmbar0] 
    #   - dR_in * [A_mbarn1 + A_mbarmbar1] 
    #   + ddR_in * [A_mbarmbar2]
    
    term0 = proj.A_nn0 + proj.A_mbarn0 + proj.A_mbarmbar0
    term1 = proj.A_mbarn1 + proj.A_mbarmbar1
    term2 = proj.A_mbarmbar2
    
    W_val = R_val * term0 - dR_val * term1 + ddR_val * term2
    print(f"         Source Integrand W = {W_val:.4e}")
    
    # 计算最终振幅 Z_infinity (LRR Eq. 25, 43)
    # 对于圆形轨道，积分 int dt e^{i(w t - m phi)} 给出 2*pi * delta(w - m*Omega)
    # 除去 delta 函数，剩下的系数为：
    # Z_inf_coeff = W_val / (2 * i * omega * B_inc)
    # (注：通常还会乘上 sqrt(2pi) 或其他归一化因子，这里计算的是核心复振幅)
    
    Z_inf = W_val / (2.0j * omega * B_inc)
    
    print("\n---------------------------------------------------------")
    print(f"RESULTS: Wave Amplitude (l={l}, m={m})")
    print(f"Z_infinity = {Z_inf}")
    print(f"|Z_inf|    = {abs(Z_inf):.6e}")
    print("---------------------------------------------------------")
    
    # 简单的一致性检查
    if abs(Z_inf) < 1e-20 or math.isnan(abs(Z_inf)):
        print("❌ Test Failed: Amplitude is zero or NaN")
        sys.exit(1)
    else:
        print("✅ Amplitude calculation successful (Non-zero result).")

if __name__ == "__main__":
    test_circular_orbit_amplitudes()