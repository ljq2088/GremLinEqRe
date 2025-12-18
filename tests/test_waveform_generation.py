import sys
import os
import math
import cmath
import numpy as np
import matplotlib.pyplot as plt

# 路径设置
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from GremLinEqRe import _core
except ImportError:
    print("❌ Error: 无法导入 C++ 模块 GremLinEqRe._core")
    sys.exit(1)

def compute_amplitude_for_mode(a, r_orbit, s, l, m):
    """计算指定模式的振幅 Z_infinity"""
    M = 1.0
    
    # 1. 几何与运动学
    geo = _core.KerrGeo.from_circular_equatorial(a, r_orbit, True) 
    state = _core.KerrGeo.State()
    state.x = [0.0, r_orbit, math.pi/2.0, 0.0]
    
    # 使用修复后的 update_kinematics 自动计算 u
    geo.update_kinematics(state, 0.0, 0.0)
    
    # 从 u 计算频率 (dphi/dt = u^phi / u^t)
    omega_phi = state.u[3] / state.u[0]
    omega_wave = m * omega_phi
    
    print(f"  [Mode {l},{m}] Omega_phi={omega_phi:.6f}, omega_wave={omega_wave:.6f}")

    # 2. 角向波函数 (SWSH)
    swsh = _core.SWSH(s, l, m, a * omega_wave)
    # 使用新接口获取 lambda
    try:
        lam = swsh.m_lambda
    except AttributeError:
        # 兼容旧绑定的 fallback
        lam = swsh.E - 2*m*a*omega_wave + (a*omega_wave)**2 - s*(s+1)

    # 3. 径向方程求解
    tr = _core.TeukolskyRadial(M, a, omega_wave, s, l, m, lam)
    nu = tr.solve_nu(complex(float(l), 0.0))
    
    # 计算系数
    coeffs_pos = tr.ComputeSeriesCoefficients(nu, 40)
    coeffs_neg = tr.ComputeSeriesCoefficients(-nu - 1.0, 40)
    K_pos = tr.k_factor(nu)
    K_neg = tr.k_factor(-nu - 1.0)
    
    # 计算波函数
    r_plus = 1.0 + np.sqrt(1.0 - a**2)
    r_match = r_plus + 1.0 * np.sqrt(1.0 - a**2)
    R, dR = tr.Evaluate_R_in(r_orbit, nu, K_pos, K_neg, coeffs_pos, coeffs_neg,r_match)
    ddR = tr.evaluate_ddR(r_orbit, R, dR)

    # 4. 源项投影
    ts = _core.TeukolskySource(a, omega_wave,int(s), int(l), int(m))
    proj = ts.ComputeProjections(state, geo, swsh)

    # 5. 振幅合成
    amps_pos = tr.ComputeAmplitudes(nu, coeffs_pos)
    phys = tr.ComputePhysicalAmplitudes(nu, coeffs_pos, amps_pos)
    
    W = R * (proj.A_nn0 + proj.A_mbarn0 + proj.A_mbarmbar0) \
      - dR * (proj.A_mbarn1 + proj.A_mbarmbar1) \
      + ddR * (proj.A_mbarmbar2)
    
    Z_inf = W / (2.0j * omega_wave * phys.B_inc)
    
    return Z_inf, omega_wave, swsh

def generate_waveform():
    print("=========================================================")
    print("   Gravitational Waveform Generation (Circular)          ")
    print("=========================================================")
    
    M=1.0
    a = 0.9
    r_orbit = 5.0
    s = -2
    # 物理参数
    M_solar = 1.989e30
    G = 6.674e-11
    c = 2.998e8
    M_phys = 1.0e6 * M_solar
    m_phys = 10.0 * M_solar
    dist_phys = 1.0e9 * 3.086e16 # 1 Gpc in meters

    # 几何单位转换因子
    L_scale = G * M_phys / c**2 # 长度标度 (米)
    T_scale = G * M_phys / c**3 # 时间标度 (秒)

    mu = m_phys / M_phys # 质量比 1e-5
    r_obs_geo = dist_phys / L_scale # 几何单位下的距离
    # 观测者
    #换算到几何单位
    r_obs=r_obs_geo/L_scale
    theta_obs = math.pi / 4.0 # 45度
    phi_obs = 0.0
    
    # 计算 l=2, m=2 主模式
    print("Computing Mode (2, 2)...")
    l=2
    m=2
    Z_22, w_22, swsh_22 = compute_amplitude_for_mode(a, r_orbit, s, int(l), int(m))
    
    # 简单的波形合成 (h ~ Z/r * exp(-i wt))
    S_22 = swsh_22.evaluate_S(math.cos(theta_obs))
    r_star = r_obs + 2*M*math.log(r_obs/(2*M) - 1.0)
    
    # h = -2/w^2 * psi4
    factor = -2.0 / (w_22**2 * r_obs_geo) * S_22 * Z_22*mu
    dt=1.0#秒
    # 时间轴: 演化 3 个周期
    T_period = 2 * math.pi / (w_22 / 2.0) # omega_wave = 2 * Omega_phi
    times = np.arange(0, 3*T_period, dt)
    
    h_plus = []
    h_cross = []
    
    print(f"Generating waveform for {len(times)} points...")
    for t in times:
        # 相位: w(t - r*) + m*phi
        phase = w_22 * (t - r_star) + 2 * phi_obs
        
        # 完整的复数应变
        # 注意 Teukolsky 约定通常带有 exp(-i omega t)
        h_comp = factor * cmath.exp(-1.0j * phase)
        
        h_plus.append(h_comp.real)
        h_cross.append(-h_comp.imag)
    times *= T_scale # 转换为秒 
    # 绘图
    plt.figure(figsize=(10, 6))
    plt.plot(times, h_plus, label=r'$h_+$')
    plt.plot(times, h_cross, label=r'$h_\times$', linestyle='--')
    plt.title(f'GW Strain (l=2,m=2) from Circular Orbit r={r_orbit}M, a={a}')
    plt.xlabel('Time (s)')
    plt.ylabel('Strain')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    filename = 'waveform_circular_test.png'
    plt.savefig(filename)
    print(f"✅ Waveform plot saved to {filename}")

if __name__ == "__main__":
    generate_waveform()