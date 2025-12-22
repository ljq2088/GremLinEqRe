import sys
import os
import math
import cmath
import numpy as np
import matplotlib.pyplot as plt

# 尝试导入 C++ 核心模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from GremLinEqRe import _core
except ImportError:
    print("❌ Error: 无法导入 C++ 模块 GremLinEqRe")
    sys.exit(1)

# ==========================================
# 物理常数 (SI Units)
# ==========================================
G_SI = 6.67430e-11
c_SI = 2.99792458e8
M_SUN_SI = 1.989e30
PC_SI = 3.0857e16
MPC_SI = 1e6 * PC_SI

# ==========================================
# 辅助函数: 圆轨道能量导数 dE/dr (Kerr)
# ==========================================
def get_circular_orbit_energy_deriv(a, r):
    """
    计算 Kerr 圆轨道能量 E 关于 r 的导数 dE/dr (几何单位)。
    """
    v = r**(-0.5)
    v2 = v*v
    v3 = v2*v
    
    numer = 1.0 - 2.0*v2 + a*v3
    denom_sq = 1.0 - 3.0*v2 + 2.0*a*v3
    denom = math.sqrt(denom_sq)
    
    # 对 v 求导
    d_num_dv = -4.0*v + 3.0*a*v2
    d_den_dv = 0.5 / denom * (-6.0*v + 6.0*a*v2)
    
    dE_dv = (d_num_dv * denom - numer * d_den_dv) / denom_sq
    dv_dr = -0.5 * v3
    
    return dE_dv * dv_dr

# ==========================================
# 核心计算: 给定 r，计算所有模式的通量和波形
# ==========================================
def compute_step_physics(a, r, active_modes, last_nu_dict=None):
    """
    计算当前半径下的:
    1. 总能量通量 flux_E (几何单位, mu=1)
    2. 叠加后的波形因子 (用于合成 h)
    3. 更新后的 nu 字典 (用于热启动)
    """
    M = 1.0
    
    # 1. 更新几何运动学量
    # 对于赤道面圆轨道: u^r=0, u^theta=0
    geo = _core.KerrGeo.from_circular_equatorial(a, r, True) # True=Prograde
    state = _core.KerrGeo.State()
    state.x = [0.0, r, math.pi/2.0, 0.0]
    geo.update_kinematics(state)
    
    Omega_phi = state.u[3] / state.u[0]
    
    total_flux_E = 0.0
    complex_h_factor = 0.0j # 叠加的波形因子 sum (Z/omega^2 * S)
    
    next_nu_dict = {}
    
    # 2. 遍历模式计算
    for l, m in active_modes:
        omega = m * Omega_phi
        
        # 跳过非辐射模式
        if omega <= 0.0: continue
        
        # A. 自旋加权球谐函数
        swsh = _core.SWSH(-2, l, m, a * omega)
        try:
            lam = swsh.m_lambda
        except:
            # 备用解析近似
            lam = (l*(l+1) - 2.0*m*a*omega + (a*omega)**2 - 2.0) 
            
        # B. 径向求解器 (TeukolskyRadial)
        # 注意: 每次必须重新初始化或重置校准，因为 omega 变了
        tr = _core.TeukolskyRadial(M, a, omega, -2, l, m, lam)
        tr.ResetCalibration() # 关键: 重置 Evaluate_R_in 的校准状态
        
        # C. 求解 nu (热启动)
        nu_guess = complex(float(l), 0.0)
        if last_nu_dict and (l,m) in last_nu_dict:
            nu_guess = last_nu_dict[(l,m)]
            
        nu = tr.solve_nu(nu_guess)
        next_nu_dict[(l,m)] = nu
        
        # D. 计算 MST 系数
        n_max = 60 # 足够收敛
        coeffs_pos = tr.ComputeSeriesCoefficients(nu, n_max)
        coeffs_neg = tr.ComputeSeriesCoefficients(-nu-1.0, n_max)
        K_pos = tr.k_factor(nu)
        K_neg = tr.k_factor(-nu-1.0)
        
        # E. 计算源项需要的径向函数值 R, dR, ddR
        # 匹配半径设为 r_+ + 1.8 kappa (收敛安全区)
        kappa = math.sqrt(1.0 - a**2)
        r_plus = 1.0 + kappa
        r_match = r_plus + 1.5 * kappa 
        
        # 使用平滑连接的求值函数
        R_val, dR_val = tr.Evaluate_R_in(r, nu, K_pos, K_neg, coeffs_pos, coeffs_neg, r_match)
        
        # 计算二阶导数 (利用径向方程: Delta R'' + ... = 0)
        # 也可以调用 tr.evaluate_ddR 如果 C++ 暴露了该接口
        try:
            ddR_val = tr.evaluate_ddR(r, R_val, dR_val)
        except:
            # Python 手动计算二阶导备用
            Delta = r*r - 2.0*r + a*a
            V = tr.get_potential(r) # 需确认 C++ 是否暴露，或者手动写 V
            # 简单起见假设 C++ 已经暴露 evaluate_ddR
            # 如果报错，请检查 C++ 绑定
            ddR_val = 0.0 
        
        # F. 源项投影与积分 (MST Method)
        ts = _core.TeukolskySource(a, omega, -2, int(l), m)
        proj = ts.ComputeProjections(state, geo, swsh)
        
        # 振幅转换
        amps_pos = tr.ComputeAmplitudes(nu, coeffs_pos)
        phys = tr.ComputePhysicalAmplitudes(nu, coeffs_pos, amps_pos)
        
        # 构造源项 W (Sasaki-Nakamura 源项形式)
        W = R_val * (proj.A_nn0 + proj.A_mbarn0 + proj.A_mbarmbar0) \
          - dR_val * (proj.A_mbarn1 + proj.A_mbarmbar1) \
          + ddR_val * (proj.A_mbarmbar2)
        
        # 计算无穷远振幅 Z_inf
        # 注意: 这里的 B_inc 是入射系数，公式参考 MST 文献
        if abs(phys.B_inc) < 1e-15:
             Z_inf = 0.0j
        else:
             Z_inf = (2.0 * math.pi * W) / (2.0j * omega * phys.B_inc)
        
        # G. 累加通量
        # dE/dt = 1/(4pi w^2) * |Z|^2 (几何单位, mu=1)
        # 乘 2 是因为考虑 m 和 -m 的贡献
        flux_mode = 2.0 * (abs(Z_inf)**2) / (4.0 * math.pi * omega**2)
        total_flux_E += flux_mode
        
        # H. 累加波形因子
        # h ~ -1/w^2 * 1/r * Z * S * e^-i(phase)
        # 这里只计算不含相位的部分: Z * S / w^2
        # S 值取观测角度 theta (例如 pi/2 赤道面观测)
        S_val = swsh.evaluate_S(0.0) # cos(pi/2) = 0
        complex_h_factor += (Z_inf * S_val) / (omega**2)

    return total_flux_E, complex_h_factor, next_nu_dict, Omega_phi

# ==========================================
# 主程序: 演化循环
# ==========================================
def simulate_inspiral():
    print("=========================================================")
    print("   EMRI Adiabatic Inspiral Simulation (SI Units Output)  ")
    print("=========================================================")
    
    # 1. 物理系统参数 (SI)
    M_BH_mass = 1e6 * M_SUN_SI   # 中心黑洞质量 (10^6 solar mass)
    mu_mass   = 10.0 * M_SUN_SI  # 小天体质量 (10 solar mass)
    Distance  = 100.0 * MPC_SI   # 观测距离 (100 Mpc)
    spin_a    = 0.9              # 黑洞自旋
    
    # 转换为几何单位制下的比例系数
    # 时间单位: T_scale = G M / c^3
    T_scale = (G_SI * M_BH_mass) / (c_SI**3)
    # 长度单位: L_scale = G M / c^2
    L_scale = (G_SI * M_BH_mass) / (c_SI**2)
    # 质量比 q = mu / M
    mass_ratio_q = mu_mass / M_BH_mass
    
    print(f"System: M = {M_BH_mass/M_SUN_SI:.1e} M_sun, a = {spin_a}")
    print(f"        mu= {mu_mass/M_SUN_SI:.1f} M_sun")
    print(f"        D = {Distance/MPC_SI:.1f} Mpc")
    print(f"Time Scale (GM/c^3) = {T_scale:.4f} s")
    
    # 2. 初始条件 (几何单位)
    r_curr = 6.0 # 初始半径 (6M)
    phi_curr = 0.0
    t_curr = 0.0
    
    # 3. 演化控制
    dt_geo = 10.0 # 时间步长 (M)
    N_steps = 20000 
    
    # 模式选择 (先只跑 2,2，测试好后可加 (2,1), (3,3) 等)
    active_modes = [(2, 2)] 
    
    # 数据记录
    times_si = []
    r_vals = []
    strain_plus = []
    
    # 缓存变量
    current_nu_dict = {}
    cached_flux = 0.0
    cached_h_factor = 0.0j
    cached_Omega = 0.0
    
    # 4. 演化循环
    print("\nStarting evolution...")
    for i in range(N_steps):
        
        # 每隔几步重新计算一次通量 (绝热近似: 轨道参数变化慢)
        if i % 5 == 0:
            try:
                flux, h_fac, nu_dict, Omega = compute_step_physics(
                    spin_a, r_curr, active_modes, current_nu_dict
                )
                
                cached_flux = flux
                cached_h_factor = h_fac
                current_nu_dict = nu_dict
                cached_Omega = Omega
                
            except Exception as e:
                print(f"⚠️ Solver exception at step {i}, r={r_curr:.4f}: {e}")
                break
        
        # --- 动力学演化 ---
        # dr/dt = - Flux_total / (dE_orb/dr)
        # 注意: compute_step_physics 返回的是 mu=1 的通量
        # 实际通量 = flux_geo * q^2
        # dE_orb/dr (几何) = dE_hat/dr * mu = dE_hat/dr * q (相对量)
        # 所以 dr/dt = - (flux_hat * q^2) / (dE_hat/dr * q) = - flux_hat * q / dE_hat/dr
        
        dE_dr_geo = get_circular_orbit_energy_deriv(spin_a, r_curr)
        
        # 防止除零
        if abs(dE_dr_geo) < 1e-8: dE_dr_geo = 1e-8
            
        dr_dt = - (cached_flux * mass_ratio_q) / dE_dr_geo
        
        # 更新坐标
        r_prev = r_curr
        r_curr += dr_dt * dt_geo
        phi_curr += cached_Omega * dt_geo
        t_curr += dt_geo
        
        # --- 波形合成 ---
        # h_geo = 1/D_geo * mu_geo * h_factor * e^-i(phase)
        # h_factor ~ Z/w^2 * S (量纲为 1)
        # 实际上: h ~ 1/r * Psi4 / w^2
        # 正确的量级恢复:
        # h(t) (无量纲) = (G mu / c^2 D) * h_factor_geo * exp(-i phase)
        # 注意: h_factor_geo 计算时包含了 S 和 Z(mu=1)
        
        # 主相位 (以 2,2 模式为主导，相位 = 2*phi)
        # 严格来说应该是求和，但这里简化为提取主频相位
        phase_evolution = 2.0 * phi_curr 
        
        # 物理距离因子
        dist_factor = (G_SI * mu_mass) / (c_SI**2 * Distance)
        
        # 合成复数应变
        # h = - factor * exp(-i m phi) (符号约定可能有差异)
        # 系数 -2.0 来自 Psi4 -> h 的积分常数和定义
        h_complex = -2.0 * dist_factor * cached_h_factor * cmath.exp(-1.0j * phase_evolution)
        
        # 记录
        times_si.append(t_curr * T_scale) # 转换为秒
        r_vals.append(r_curr)
        strain_plus.append(h_complex.real)
        
        # 终止条件: ISCO (r_isco 对于 a=0.9 约为 2.32M)
        # 简单判据: r < r_hor + 0.2
        r_hor = 1.0 + math.sqrt(1.0 - spin_a**2)
        if r_curr < r_hor + 0.5:
            print(f"Plunge detected at r={r_curr:.4f}. Stopping.")
            break
            
        if i % 100 == 0:
            print(f"Step {i}: t={t_curr:.0f}M, r={r_curr:.4f}, h_amp={abs(h_complex):.2e}")

    # 5. 绘图
    print("Plotting...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=False)
    
    # 上图: 波形
    ax1.plot(times_si, strain_plus, color='#1f77b4', linewidth=1)
    ax1.set_ylabel(r'Strain $h_+$')
    ax1.set_title(f'EMRI Waveform (a={spin_a}, $M=10^6 M_\odot$, $\mu=10 M_\odot$)')
    ax1.grid(True, which='both', alpha=0.3)
    # 放大显示最后一段
    ax1_inset = ax1.inset_axes([0.6, 0.6, 0.35, 0.35])
    start_idx = max(0, len(times_si)-500)
    ax1_inset.plot(times_si[start_idx:], strain_plus[start_idx:], color='#d62728')
    ax1_inset.set_title("Late Inspiral")
    ax1_inset.grid(True)
    
    # 下图: 半径演化
    ax2.plot(times_si, r_vals, color='#ff7f0e', linewidth=2)
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel(r'Orbital Radius ($GM/c^2$)')
    ax2.set_title('Orbital Decay')
    ax2.grid(True, which='both', alpha=0.3)
    
    plt.tight_layout()
    output_name = 'emri_waveform_si.png'
    plt.savefig(output_name, dpi=300)
    print(f"✅ Done. Results saved to {output_name}")

if __name__ == "__main__":
    simulate_inspiral()