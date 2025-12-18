import sys
import os
import math
import cmath
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from GremLinEqRe import _core
except ImportError:
    print("❌ Error: 无法导入 C++ 模块")
    sys.exit(1)

# ==========================================
# 辅助函数: 圆轨道能量对半径的导数 dE/dr
# ==========================================
def get_circular_orbit_energy_deriv(a, r):
    """
    计算圆轨道能量 E 关于 r 的导数 dE/dr。
    E = (1 - 2v^2 + av^3) / sqrt(1 - 3v^2 + 2av^3)
    其中 v = r^(-1/2)
    """
    v = r**(-0.5)
    v2 = v*v
    v3 = v2*v
    
    numer = 1.0 - 2.0*v2 + a*v3
    denom_sq = 1.0 - 3.0*v2 + 2.0*a*v3
    denom = math.sqrt(denom_sq)
    
    E = numer / denom
    
    # 对 v 求导
    # d(numer)/dv = -4v + 3av^2
    d_num_dv = -4.0*v + 3.0*a*v2
    
    # d(denom)/dv = 1/(2*sqrt) * (-6v + 6av^2)
    d_den_dv = 0.5 / denom * (-6.0*v + 6.0*a*v2)
    
    dE_dv = (d_num_dv * denom - numer * d_den_dv) / denom_sq
    
    # dv/dr = -0.5 * r^(-1.5) = -0.5 * v^3
    dv_dr = -0.5 * v3
    
    dE_dr = dE_dv * dv_dr
    return dE_dr

# ==========================================
# 单步计算: 给定 r，计算通量和振幅
# ==========================================
def compute_step_physics(a, r, last_nu_dict=None):
    """
    计算当前半径下的 GW 通量和主要模式的振幅。
    last_nu_dict: 用于热启动 { (l,m): nu_guess }
    """
    M = 1.0
    
    # 1. 几何更新
    geo = _core.KerrGeo.from_circular_equatorial(a, r, True)
    state = _core.KerrGeo.State()
    state.x = [0.0, r, math.pi/2.0, 0.0]
    geo.update_kinematics(state)
    
    Omega_phi = state.u[3] / state.u[0]
    
    # 累加总能量通量
    total_flux_E = 0.0
    
    # 记录主要模式的波形信息 (这里只记录 2,2 用于画图)
    waveform_data = {}
    next_nu_dict = {}
    
    # 计算的模式列表
    modes = [(2, 2)] # 可以添加 (2,1), (3,3) 等
    
    for l, m in modes:
        omega = m * Omega_phi
        
        # SWSH
        swsh = _core.SWSH(-2, l, m, a * omega)
        try: lam = swsh.m_lambda
        except: lam = swsh.E - 2*m*a*omega + (a*omega)**2 - (-2)*(-1)
            
        # Radial (Hot Start)
        tr = _core.TeukolskyRadial(M, a, omega, -2, l, m, lam)
        
        # 获取猜测值
        nu_guess = complex(float(l), 0.0)
        if last_nu_dict and (l,m) in last_nu_dict:
            nu_guess = last_nu_dict[(l,m)]
            
        nu = tr.solve_nu(nu_guess)
        next_nu_dict[(l,m)] = nu # 保存供下一步使用
        
        # 计算源项
        n_max=100
        coeffs_pos = tr.ComputeSeriesCoefficients(nu, n_max)
        coeffs_neg = tr.ComputeSeriesCoefficients(-nu-1, n_max)
        K_pos = tr.k_factor(nu)
        K_neg = tr.k_factor(-nu-1)
        r_plus= 1.0 + np.sqrt(1.0 - a**2)
        r_match=r_plus + 1.8 * np.sqrt(1.0 - a**2)
        r_match_test=20.0
        R, dR = tr.Evaluate_R_in(r, nu, K_pos, K_neg, coeffs_pos, coeffs_neg, r_match)
        ddR = tr.evaluate_ddR(r, R, dR)
        
        ts = _core.TeukolskySource(a, omega,-2, int(l), m)
        proj = ts.ComputeProjections(state, geo, swsh)
        
        amps_pos = tr.ComputeAmplitudes(nu, coeffs_pos)
        phys = tr.ComputePhysicalAmplitudes(nu, coeffs_pos, amps_pos)
        
        W = R * (proj.A_nn0 + proj.A_mbarn0 + proj.A_mbarmbar0) \
          - dR * (proj.A_mbarn1 + proj.A_mbarmbar1) \
          + ddR * (proj.A_mbarmbar2)
          
        Z_inf = W / (2.0j * omega * phys.B_inc)
        
        # 通量贡献 dE/dt = 1/(4pi w^2) * |Z|^2 * 2 (m!=0 时正负m对称贡献)
        # 注意: Z_{l,-m} 和 Z_{l,m} 给出的通量相同，这里乘2考虑负m模式
        flux_contribution = 2.0 * (abs(Z_inf)**2) / (4.0 * math.pi * omega**2)
        total_flux_E += flux_contribution
        
        if l==2 and m==2:
            waveform_data['Z'] = Z_inf
            waveform_data['omega'] = omega
            waveform_data['S'] = swsh.evaluate_S(math.cos(math.pi/4.0)) # 观测角45度
            
    return total_flux_E, waveform_data, next_nu_dict

# ==========================================
# 演化主循环
# ==========================================
def simulate_inspiral():
    print("=========================================================")
    print("   Adiabatic Inspiral Simulation (Radiation Reaction)    ")
    print("=========================================================")
    
    # 物理系统
    M_sys = 1.0 # 几何单位
    a = 0.3
    
    # 初始条件
    r_curr = 10.0 # 从 r=6 (ISCO 附近) 开始
    t_curr = 0.0
    phi_curr = 0.0
    
    # 演化控制
    dt = 10.0 # 时间步长 (M)
    N_steps = 1000 # 步数
    
    # 为了演示效果，人为放大通量 (Accelerated Inspiral)
    # 真实的质量比 1e-5 需要演化百万步才能看到 r 明显变化
    # 这里假设我们模拟的是一个比较极端的质量比，或者单纯加速物理过程
    flux_multiplier = 50.0 
    
    # 数据记录
    times = []
    r_vals = []
    h_plus_vals = []
    
    # 热启动缓存
    current_nu_dict = None
    
    print(f"Start: r={r_curr:.4f}, a={a}, flux_scale={flux_multiplier}")
    print("Evolving...")
    
    for i in range(N_steps):
        # 1. 计算当前物理量
        # 为提高速度，不需要每一步都解 Teukolsky (r 变化很慢)
        # 但为了保证相位准确，频率必须每步积分。
        # 这里每 5 步更新一次 Flux 和 Amplitude
        
        if i % 5 == 0:
            try:
                flux_E, wave_data, nu_dict = compute_step_physics(a, r_curr, current_nu_dict)
                
                # 更新缓存
                current_nu_dict = nu_dict
                cached_flux = flux_E
                cached_Z = wave_data['Z']
                cached_omega = wave_data['omega']
                cached_S = wave_data['S']
                
            except Exception as e:
                print(f"Solver failed at step {i}, r={r_curr}: {e}")
                break
        
        # 2. 演化方程
        # dr/dt = - Flux / (dE/dr)
        dE_dr = get_circular_orbit_energy_deriv(a, r_curr)
        
        # 放大通量以模拟 fast inspiral
        dr_dt = - (cached_flux * flux_multiplier) / dE_dr
        
        # 3. 更新状态
        r_prev = r_curr
        r_curr += dr_dt * dt
        
        # 更新相位: dphi/dt = Omega. 
        # Omega = omega_wave / m = cached_omega / 2
        Omega_phi = cached_omega / 2.0
        phi_curr += Omega_phi * dt
        t_curr += dt
        
        # 4. 生成波形 (h ~ Z/r * exp(-i phase))
        # 观测距离 r_obs = 100.0 (任意几何距离)
        r_obs = 100.0
        factor = -2.0 / (cached_omega**2 * r_obs) * cached_S * cached_Z
        
        # 相位 = omega(t-r*) + m*phi. 
        # 简单起见，我们直接积分类似 2*phi_curr 的相位
        # 严格来说是 \int omega(t) dt
        gw_phase = 2.0 * phi_curr 
        
        h_val = factor * cmath.exp(-1.0j * gw_phase)
        
        times.append(t_curr)
        r_vals.append(r_curr)
        h_plus_vals.append(h_val.real)
        
        if r_curr < 2.5: # 接近视界/不可逆轨道
            print(f"Plunge detected at step {i}, r={r_curr:.4f}")
            break
            
        if i % 100 == 0:
            print(f"Step {i}: r={r_curr:.4f}, Omega={Omega_phi:.4f}, nu(2,2)={current_nu_dict.get((2,2))}")

    # 绘图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    ax1.plot(times, h_plus_vals, color='tab:blue')
    ax1.set_ylabel(r'Strain $h_+$')
    ax1.set_title(f'Inspiral Waveform (Chirp), a={a}')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(times, r_vals, color='tab:orange', linewidth=2)
    ax2.set_ylabel(r'Orbit Radius $r$ ($M$)')
    ax2.set_xlabel(r'Time ($M$)')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('inspiral_chirp.png')
    print("✅ Simulation complete. Results saved to 'inspiral_chirp.png'")

if __name__ == "__main__":
    simulate_inspiral()