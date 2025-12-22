import sys
import os
import math
import cmath
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from scipy.interpolate import CubicSpline  
# ============================================================
# 导入 C++ 核心模块 GremLinEqRe._core
# ============================================================
this_dir = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(this_dir, '.')))
sys.path.append(os.path.abspath(os.path.join(this_dir, '..')))

try:
    from GremLinEqRe import _core
except ImportError:
    print("❌ Error: 无法导入 C++ 模块 GremLinEqRe")
    sys.exit(1)

# ============================================================
# 物理常数 (SI Units)
# ============================================================
G_SI = 6.67430e-11
c_SI = 2.99792458e8
M_SUN_SI = 1.989e30

PC_SI = 3.0857e16        # 1 pc
MPC_SI = 1e6 * PC_SI     # 1 Mpc
YEAR_SI = 365.25 * 24.0 * 3600.0  # 1 year in seconds

# ============================================================
# 波形生成器类 (多模式叠加) —— 来自 inspiral 的框架
# ============================================================
class WaveformGenerator:
    """
    使用 (l, m, omega, Z_inf) 列表计算 h = h_+ - i h_x ，
    并乘上 G mu / (c^2 D) 的物理因子。
    """
    def __init__(self, M_BH_mass, mu_mass, distance, spin_a):
        self.M_BH = M_BH_mass
        self.mu = mu_mass
        self.dist = distance
        self.a = spin_a

        # Strain 量纲：h = (G mu / c^2 D) * h_code
        self.amp_scale = (G_SI * self.mu) / (c_SI**2 * self.dist)

    def get_swsh_val(self, s, l, m, a_omega, theta_obs):
        """计算自旋加权球谐 S_{lm}^s(theta)。"""
        x = math.cos(theta_obs)
        swsh = _core.SWSH(s, l, m, a_omega)
        return swsh.evaluate_S(x)

    def compute_h_complex(self, mode_data_list, theta_obs, phi_obs, phase_orb_m1):
        """
        mode_data_list: list of dicts
            {'l': int, 'm': int, 'omega': float, 'Z': complex}
        theta_obs: 观测者极角 θ
        phi_obs:   观测者方位角 φ
        phase_orb_m1: 轨道相位 (m=1 模式的相位) = φ_orb(t)
        """
        h_total = 0.0j

        for data in mode_data_list:
            l = data['l']
            m = data['m']
            w = data['omega']
            Z = data['Z']

            # 1. 基础因子 h_lm ~ -2 Z / w^2
            factor = -2.0 * Z / (w**2)

            # 2. 角度部分
            # S_val = self.get_swsh_val(-2, l, m, self.a * w, theta_obs)
            S_val = data['S_val']
            # 3. 相位 m * φ_orb - m * φ_obs
            phase_gw = m * phase_orb_m1
            total_phase = phase_gw - m * phi_obs

            # 4. 单模波形
            h_lm = factor * S_val * cmath.exp(-1.0j * total_phase)

            # 5. 累加
            h_total += h_lm

            # # 6. 可选：利用 h_{l,-m} = (-1)^l h_{l,m}^* 做对称模式
            # if abs(theta_obs - math.pi/2) < 1e-3:  # 赤道面近似
            #     h_minus_m = ((-1)**l) * h_lm.conjugate()
            #     h_total += h_minus_m

        return h_total * self.amp_scale


# ============================================================
# 辅助函数: Kerr 圆轨道能量导数 dE/dr
# ============================================================
def get_circular_orbit_energy_deriv(a, r):
    """
    计算 Kerr 圆轨道能量 E 关于 r 的导数 dE/dr (几何单位, M=1)。
    对应标准公式 E(r,a)，再对 r 求导。
    """
    v = r**(-0.5)
    v2 = v * v
    v3 = v2 * v

    numer = 1.0 - 2.0 * v2 + a * v3
    denom_sq = 1.0 - 3.0 * v2 + 2.0 * a * v3
    denom = math.sqrt(denom_sq)

    # dE/dv
    d_num_dv = -4.0 * v + 3.0 * a * v2
    d_den_dv = 0.5 / denom * (-6.0 * v + 6.0 * a * v2)

    dE_dv = (d_num_dv * denom - numer * d_den_dv) / denom_sq
    dv_dr = -0.5 * v3

    return dE_dv * dv_dr
def generate_active_modes(l_max):
    """
    根据 l_max 自动生成 (l, m) 列表:
        l = 2,...,l_max
        m = 1,...,l
    对于当前圆赤道轨道，只需要 m>0；负 m 可由对称关系补。
    """
    modes = []
    for l in range(2, l_max + 1):
        for m in range(1, l + 1):
            modes.append((l, m))
    return modes
def _compute_single_mode_worker(args):
    """
    单个 (l, m) 模式的 Teukolsky + MST 计算，用于并行调用。

    args: (a, r, l, m, nu_guess)
    返回: (l, m, omega, Z_inf, flux_mode, nu_new)
    """
    a, r, l, m, nu_guess = args
    M = 1.0

    # 1. 轨道：在 worker 内重建 geo/state
    geo = _core.KerrGeo.from_circular_equatorial(a, r, True)
    state = _core.KerrGeo.State()
    state.x = [0.0, r, math.pi/2.0, 0.0]
    geo.update_kinematics(state)
    Omega_phi = state.u[3] / state.u[0]

    omega = m * Omega_phi
    if omega <= 0.0:
        # 这个模式没有物理意义，直接返回零贡献
        return (l, m, omega, 0.0j, 0.0, nu_guess)

    # 2. SWSH & lambda
    swsh = _core.SWSH(-2, l, m, a * omega)
    try:
        lam = swsh.m_lambda
    except Exception:
        lam = (l * (l + 1) - 2.0 * m * a * omega + (a * omega)**2 - 2.0)

    # 3. 径向 Teukolsky
    tr = _core.TeukolskyRadial(M, a, omega, -2, l, m, lam)
    tr.ResetCalibration()

    # nu 初值
    nu = tr.solve_nu(nu_guess)

    # 4. MST 系数
    n_max = 60
    coeffs_pos = tr.ComputeSeriesCoefficients(nu, n_max)
    coeffs_neg = tr.ComputeSeriesCoefficients(-nu - 1.0, n_max)
    K_pos = tr.k_factor(nu)
    K_neg = tr.k_factor(-nu - 1.0)

    # 5. 匹配半径, R, dR, ddR
    kappa = math.sqrt(1.0 - a**2)
    r_plus = 1.0 + kappa
    r_match = r_plus + 1.5 * kappa

    R_val, dR_val = tr.Evaluate_R_in(
        r, nu, K_pos, K_neg, coeffs_pos, coeffs_neg, r_match
    )
    try:
        ddR_val = tr.evaluate_ddR(r, R_val, dR_val)
    except Exception:
        ddR_val = 0.0

    # 6. 源项投影 + 物理振幅
    ts = _core.TeukolskySource(a, omega, -2, int(l), m)
    proj = ts.ComputeProjections(state, geo, swsh)

    amps_pos = tr.ComputeAmplitudes(nu, coeffs_pos)
    phys = tr.ComputePhysicalAmplitudes(nu, coeffs_pos, amps_pos)

    W = R_val * (proj.A_nn0 + proj.A_mbarn0 + proj.A_mbarmbar0) \
        - dR_val * (proj.A_mbarn1 + proj.A_mbarmbar1) \
        + ddR_val * (proj.A_mbarmbar2)

    if abs(phys.B_inc) < 1e-15:
        Z_inf = 0.0j
        flux_mode = 0.0
    else:
        Z_inf = (2.0 * math.pi * W) / (2.0j * omega * phys.B_inc)
        # m 和 -m 两个模式的能量通量
        flux_mode = 2.0 * (abs(Z_inf)**2) / (4.0 * math.pi * omega**2)

    return (l, m, omega, Z_inf, flux_mode, nu)


# ============================================================
# 核心一步：给定 r，算多模能量通量和每个模式的 Z_inf
# ============================================================
# def compute_step_physics(a, r, active_modes, last_nu_dict=None, executor=None):
#     """
#     计算当前半径下：
#       1. 总能量通量 flux_E (几何单位, mu=1)
#       2. 每个 (l,m) 模式的 omega, Z_inf
#       3. 更新后的 nu 初值字典
#       4. 轨道 azimuthal 频率 Omega_phi

#     返回:
#       total_flux_E, modes_data, next_nu_dict, Omega_phi
#     其中 modes_data[(l,m)] = {'l', 'm', 'omega', 'Z_inf'}
#     """
#     M = 1.0

#     # 1. 更新几何量 (赤道圆轨道)
#     geo = _core.KerrGeo.from_circular_equatorial(a, r, True)
#     state = _core.KerrGeo.State()
#     state.x = [0.0, r, math.pi/2.0, 0.0]
#     geo.update_kinematics(state)

#     Omega_phi = state.u[3] / state.u[0]

#     total_flux_E = 0.0
#     modes_data = {}
#     next_nu_dict = {}

#     # 2. 遍历 (l,m) 模式
#     for l, m in active_modes:
#         omega = m * Omega_phi
#         if omega <= 0.0:
#             continue

#         # A. 自旋加权球谐 + λ
#         swsh = _core.SWSH(-2, l, m, a * omega)
#         try:
#             lam = swsh.m_lambda
#         except Exception:
#             lam = (l * (l + 1) - 2.0 * m * a * omega + (a * omega)**2 - 2.0)

#         # B. 径向求解器
#         tr = _core.TeukolskyRadial(M, a, omega, -2, l, m, lam)
#         tr.ResetCalibration()

#         # C. 求解 ν（热启动）
#         nu_guess = complex(float(l), 0.0)
#         if last_nu_dict and (l, m) in last_nu_dict:
#             nu_guess = last_nu_dict[(l, m)]
#         nu = tr.solve_nu(nu_guess)
#         next_nu_dict[(l, m)] = nu

#         # D. MST 系数
#         n_max = 60
#         coeffs_pos = tr.ComputeSeriesCoefficients(nu, n_max)
#         coeffs_neg = tr.ComputeSeriesCoefficients(-nu - 1.0, n_max)
#         K_pos = tr.k_factor(nu)
#         K_neg = tr.k_factor(-nu - 1.0)

#         # E. 选择匹配半径，计算 R, dR, ddR
#         kappa = math.sqrt(1.0 - a**2)
#         r_plus = 1.0 + kappa
#         r_match = r_plus + 1.0 * kappa

#         R_val, dR_val = tr.Evaluate_R_in(
#             r, nu, K_pos, K_neg, coeffs_pos, coeffs_neg, r_match
#         )

#         try:
#             ddR_val = tr.evaluate_ddR(r, R_val, dR_val)
#         except Exception:
#             # 如果没有实现二阶导接口，就先忽略这项
#             ddR_val = 0.0

#         # F. 源项投影 + 物理振幅
#         ts = _core.TeukolskySource(a, omega, -2, int(l), m)
#         proj = ts.ComputeProjections(state, geo, swsh)

#         amps_pos = tr.ComputeAmplitudes(nu, coeffs_pos)
#         phys = tr.ComputePhysicalAmplitudes(nu, coeffs_pos, amps_pos)

#         # W 源项 (与你之前脚本里一致)
#         W = R_val * (proj.A_nn0 + proj.A_mbarn0 + proj.A_mbarmbar0) \
#             - dR_val * (proj.A_mbarn1 + proj.A_mbarmbar1) \
#             + ddR_val * (proj.A_mbarmbar2)

#         # G. 无穷远振幅 Z_inf
#         if abs(phys.B_inc) < 1e-15:
#             Z_inf = 0.0j
#         else:
#             Z_inf = (2.0 * math.pi * W) / (2.0j * omega * phys.B_inc)

#         # H. 模式能量通量 dE/dt
#         flux_mode = 2.0 * (abs(Z_inf)**2) / (4.0 * math.pi * omega**2)
#         total_flux_E += flux_mode

#         modes_data[(l, m)] = {
#             "l": l,
#             "m": m,
#             "omega": omega,
#             "Z_inf": Z_inf,
#         }

#     return total_flux_E, modes_data, next_nu_dict, Omega_phi
def build_flux_interpolator(a, r_min, r_max, active_modes, num_points=50, n_workers=4):
    """
    在 [r_min, r_max] 范围内生成稀疏网格，并行计算通量和波形振幅，
    并返回插值器对象。
    """
    print(f"\nBuilding flux interpolators from r={r_min:.3f} to {r_max:.3f} with {num_points} points...")
    
    # 1. 生成网格节点 (可以使用对数分布，因为靠近 ISCO 变化快)
    #    这里简单使用线性分布，稍微向 ISCO 加密
    #    注意：演化是从 r_max -> r_min
    r_nodes = np.linspace(r_min, r_max, num_points)
    
    # 2. 准备并行任务
    #    compute_step_physics 原本设计为计算单步，我们可以复用它内部的逻辑，
    #    但为了并行效率，直接复用 _compute_single_mode_worker 更直接。
    #    我们需要对每个 r 节点上的每个 mode 都建立任务。
    
    tasks = []
    # 任务结构: (r_index, mode_index, args)
    # 我们需要记录索引以便重组数据
    
    for i, r in enumerate(r_nodes):
        for j, (l, m) in enumerate(active_modes):
            # nu_guess 简单给个初值，预计算时不强求热启动，或者可以用 l
            nu_guess = complex(float(l), 0.0) 
            tasks.append( (a, r, l, m, nu_guess) )

    print(f"  Total tasks: {len(tasks)} (Teukolsky solves)")
    
    # 3. 并行计算
    #    注意：_compute_single_mode_worker 需要是顶层函数才能被 pickle
    results = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # 使用 tqdm 显示预计算进度
        results = list(tqdm(executor.map(_compute_single_mode_worker, tasks), 
                            total=len(tasks), desc="  Pre-computing Flux grid"))

    # 4. 重组数据
    #    Flux_total(r)
    #    Z_lm(r) 
    flux_grid = np.zeros(num_points)
    #    存储结构: Z_data[mode_idx][r_idx]
    Z_grids = { (l,m): np.zeros(num_points, dtype=complex) for l,m in active_modes }
    omega_grids = { (l,m): np.zeros(num_points) for l,m in active_modes }

    #    因为并行返回是乱序的或者列表形式，我们需要根据输入参数重新匹配吗？
    #    executor.map 保证返回顺序与输入顺序一致。
    #    输入顺序是：外层 r，内层 mode
    
    ptr = 0
    for i in range(num_points):
        step_flux = 0.0
        for j, (l, m) in enumerate(active_modes):
            res = results[ptr]
            ptr += 1
            
            # 解包: (l, m, omega, Z_inf, flux_mode, nu_new) = res
            _, _, omega, Z_inf, flux_mode, _ = res
            
            step_flux += flux_mode
            Z_grids[(l,m)][i] = Z_inf
            omega_grids[(l,m)][i] = omega
            
        flux_grid[i] = step_flux

    # 5. 构建插值器 (CubicSpline)
    #    注意：CubicSpline 要求 x 单调递增
    interp_flux = CubicSpline(r_nodes, flux_grid)
    
    interp_modes = {}
    for (l,m) in active_modes:
        interp_modes[(l,m)] = {
            'Z': CubicSpline(r_nodes, Z_grids[(l,m)]),
            'omega': CubicSpline(r_nodes, omega_grids[(l,m)]) 
            # Omega 也可以插值，或者直接由 r 解析计算。
            # 为了保持一致性，插值 omega 也是可以的，或者用解析公式 Omega_phi(r)
        }
        
    print("  Interpolators built successfully.")
    return interp_flux, interp_modes
def compute_step_physics(a, r,theta_obs, active_modes, last_nu_dict=None, executor=None):
    """
    计算当前半径下 总能量通量 + 每个模式的数据。
    如果 executor 为 None，则串行；否则用 executor 并行。
    """
    M = 1.0

    # 先单独算一次 Omega_phi，用于返回给主程序
    geo = _core.KerrGeo.from_circular_equatorial(a, r, True)
    state = _core.KerrGeo.State()
    state.x = [0.0, r, math.pi/2.0, 0.0]
    geo.update_kinematics(state)
    Omega_phi = state.u[3] / state.u[0]

    total_flux_E = 0.0
    modes_data = {}
    next_nu_dict = {}

    # 1) 准备任务列表：每个元素是 (a, r, l, m, nu_guess)
    task_args = []
    for l, m in active_modes:
        # nu 初值
        if last_nu_dict and (l, m) in last_nu_dict:
            nu_guess = last_nu_dict[(l, m)]
        else:
            nu_guess = complex(float(l), 0.0)

        task_args.append((a, r, l, m, nu_guess))

    # 2) 串行 or 并行执行
    if executor is None or len(task_args) == 1:
        results = [_compute_single_mode_worker(arg) for arg in task_args]
    else:
        results = list(executor.map(_compute_single_mode_worker, task_args))

    # 3) 汇总结果
    for (l, m, omega, Z_inf, flux_mode, nu_new) in results:
        total_flux_E += flux_mode
        next_nu_dict[(l, m)] = nu_new
        swsh_temp = _core.SWSH(-2, l, m, a * omega)
        S_val_cached = swsh_temp.evaluate_S(math.cos(theta_obs))
        modes_data[(l, m)] = {
            "l": l,
            "m": m,
            "omega": omega,
            "Z_inf": Z_inf,
            "S_val": S_val_cached,
        }

    return total_flux_E, modes_data, next_nu_dict, Omega_phi



# ============================================================
# 主程序: 多模式 EMRI 演化 + 波形 (SI 单位输出)
# ============================================================
def simulate_inspiral_multimode(
    M_Msun=1e6,
    mu_Msun=10.0,
    T_years=0.5,
    dt_seconds=1.0,
    spin_a=0.8,
    r0=10.0,
    distance_Mpc=100.0,
    l_max=4,
    theta_obs=math.pi / 2.0,
    phi_obs=0.0,
):
    """
    多模式 EMRI 绝热旋近波形生成.

    参数:
      M_Msun       : 大质量黑洞质量 (单位: M_sun)
      mu_Msun      : 小天体质量 (单位: M_sun)
      T_years      : 总演化时间 (单位: 年)
      dt_seconds   : 采样时间步长 (单位: 秒)
      spin_a       : Kerr 自旋参数 a (0 <= a < 1)
      r0           : 初始圆轨道半径 (单位: M)
      distance_Mpc : 源到地球距离 (单位: Mpc)
      active_modes : 要叠加的 (l,m) 列表。如果为 None，使用默认几个主模
      theta_obs    : 观测者极角
      phi_obs      : 观测者方位角
    """

    # if active_modes is None:
    #     # 你可以根据需要增减模式
    #     active_modes = [(2, 2), (2, 1), (3, 3), (3, 2), (4, 4)]
    # ---------- 模式列表 ----------
    active_modes = generate_active_modes(l_max)
    print(f"Modes (l <= {l_max}): {active_modes}")


    # ---------- 质量、时间尺度 ----------
    M_BH_mass = M_Msun * M_SUN_SI
    mu_mass = mu_Msun * M_SUN_SI
    Distance = distance_Mpc * MPC_SI

    # 时间、长度尺度 (几何单位 M ↔ SI)
    T_scale = (G_SI * M_BH_mass) / (c_SI**3)   # 1 M 对应的秒数
    # L_scale = (G_SI * M_BH_mass) / (c_SI**2) # 1 M 对应的米数 (目前用不上)

    # # 1. 确定插值范围
    # #    r_start = r0
    # #    r_end   = ISCO + buffer (例如 ISCO + 0.1M)
    # #    ISCO 近似计算或直接给个小值，比如 r_hor + 0.5
    # r_hor = 1.0 + math.sqrt(1.0 - spin_a**2)
    # r_isco_approx = r_hor + 1.0 # 简单估算，或者设为 r_hor + 0.2
    
    # #    预计算范围 [r_end, r_start]。注意 linspace 需要从小到大，build函数内部处理了
    # num_interp_points = 60 # 60个点足够覆盖平滑变化
    
    # interp_flux, interp_modes = build_flux_interpolator(
    #     spin_a, r_isco_approx, r0, active_modes, 
    #     num_points=num_interp_points, n_workers=n_workers
    # )
    mass_ratio_q = mu_mass / M_BH_mass

    print("=========================================================")
    print("   Multi-Mode EMRI Adiabatic Inspiral (Teukolsky+MST)   ")
    print("=========================================================")
    print(f"M = {M_Msun:.3e} M_sun,  mu = {mu_Msun:.2f} M_sun,  a = {spin_a:.3f}")
    print(f"Distance = {distance_Mpc:.1f} Mpc")
    print(f"T_scale = {T_scale:.4f} s  (1 M in seconds)")
    print(f"Initial radius r0 = {r0:.3f} M")
    print(f"Modes: {active_modes}")
    print("---------------------------------------------------------")

    # ---------- 时间步数 ----------
    total_T_sec = T_years * YEAR_SI
    if dt_seconds <= 0.0:
        raise ValueError("dt_seconds 必须为正。")

    N_steps = int(total_T_sec / dt_seconds)
    if N_steps < 1:
        N_steps = 1

    if N_steps > 200000:
        print(f"⚠️ 警告: N_steps ≈ {N_steps:d}, 计算可能非常耗时。")
        print("   可以考虑增大 dt_seconds 或减小 T_years。")

    # 几何单位的时间步长
    dt_geo = dt_seconds / T_scale

    # ---------- 初始条件 ----------
    r_curr = r0
    phi_curr = 0.0
    t_geo = 0.0

    # 数据记录
    times_si = []
    r_vals = []
    h_plus = []
    h_cross = []

    # 缓存
    current_nu_dict = {}
    cached_flux = 0.0
    cached_modes = {}
    cached_Omega = 0.0

    # 波形生成器
    wf_gen = WaveformGenerator(M_BH_mass, mu_mass, Distance, spin_a)

    print("\nStarting evolution...")
    recalc_every = 10000  # 每隔多少步刷新一次 Teukolsky 通量 (绝热近似)
    # 这里可以根据 CPU 核数设定 worker 数
    n_workers = 4  # 或者用 os.cpu_count()
    # 创建进度条对象，便于后面更新 postfix

    
    pbar = tqdm(range(N_steps), desc="Evolving inspiral", dynamic_ncols=True)
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        for i in pbar:

            # -------- A. 重新计算通量与模式振幅 --------
            if i % recalc_every == 0:
                try:
                    flux_E, modes_data, nu_dict, Omega_phi = compute_step_physics(
                        spin_a, r_curr,theta_obs, active_modes, current_nu_dict,
                        executor=executor,
                    )
                    cached_flux = flux_E
                    cached_modes = modes_data
                    current_nu_dict = nu_dict
                    cached_Omega = Omega_phi
                except Exception as e:
                    print(f"⚠️ Solver exception at step {i}, r={r_curr:.4f}: {e}")
                    break

            # -------- B. 轨道演化 (dr/dt) --------
            dE_dr_geo = get_circular_orbit_energy_deriv(spin_a, r_curr)
            if abs(dE_dr_geo) < 1e-10:
                dE_dr_geo = math.copysign(1e-10, dE_dr_geo if dE_dr_geo != 0 else -1.0)

            # 注意：flux_E 是 mu=1 的几何通量 → 实际通量 ∝ q^2
            dr_dt = - (cached_flux * mass_ratio_q) / dE_dr_geo

            r_curr += dr_dt * dt_geo
            phi_curr += cached_Omega * dt_geo
            t_geo += dt_geo

            # -------- C. 多模波形叠加 --------
            mode_list_for_wf = []
            for (l, m), md in cached_modes.items():
                mode_list_for_wf.append({
                    "l": l,
                    "m": m,
                    "omega": md["omega"],
                    "Z": md["Z_inf"],
                    "S_val": md["S_val"],
                })

            h_complex = wf_gen.compute_h_complex(
                mode_list_for_wf,
                theta_obs=theta_obs,
                phi_obs=phi_obs,
                phase_orb_m1=phi_curr
            )

            # 记录 SI 时间和波形
            times_si.append(t_geo * T_scale)
            r_vals.append(r_curr)
            h_plus.append(h_complex.real)
            h_cross.append(-h_complex.imag)
            if i % 10 == 0:  # 每 10 步更新一次，避免太频繁
                pbar.set_postfix({
                "r/M": f"{r_curr:7.3f}",
                "|h|": f"{abs(h_complex):.2e}",
                "Omega": f"{cached_Omega:.3e}"
            })
            # -------- D. 终止条件：接近视界 (近似 plunge) --------
            r_hor = 1.0 + 1.1 * math.sqrt(1.0 - spin_a**2)
            if r_curr < r_hor + 0.5:
                print(f"Plunge detected at r = {r_curr:.4f}. Stop evolution.")
                break

            # if i % max(1, N_steps // 20) == 0:
            #     print(f"Step {i:6d} / {N_steps:6d} : "
            #           f"t = {t_geo:9.1f} M, r = {r_curr:7.3f}, |h| = {abs(h_complex):.3e}")

    # ========================================================
    # 绘图: 上图 h_+(t)，下图 r(t)
    # ========================================================
    print("\nPlotting results...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=False)

    # 上图: h_+
    ax1.plot(times_si, h_plus, color="#1f77b4", linewidth=1.0, label=r"$h_+$")
    ax1.set_ylabel(r"Strain $h_+$")
    ax1.set_title(
        rf"Multi-mode EMRI waveform ($a={spin_a:.2f}$, "
        rf"$M={M_Msun:.1e}M_\odot$, $\mu={mu_Msun:.1f}M_\odot$)"
    )
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend(loc="upper right")

    # 小插图：晚期演化
    if len(times_si) > 500:
        inset = ax1.inset_axes([0.6, 0.6, 0.35, 0.35])
        start_idx = len(times_si) - 500
        inset.plot(times_si[start_idx:], h_plus[start_idx:], color="#d62728", linewidth=1.0)
        inset.set_title("Late inspiral", fontsize=9)
        inset.grid(True, alpha=0.3)

    # 下图: 半径演化
    ax2.plot(times_si, r_vals, color="#ff7f0e", linewidth=2.0)
    ax2.set_xlabel("Time (seconds)")
    ax2.set_ylabel(r"Orbital radius $r/M$")
    ax2.set_title("Orbital decay")
    ax2.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    output_name = "emri_waveform_si.png"
    plt.savefig(output_name, dpi=300)
    print(f"✅ Done. Results saved to {output_name}")


if __name__ == "__main__":
    # 这里可以直接改成你想要的参数
    simulate_inspiral_multimode(
        M_Msun=1e6,      # 中心黑洞质量 (solar masses)
        mu_Msun=10.0,    # 小天体质量 (solar masses)
        T_years=2.0,     # 总演化时间 (years)
        dt_seconds=5.0,  # 采样步长 (seconds)
        spin_a=0.8,      # Kerr 自旋
        r0=6.0,         # 初始半径 (M)
        distance_Mpc=100.0,
        l_max=4,
        theta_obs=math.pi / 2.0,
        phi_obs=0.0,
    )
