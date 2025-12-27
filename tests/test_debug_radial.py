import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import cmath

# 路径设置，确保能导入 GremLinEqRe
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from GremLinEqRe import _core
except ImportError:
    print("❌ Error: 无法导入 C++ 模块 GremLinEqRe._core")
    sys.exit(1)

def evaluate_teukolsky_equation_residual(tr, r, R, dR):
    """
    计算 Teukolsky 径向方程的残差。
    方程形式: Delta^{-s} d/dr(Delta^{s+1} dR/dr) + V R = 0
    残差定义为: |R''_calc - R''_ode| / |R''_ode|
    其中 R''_calc 是通过 dR 的数值差分得到的，
    R''_ode 是通过方程 R'' = -1/Delta * (...) 算出的。
    """
    # 1. 使用 C++ 模块内部实现的方程逻辑计算理论上的 R''
    # tr.evaluate_ddR 返回的是根据方程推出的 d^2R/dr^2
    ddR_ode = tr.evaluate_ddR(r, R, dR)
    return ddR_ode

def compute_numerical_derivative(x_vals, y_vals):
    """使用中心差分计算数值导数"""
    dy = np.zeros_like(y_vals, dtype=complex)
    # 内部点: 中心差分
    dy[1:-1] = (y_vals[2:] - y_vals[:-2]) / (x_vals[2:] - x_vals[:-2])
    # 边界点: 前向/后向差分
    dy[0] = (y_vals[1] - y_vals[0]) / (x_vals[1] - x_vals[0])
    dy[-1] = (y_vals[-1] - y_vals[-2]) / (x_vals[-1] - x_vals[-2])
    return dy

def analyze_errors():
    print("=========================================================")
    print("   Teukolsky Radial Solver Error Analysis Tool")
    print("=========================================================")

    # 1. 参数设置 (选择一个典型但不极端的参数)
    M = 1.0
    a = 0.9      # 高自旋，更能暴露问题
    omega = 0.3  # 频率
    s = -2
    l = 2
    m = 2

    print(f"Parameters: a={a}, omega={omega}, s={s}, l={l}, m={m}")

    # 2. 初始化
    swsh = _core.SWSH(s, l, m, a * omega)
    lambda_val = swsh.m_lambda
    print(f"Lambda: {lambda_val:.10f}")

    tr = _core.TeukolskyRadial(M, a, omega, s, l, m, lambda_val)

    # 3. 求解 nu
    print("Solving nu...")
    nu = tr.solve_nu(complex(float(l), 0.0))
    print(f"Solved nu: {nu}")

    # 4. 计算系数 (使用非常高的 n_max 以排除级数截断误差)
    n_max = 150
    print(f"Computing coefficients (n_max={n_max})...")
    a_coeffs_pos = tr.ComputeSeriesCoefficients(nu, n_max)
    nu_neg = -nu - 1.0
    a_coeffs_neg = tr.ComputeSeriesCoefficients(nu_neg, n_max)

    # 5. 计算 K 因子
    print("Computing K factors...")
    K_pos = tr.k_factor(nu)
    K_neg = tr.k_factor(nu_neg)
    print(f"K(nu)    : {K_pos:.6e}")
    print(f"K(-nu-1) : {K_neg:.6e}")

    # 6. 设置扫描网格
    # 重点关注 r_+ 到 远场
    r_plus = 1.0 + np.sqrt(1.0 - a**2)
    kappa = np.sqrt(1.0 - a**2)
    r_match_guess = r_plus + 1.5 * kappa # 猜测的匹配点

    r_grid = np.linspace(r_plus + 0.01, r_plus + 4.0, 500)
    
    # 存储结果
    R_near_vals = []
    dR_near_vals = []
    R_far_vals = []
    dR_far_vals = [] # Far solution constructed via K factors

    print("Evaluating solutions on grid...")
    for r in r_grid:
        # A. Near Solution (Hypergeometric)
        # 即使发散也计算，为了看残差何时爆炸
        try:
            val, der = tr.Evaluate_Hypergeometric(r, nu, a_coeffs_pos)
            R_near_vals.append(val)
            dR_near_vals.append(der)
        except Exception:
            R_near_vals.append(complex(np.nan, np.nan))
            dR_near_vals.append(complex(np.nan, np.nan))

        # B. Far Solution (Coulomb Combined)
        # R_far = K_pos * R_C(nu) + K_neg * R_C(-nu-1)
        try:
            res_c1 = tr.Evaluate_Coulomb(r, nu, a_coeffs_pos)
            res_c2 = tr.Evaluate_Coulomb(r, nu_neg, a_coeffs_neg)
            
            val_f = K_pos * res_c1[0] + K_neg * res_c2[0]
            der_f = K_pos * res_c1[1] + K_neg * res_c2[1]
            
            R_far_vals.append(val_f)
            dR_far_vals.append(der_f)
        except Exception:
            R_far_vals.append(complex(np.nan, np.nan))
            dR_far_vals.append(complex(np.nan, np.nan))

    R_near_vals = np.array(R_near_vals)
    dR_near_vals = np.array(dR_near_vals)
    R_far_vals = np.array(R_far_vals)
    dR_far_vals = np.array(dR_far_vals)

    # ==========================================
    # 诊断 1: 方程残差测试 (ODE Consistency)
    # ==========================================
    # 我们通过数值微分 R' 得到 R''_num，并与 evaluate_ddR 比较
    # 或者，直接将 R, R' 代入 evaluate_ddR 得到的 R'' 是否与 数值微分一致
    
    # 计算数值二阶导数
    ddR_near_num = compute_numerical_derivative(r_grid, dR_near_vals)
    ddR_far_num  = compute_numerical_derivative(r_grid, dR_far_vals)
    
    ode_resid_near = []
    ode_resid_far = []

    for i, r in enumerate(r_grid):
        # Check Near
        if not np.isnan(R_near_vals[i]):
            # 理论上的 R''
            ddR_thy = tr.evaluate_ddR(r, R_near_vals[i], dR_near_vals[i])
            # 误差 = |R''_thy - R''_num|
            err = abs(ddR_thy - ddR_near_num[i])
            # 归一化误差
            scale = abs(ddR_thy) + 1e-20
            ode_resid_near.append(err / scale)
        else:
            ode_resid_near.append(np.nan)
            
        # Check Far
        if not np.isnan(R_far_vals[i]):
            ddR_thy = tr.evaluate_ddR(r, R_far_vals[i], dR_far_vals[i])
            err = abs(ddR_thy - ddR_far_num[i])
            scale = abs(ddR_thy) + 1e-20
            ode_resid_far.append(err / scale)
        else:
            ode_resid_far.append(np.nan)

    # ==========================================
    # 诊断 2: 匹配区域比值测试 (The Ratio Test)
    # ==========================================
    # 如果两个解都满足方程（残差低），但它们不重合，
    # 那么一定是 K_nu 计算错了，或者是 Evaluate_Coulomb 差了常数因子。
    
    ratios = []
    phases = []
    valid_mask = []
    
    for i, r in enumerate(r_grid):
        v_n = R_near_vals[i]
        v_f = R_far_vals[i]
        
        if np.isnan(v_n) or np.isnan(v_f) or abs(v_n) < 1e-15 or abs(v_f) < 1e-15:
            ratios.append(np.nan)
            phases.append(np.nan)
            valid_mask.append(False)
        else:
            ratio = v_n / v_f
            ratios.append(abs(ratio))
            # 相对相位差 (弧度)
            phases.append(np.angle(ratio)) 
            valid_mask.append(True)

    # ==========================================
    # 绘图
    # ==========================================
    fig, axes = plt.subplots(3, 1, figsize=(10, 15), sharex=True)

    # Plot 1: ODE Residuals (Check correctness of individual solutions)
    ax = axes[0]
    ax.semilogy(r_grid, ode_resid_near, 'r-', label='Near Solution Residual')
    ax.semilogy(r_grid, ode_resid_far, 'b-', label='Far Solution Residual')
    ax.set_title(f'ODE Consistency Check (Is the solution valid?) a={a}')
    ax.set_ylabel('|Residual|')
    ax.legend()
    ax.grid(True)
    ax.axvline(r_match_guess, color='k', linestyle=':', label='Est. Match')

    # Plot 2: Matching Amplitude Ratio (Near / Far)
    ax = axes[1]
    ax.plot(r_grid, ratios, 'k-', lw=2, label='|R_near / R_far|')
    ax.axhline(1.0, color='r', linestyle='--')
    ax.axhline(2.0, color='g', linestyle='--', label='Factor of 2?')
    ax.axhline(0.5, color='g', linestyle='--')
    ax.set_title('Amplitude Ratio (Should be 1.0)')
    ax.set_ylim(0, 3) # 重点关注 1.0 附近
    ax.legend()
    ax.grid(True)
    
    # Plot 3: Matching Phase Difference
    ax = axes[2]
    ax.plot(r_grid, phases, 'k-', lw=2, label='Arg(R_near / R_far)')
    ax.axhline(0.0, color='r', linestyle='--')
    ax.set_title('Phase Difference (Radians) (Should be 0.0)')
    ax.set_xlabel('r/M')
    ax.grid(True)

    plt.tight_layout()
    output_name = f"error_analysis_a{a}_w{omega}.png"
    plt.savefig(output_name)
    print(f"Analysis plot saved to {output_name}")

    # ==========================================
    # 自动结论输出
    # ==========================================
    # 寻找最佳匹配点（残差均较小的区域）
    best_idx = -1
    min_combined_resid = 1e9
    
    for i in range(len(r_grid)):
        if ode_resid_near[i] < 1e-5 and ode_resid_far[i] < 1e-5:
            comb = ode_resid_near[i] + ode_resid_far[i]
            if comb < min_combined_resid:
                min_combined_resid = comb
                best_idx = i
                
    if best_idx != -1:
        r_best = r_grid[best_idx]
        ratio_best = ratios[best_idx]
        phase_best = phases[best_idx]
        
        print("\n--- Diagnosis at Optimal Match Point (r = {:.4f}) ---".format(r_best))
        print(f"ODE Residual Near: {ode_resid_near[best_idx]:.2e}")
        print(f"ODE Residual Far : {ode_resid_far[best_idx]:.2e}")
        print(f"Amplitude Ratio  : {ratio_best:.6f} (Target: 1.0)")
        print(f"Phase Difference : {phase_best:.6f} rad (Target: 0.0)")
        
        if abs(ratio_best - 2.0) < 0.1:
            print("🚨 警告: 振幅比接近 2.0，疑似 Evaluate_Coulomb 缺少因子 2")
        elif abs(ratio_best - 0.5) < 0.05:
             print("🚨 警告: 振幅比接近 0.5，疑似 Evaluate_Coulomb 多了因子 2")
        elif abs(ratio_best - 1.0) > 0.1:
            print("🚨 警告: 振幅严重不匹配，检查 K_nu 收敛性或归一化")
            
        if abs(phase_best) > 0.1:
            print("🚨 警告: 相位不匹配，检查 K_nu 中的相位项或 Evaluate_Coulomb 的相位定义")
    else:
        print("\n❌ 未找到两者同时收敛的区域 (Overlap region not found)")

if __name__ == "__main__":
    analyze_errors()