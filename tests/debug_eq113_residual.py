import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# 1. 环境与路径设置
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from GremLinEqRe import _core
except ImportError:
    print("❌ Error: 无法导入 C++ 模块 GremLinEqRe._core")
    sys.exit(1)

# =========================================================================
# 2. 核心函数：Teukolsky 方程残差计算
# =========================================================================

def compute_teukolsky_residual(r_grid, R_vals, dR_vals, d2R_vals, params):
    """
    计算 LRR Eq. 113 的方程残差。
    方程形式: Delta R'' + (s+1) Delta' R' + V R = 0
    """
    M, a, omega, s, m, lambda_val = params
    residuals = []
    
    for i, r in enumerate(r_grid):
        R = R_vals[i]
        dR = dR_vals[i]
        ddR = d2R_vals[i] # 来自数值微分
        
        # 忽略无效值
        if np.isnan(R) or np.abs(R) < 1e-20:
            residuals.append(np.nan)
            continue

        # --- 物理量定义 ---
        # Delta = r^2 - 2Mr + a^2
        Delta = r**2 - 2*M*r + a**2
        
        # Delta' = 2r - 2M
        dDelta_dr = 2.0 * (r - M)
        
        # K = (r^2 + a^2)omega - am
        K = (r**2 + a**2) * omega - a * m
        
        # --- 方程各项 ---
        # Term 1: Delta * R''
        term1 = Delta * ddR
        
        # Term 2: (s+1) * Delta' * R'
        term2 = (s + 1.0) * dDelta_dr * dR
        
        # Term 3: V(r) * R
        # V = [K^2 - 2is(r-M)K] / Delta + 4is*omega*r - lambda
        numerator = K**2 - 2.0j * s * (r - M) * K
        V_potential = numerator / Delta + 4.0j * s * omega * r - lambda_val
        
        term3 = V_potential * R
        
        # --- 计算左边 (LHS) ---
        lhs = term1 + term2 + term3
        
        # 归一化残差 (相对于主导项或 R 的大小)
        # 这里除以 |R| 得到“方程每单位振幅的违背程度”
        residuals.append(abs(lhs) / abs(R))
        
    return np.array(residuals)

# =========================================================================
# 3. 主分析流程
# =========================================================================

def analyze_eq113():
    print("=========================================================")
    print("   Teukolsky Eq. 113 Residual Check (Full Equation)")
    print("=========================================================")

    # 1. 物理参数
    M = 1.0
    a = 0.9      
    omega = 0.3
    s = -2
    l = 2
    m = 2
    
    print(f"Parameters: a={a}, omega={omega}, s={s}, l={l}, m={m}")

    # 2. 求解特征值 lambda 和 nu
    swsh = _core.SWSH(s, l, m, a * omega)
    lambda_val = swsh.m_lambda
    print(f"Lambda: {lambda_val:.8f}")

    tr = _core.TeukolskyRadial(M, a, omega, s, l, m, lambda_val)
    
    print("Solving nu...")
    nu = tr.solve_nu(complex(float(l), 0.1))
    print(f"nu: {nu}")
    
    # 3. 计算级数系数
    n_max = 60
    a_coeffs = tr.ComputeSeriesCoefficients(nu, n_max)

    # 4. 定义扫描网格
    r_plus = 1.0 + np.sqrt(1.0 - a**2)
    r_grid = np.linspace(r_plus + 0.01, 20.0, 1000) # 覆盖近场到远场
    dr = r_grid[1] - r_grid[0]
    
    # 5. 计算近场解 (Hypergeometric)
    R_near = []
    dR_near = []
    
    print("Evaluating Near Field Solution...")
    for r in r_grid:
        try:
            val, der = tr.Evaluate_Hypergeometric(r, nu, a_coeffs)
            R_near.append(val)
            dR_near.append(der)
        except:
            R_near.append(np.nan)
            dR_near.append(np.nan)

    # 6. 计算远场解 (Coulomb)
    # 注意: Evaluate_Coulomb 返回的是 R_C^nu，它也是 Teukolsky 方程的解
    R_far = []
    dR_far = []
    
    print("Evaluating Far Field Solution...")
    for r in r_grid:
        try:
            val, der = tr.Evaluate_Coulomb(r, nu, a_coeffs)
            R_far.append(val)
            dR_far.append(der)
        except:
            R_far.append(np.nan)
            dR_far.append(np.nan)
            
    # 7. 数值微分计算 d2R
    # 使用 np.gradient (二阶中心差分精度)
    # R_near 和 R_far 都是 Teukolsky 方程的解，所以都应该满足方程
    
    # 处理 Near
    dR_near_arr = np.array(dR_near)
    # 这里通过对 dR 再次微分得到 d2R
    d2R_near = np.gradient(dR_near_arr, dr)
    
    # 处理 Far
    dR_far_arr = np.array(dR_far)
    d2R_far = np.gradient(dR_far_arr, dr)
    
    # 8. 代回 Eq. 113 计算残差
    params = (M, a, omega, s, m, lambda_val)
    
    res_near = compute_teukolsky_residual(r_grid, R_near, dR_near_arr, d2R_near, params)
    res_far = compute_teukolsky_residual(r_grid, R_far, dR_far_arr, d2R_far, params)
    
    # 9. 绘图
    plt.figure(figsize=(10, 8))
    
    plt.semilogy(r_grid, res_near, 'r-', linewidth=2, label='Near (Hypergeo) Residual')
    plt.semilogy(r_grid, res_far, 'b--', linewidth=2, label='Far (Coulomb) Residual')
    
    # 标记收敛界限的大致位置 (经验值 r_+ + 2*kappa)
    kappa = np.sqrt(1.0 - a**2)
    limit_approx = r_plus + 2.0 * kappa
    plt.axvline(limit_approx, color='g', linestyle=':', label='Approx Convergence Boundary')
    
    plt.title(f"Check: Teukolsky Eq. 113 Validity\n(a={a}, w={omega})")
    plt.xlabel('r/M')
    plt.ylabel('Relative Residual (|LHS| / |R|)')
    plt.ylim(1e-12, 1e0) # 关注 10^-5 以下的区域
    plt.legend()
    plt.grid(True, which='both', alpha=0.3)
    
    outfile = "eq113_residual_check.png"
    plt.savefig(outfile)
    print(f"Done. Plot saved to {outfile}")

if __name__ == "__main__":
    analyze_eq113()