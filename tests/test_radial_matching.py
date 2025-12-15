import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import cmath

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from GremLinEqRe import _core
except ImportError:
    print("❌ Error: 无法导入 C++ 模块 GremLinEqRe._core")
    sys.exit(1)

def test_matching():
    print("=========================================================")
    print("Testing Radial Function Matching (with SWSH & Correct Lambda)")
    print("=========================================================")

    # 1. 物理参数
    M = 1.0
    a = 0.5
    omega = 0.2     # 频率 M*omega
    s = -2
    l = 2
    m = 2

    # 2. [关键修正] 使用 SWSH 模块计算精确的 Lambda
    print(f"Calculating correct lambda for a*omega = {a*omega:.4f} ...")
    swsh = _core.SWSH(s, l, m, a * omega)
    lambda_val = swsh.E # 获取计算出的特征值
    print(f"Correct Lambda: {lambda_val:.6f}")

    # 3. 初始化径向求解器
    tr = _core.TeukolskyRadial(M, a, omega, s, l, m, lambda_val)

    # 4. 求解 nu
    nu_guess = complex(float(l), 0.0)
    nu = tr.solve_nu(nu_guess)
    print(f"Solved nu: {nu:.6f}")

    # 5. 计算系数与 K因子
    n_max = 30 # 增加阶数以提高精度
    a_coeffs_pos = tr.ComputeSeriesCoefficients(nu, n_max)
    
    nu_neg = -nu - 1.0
    a_coeffs_neg = tr.ComputeSeriesCoefficients(nu_neg, n_max)
    
    K_pos = tr.k_factor(nu)
    K_neg = tr.k_factor(nu_neg)
    print(f"K(nu)    : {K_pos:.4e}")
    print(f"K(nu_neg): {K_neg:.4e}")

    # 6. 绘图验证 (调整范围，关注视界附近)
    r_plus = 1.0 + np.sqrt(1.0 - a**2)
   
    kappa = np.sqrt(1.0 - a**2)
    # 推荐位置: r_match = r_+ + 1.2 * kappa (既不太靠近视界，又在收敛区内)
    r_match_optimal = r_plus + 1.2 * kappa
    print(f"Horizon r+: {r_plus:.4f}")
    print(f"Optimal Matching Radius (r_+ + 1.2*kappa): {r_match_optimal:.4f}")

    # 扫描区域：集中在匹配点附近
    r_values = np.linspace(r_plus + 0.05, 6.0, 400)
    
    val_near_real = []
    val_far_real = []
    diff_list = []
    
    for r in r_values:
        # A. 近场 (Hypergeometric): 只在收敛半径内计算
        # 如果超出 r_+ + 2*kappa (约3.6M)，数值误差会指数级放大
        if r < r_match_optimal + 1.0: 
            res_near = tr.Evaluate_Hypergeometric(r, nu, a_coeffs_pos)
            v_near = res_near[0]
        else:
            v_near = complex(np.nan, np.nan)
        
        # B. 远场 (Coulomb): 在匹配点之外计算
        if r > r_match_optimal - 0.5:
            res_c_pos = tr.Evaluate_Coulomb(r, nu, a_coeffs_pos)
            res_c_neg = tr.Evaluate_Coulomb(r, nu_neg, a_coeffs_neg)
            # 组合远场解
            v_far = K_pos * res_c_pos[0] + K_neg * res_c_neg[0]
        else:
            v_far = complex(np.nan, np.nan)
            
        val_near_real.append(v_near.real)
        val_far_real.append(v_far.real)
        
        # 计算误差
        if not np.isnan(v_near.real) and not np.isnan(v_far.real):
            abs_diff = abs(v_near - v_far)
            # 相对误差处理: 避免除以0
            denom = max(abs(v_near), 1e-10)
            diff_list.append(abs_diff / denom)
        else:
            diff_list.append(np.nan)

    # 7. 绘图
    plt.figure(figsize=(10, 10))
    
    # 子图1: 波函数实部
    plt.subplot(2, 1, 1)
    # 限制 Y 轴范围，避免因某一点发散导致全图不可读
    plt.ylim(-500, 500) # 根据 K_nu ~ 300，波幅可能在这个量级
    
    plt.plot(r_values, val_near_real, 'r-', lw=4, alpha=0.5, label='Near (Hypergeo)')
    plt.plot(r_values, val_far_real, 'b--', lw=1.5, label='Far (Coulomb)')
    plt.axvline(x=r_match_optimal, color='g', linestyle=':', label=f'Match r={r_match_optimal:.2f}')
    plt.title(f'Radial Match (r_match inside convergence radius)')
    plt.legend()
    plt.grid(True)
    
    # 子图2: 相对误差
    plt.subplot(2, 1, 2)
    plt.semilogy(r_values, diff_list, 'k-', lw=1)
    plt.axvline(x=r_match_optimal, color='g', linestyle=':')
    plt.ylabel('Relative Error')
    plt.xlabel('r/M')
    plt.grid(True, which="both")
    
    plt.savefig("radial_matching_optimized.png")

if __name__ == "__main__":
    test_matching()