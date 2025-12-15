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
    print(f"Horizon r+: {r_plus:.4f}")

    # 重点扫描 [r_plus, 10M] 区域
    r_values = np.linspace(r_plus + 0.01, 10.0, 300)
    
    val_near_real = []
    val_far_real = []
    diff_list = []
    
    for r in r_values:
        # A. 近场 (Hypergeometric) - 只要 r 不是太大
        if r < 6.0: 
            res_near = tr.Evaluate_Hypergeometric(r, nu, a_coeffs_pos)
            v_near = res_near[0]
        else:
            v_near = complex(np.nan, np.nan)
        
        # B. 远场 (Coulomb) - 只要 r 不是太小
        if r > 2.2:
            res_c_pos = tr.Evaluate_Coulomb(r, nu, a_coeffs_pos)
            res_c_neg = tr.Evaluate_Coulomb(r, nu_neg, a_coeffs_neg)
            v_far = K_pos * res_c_pos[0] + K_neg * res_c_neg[0]
        else:
            v_far = complex(np.nan, np.nan)
            
        val_near_real.append(v_near.real)
        val_far_real.append(v_far.real)
        
        # 记录差异 (仅当两者都有效时)
        if not np.isnan(v_near.real) and not np.isnan(v_far.real):
            diff_list.append(abs(v_near - v_far))
        else:
            diff_list.append(np.nan)

    # 7. 绘图
    plt.figure(figsize=(10, 10))
    
    # 子图 1: 波函数实部
    plt.subplot(2, 1, 1)
    plt.plot(r_values, val_near_real, 'r-', lw=4, alpha=0.5, label='Near (Hypergeo)')
    plt.plot(r_values, val_far_real, 'b--', lw=1.5, label='Far (Coulomb Combined)')
    plt.ylim(-0.5, 5.0) # 限制纵坐标，防止发散点破坏视图
    plt.ylabel('Real[R(r)]')
    plt.title(f'Matching Check (Correct $\lambda$={lambda_val:.3f})')
    plt.legend()
    plt.grid(True)
    
    # 子图 2: 误差 (对数坐标)
    plt.subplot(2, 1, 2)
    plt.semilogy(r_values, diff_list, 'k-', lw=1)
    plt.axvline(x=3.0, color='g', linestyle='--', label='Possible Match r=3M')
    plt.ylabel('Absolute Difference')
    plt.xlabel('Radius r/M')
    plt.grid(True, which="both")
    plt.legend()

    plt.savefig("radial_matching_fixed.png")
    print("\nPlot saved. Look for the region where Error is minimal (~1e-8 or less).")

if __name__ == "__main__":
    test_matching()