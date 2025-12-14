import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import cmath

# 确保能导入 C++ 模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from GremLinEqRe import _core
except ImportError:
    print("❌ Error: 无法导入 C++ 模块 GremLinEqRe._core")
    sys.exit(1)

def test_matching():
    print("=========================================================")
    print("Testing Radial Function Matching (MST Method)")
    print("=========================================================")

    # 1. 设置物理参数
    # 选择一个典型的 EMRI 参数，避免极端值
    M = 1.0
    a = 0.5         # 中等自旋
    omega = 0.2     # 低频 (M*omega = 0.2)
    s = -2          # 引力波自旋
    l = 2
    m = 2
    lambda_val = 4.0 # 近似值，实际应由 SWSH 求解，但对此测试不影响连续性验证

    print(f"Parameters: a={a}, omega={omega}, l={l}, m={m}")

    # 2. 初始化求解器
    tr = _core.TeukolskyRadial(M, a, omega, s, l, m, lambda_val)

    # 3. 求解重整化角动量 nu
    # 初始猜测 nu ~ l
    nu_guess = complex(float(l)+ 0.01, 0.01)
    nu = tr.solve_nu(nu_guess)
    print(f"Solved nu: {nu:.6f}")

    # 4. 计算级数系数 (a_n)
    # 计算正分支 (nu)
    n_max = 20
    a_coeffs_pos = tr.ComputeSeriesCoefficients(nu, n_max)
    
    # 计算负分支 (-nu-1)
    # 注意: MST 方法的远场解需要两个线性独立的库伦解组合
    nu_neg = -nu - 1.0
    a_coeffs_neg = tr.ComputeSeriesCoefficients(nu_neg, n_max)
    print(f"Computed series coefficients up to n={n_max}")

    # 5. 计算连接因子 K_nu
    # R_in = K_nu * R_C(nu) + K_-nu-1 * R_C(-nu-1)
    K_pos = tr.k_factor(nu)
    K_neg = tr.k_factor(nu_neg)
    print(f"K(nu)    : {K_pos:.4e}")
    print(f"K(-nu-1) : {K_neg:.4e}")

    # 6. 准备绘图网格
    # 视界半径 r_plus
    r_plus = 1.0 + np.sqrt(1.0 - a**2)
    print(f"Horizon r+: {r_plus:.4f}")

    # 我们在 r_plus + 0.1 到 20M 之间画图
    r_values = np.linspace(r_plus + 0.05, 20.0, 200)
    
    val_near_list = []
    val_far_list = []
    
    # 7. 逐点计算
    for r in r_values:
        # --- A. 近场解 (Hypergeometric) ---
        # 只在近处计算，远处会发散
        if r < 10.0: 
            res_near = tr.Evaluate_Hypergeometric(r, nu, a_coeffs_pos)
            # res_near 是 (Value, Deriv) 的 tuple
            val_near = res_near[0] 
        else:
            val_near = complex(np.nan, np.nan)
        
        # --- B. 远场解 (Coulomb 组合) ---
        # Eq. 166: R_in = K_nu * R_C^nu + K_-nu-1 * R_C^-nu-1
        
        # 分支 1: nu
        res_c_pos = tr.Evaluate_Coulomb(r, nu, a_coeffs_pos)
        val_c_pos = res_c_pos[0]
        
        # 分支 2: -nu-1
        res_c_neg = tr.Evaluate_Coulomb(r, nu_neg, a_coeffs_neg)
        val_c_neg = res_c_neg[0]
        
        # 线性组合
        val_far = K_pos * val_c_pos + K_neg * val_c_neg
        
        val_near_list.append(val_near)
        val_far_list.append(val_far)

    val_near_arr = np.array(val_near_list)
    val_far_arr = np.array(val_far_list)

    # 8. 绘图验证
    plt.figure(figsize=(12, 8))
    
    # 实部
    plt.subplot(2, 1, 1)
    plt.plot(r_values, val_near_arr.real, 'r-', lw=3, alpha=0.6, label='Near (Hypergeo)')
    plt.plot(r_values, val_far_arr.real, 'b--', lw=2, label='Far (Coulomb Combined)')
    plt.axvline(x=5.0, color='k', linestyle=':', label='Typical Matching r=5M')
    plt.ylabel('Real Part of $R^{in}(r)$')
    plt.title(f'Radial Function Matching Check (a={a}, $\omega$={omega})')
    plt.legend()
    plt.grid(True)
    
    # 虚部
    plt.subplot(2, 1, 2)
    plt.plot(r_values, val_near_arr.imag, 'r-', lw=3, alpha=0.6, label='Near (Hypergeo)')
    plt.plot(r_values, val_far_arr.imag, 'b--', lw=2, label='Far (Coulomb Combined)')
    plt.axvline(x=5.0, color='k', linestyle=':', label='Matching Radius')
    plt.ylabel('Imag Part of $R^{in}(r)$')
    plt.xlabel('Radius r/M')
    plt.legend()
    plt.grid(True)

    # 9. 计算重合区域误差 (例如在 r=4 到 r=6)
    idx_match = (r_values >= 4.0) & (r_values <= 6.0)
    if np.any(idx_match):
        diff = np.abs(val_near_arr[idx_match] - val_far_arr[idx_match])
        mean_val = np.abs(val_near_arr[idx_match])
        rel_err = np.mean(diff / mean_val)
        print(f"\nAverage Relative Error in Matching Region [4M, 6M]: {rel_err:.4e}")
        
        if rel_err < 1e-5:
            print("✅ Matching Successful! (High Precision)")
        elif rel_err < 1e-2:
            print("⚠️ Matching Acceptable but check precision (Medium Precision)")
        else:
            print("❌ Matching Failed! (Curves verify diverged)")

    output_file = "radial_matching_test.png"
    plt.savefig(output_file)
    print(f"\nPlot saved to {output_file}. Check this image manually!")
    # plt.show() # 如果在远程服务器，请注释掉此行

if __name__ == "__main__":
    test_matching()