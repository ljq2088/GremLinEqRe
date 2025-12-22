# import sys
# import os
# import numpy as np
# import matplotlib.pyplot as plt
# import cmath

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# try:
#     from GremLinEqRe import _core
# except ImportError:
#     print("❌ Error: 无法导入 C++ 模块 GremLinEqRe._core")
#     sys.exit(1)

# def test_matching():
#     print("=========================================================")
#     print("Testing Radial Function Matching (with SWSH & Correct Lambda)")
#     print("=========================================================")

#     # 1. 物理参数
#     M = 1.0
#     a = 0.5
#     omega = 0.1    # 频率 M*omega
#     s = -2
#     l = 2
#     m = 2

#     # 2. [关键修正] 使用 SWSH 模块计算精确的 Lambda
#     print(f"Calculating correct lambda for a*omega = {a*omega:.4f} ...")
#     swsh = _core.SWSH(s, l, m, a * omega)
#     lambda_val = swsh.m_lambda # 获取计算出的特征值
#     print(f"Correct Lambda: {lambda_val:.6f}")

#     # 3. 初始化径向求解器
#     tr = _core.TeukolskyRadial(M, a, omega, s, l, m, lambda_val)

#     # 4. 求解 nu
#     nu_guess = complex(float(l), 0.0)
#     nu = tr.solve_nu(nu_guess)
#     print(f"Solved nu: {nu:.6f}")

#     # 5. 计算系数与 K因子
#     n_max = 100 # 增加阶数以提高精度
#     a_coeffs_pos = tr.ComputeSeriesCoefficients(nu, n_max)
    
#     nu_neg = -nu - 1.0
#     a_coeffs_neg = tr.ComputeSeriesCoefficients(nu_neg, n_max)
    
#     K_pos = tr.k_factor(nu)
#     K_neg = tr.k_factor(nu_neg)
#     print(f"K(nu)    : {K_pos:.4e}")
#     print(f"K(nu_neg): {K_neg:.4e}")

#     # 6. 绘图验证 (调整范围，关注视界附近)
#     r_plus = 1.0 + np.sqrt(1.0 - a**2)
   
#     kappa = np.sqrt(1.0 - a**2)
#     # 推荐位置: r_match = r_+ + 1.5 * kappa (既不太靠近视界，又在收敛区内)
#     r_match_optimal = r_plus + 1.2 * kappa
#     print(f"Horizon r+: {r_plus:.4f}")
#     print(f"Optimal Matching Radius (r_+ + 1.5*kappa): {r_match_optimal:.4f}")

#     # 扫描区域：集中在匹配点附近
#     r_values = np.linspace(r_plus + 0.05, 6.0, 800)
    
#     val_near_real = []
#     val_far_real = []
#     diff_list = []
#     val_R_in=[]
#     for r in r_values:
#         # A. 近场 (Hypergeometric): 只在收敛半径内计算
#         # 如果超出 r_+ + 2*kappa (约3.6M)，数值误差会指数级放大
#         if r < r_match_optimal : 
#             res_near = tr.Evaluate_Hypergeometric(r, nu, a_coeffs_pos)
#             v_near = res_near[0]
#         else:
#             v_near = complex(np.nan, np.nan)
        
#         # B. 远场 (Coulomb): 在匹配点之外计算
#         if r > r_match_optimal - kappa:
#             res_c_pos = tr.Evaluate_Coulomb(r, nu, a_coeffs_pos)
#             res_c_neg = tr.Evaluate_Coulomb(r, nu_neg, a_coeffs_neg)
#             # 组合远场解
#             v_far = K_pos * res_c_pos[0] + K_neg * res_c_neg[0]
#         else:
#             v_far = complex(np.nan, np.nan)
        
#         val_near_real.append(v_near.real)
    
#         val_far_real.append(v_far.real)
#         if np.isnan(v_near.real)& (r<r_match_optimal-1.0):
#             print(f"r={r}: Near is NaN")
#             print(f"Near: {v_near}")

#         if np.isnan(v_far.real)& (r>r_match_optimal):
#             print(f"r={r}: Far is NaN")
#             print(f"Far: {v_far}")
#             print("K_pos:", K_pos)
#             print("K_neg:", K_neg)
#             print("a_coeffs_pos:", a_coeffs_pos)
#             print("a_coeffs_neg:", a_coeffs_neg)
#             print("res_c_pos:", res_c_pos)
#             print("res_c_neg:", res_c_neg)
#         # # C.Evaluate_R_in
        
#         # v_R_in,_ = tr.Evaluate_R_in(r, nu,K_pos,K_neg, a_coeffs_pos, a_coeffs_neg, r_match_optimal)
#         # if np.isnan(v_R_in.real):
#         #     v_R_in = complex(np.nan, np.nan)
#         #     print(f"R_in: {v_R_in}")
#         # val_R_in.append(v_R_in.real)
#         # 计算误差
#         if not np.isnan(v_near.real) and not np.isnan(v_far.real):
#             abs_diff = abs(v_near - v_far)
#             # 相对误差处理: 避免除以0
#             denom = max(abs(v_near), 1e-10)
#             diff_list.append(abs_diff / denom)
#         else:
#             diff_list.append(np.nan)
        
#     print(diff_list)
#     # 7. 绘图
#     plt.figure(figsize=(10, 10))
    
#     # 子图1: 波函数实部
#     plt.subplot(2, 1, 1)
#     # 限制 Y 轴范围，避免因某一点发散导致全图不可读
#     plt.ylim(-500, 500) # 根据 K_nu ~ 300，波幅可能在这个量级
    
#     plt.plot(r_values, val_near_real, 'r-', lw=4, alpha=0.5, label='Near (Hypergeo)')
#     plt.plot(r_values, val_far_real, 'b--', lw=1.5, label='Far (Coulomb)')
#     # plt.plot(r_values, val_R_in, 'g-', lw=1, label='R_in')
#     plt.axvline(x=r_match_optimal, color='g', linestyle=':', label=f'Match r={r_match_optimal:.2f}')
#     plt.title(f'Radial Match (r_match inside convergence radius)')
#     plt.legend()
#     plt.grid(True)
    
#     # 子图2: 相对误差
#     plt.subplot(2, 1, 2)
#     plt.semilogy(r_values, diff_list, 'k-', lw=1)
#     plt.axvline(x=r_match_optimal, color='g', linestyle=':')
#     plt.ylabel('Relative Error')
#     plt.xlabel('r/M')
#     plt.grid(True, which="both")
    
#     plt.savefig("radial_matching_optimized.png")

# if __name__ == "__main__":
#     test_matching()
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import cmath

# 确保能导入模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from GremLinEqRe import _core
except ImportError:
    print("❌ Error: 无法导入 C++ 模块")
    sys.exit(1)

def test_full_matching():
    print("=========================================================")
    print("Diagnostic Radial Matching (Amplitude & Phase)")
    print("=========================================================")

    M, a, omega = 1.0, 0.5, 0.1  # 此时 omega 较小，远场收敛可能较慢
    s, l, m = -2, 2, 2

    # 1. 准备 Lambda 和 Solver
    swsh = _core.SWSH(s, l, m, a * omega)
    lambda_val = swsh.m_lambda
    print(f"Lambda: {lambda_val:.6f}")

    tr = _core.TeukolskyRadial(M, a, omega, s, l, m, lambda_val)
    
    # 2. 求解 nu 和 K
    nu = tr.solve_nu(complex(float(l), 0.0))
    n_max =50 # 增加级数项数
    
    print("Computing coefficients...")
    a_coeffs_pos = tr.ComputeSeriesCoefficients(nu, n_max)
    nu_neg = -nu - 1.0
    a_coeffs_neg = tr.ComputeSeriesCoefficients(nu_neg, n_max)
    # print('a_coeffs_pos:', a_coeffs_pos)
    # print('a_coeffs_neg:', a_coeffs_neg)
    K_pos = tr.k_factor(nu)
    K_neg = tr.k_factor(nu_neg)
    
    print(f"K(nu)    : {K_pos:.4e}")
    
    # 3. 设定扫描范围
    r_plus = 1.0 + np.sqrt(1.0 - a**2)
    kappa = np.sqrt(1.0 - a**2)
    
    # 扫描从 r_plus 附近直到远场
    r_values = np.linspace(r_plus + 0.01, 8.0, 500)
    
    near_abs, near_phase = [], []
    far_abs, far_phase = [], []
    
    print("Scanning r values...")
    
    for r in r_values:
        # --- Near Solution ---
        # 即使超出收敛半径也强行算一下，看看什么时候爆炸
        try:
            res_near = tr.Evaluate_Hypergeometric(r, nu, a_coeffs_pos)
            val_n = res_near[0]
        except:
            val_n = complex(np.nan, np.nan)
            
        # --- Far Solution ---
        try:
            res_c_pos = tr.Evaluate_Coulomb(r, nu, a_coeffs_pos)
            res_c_neg = tr.Evaluate_Coulomb(r, nu_neg, a_coeffs_neg)
            val_f = K_pos * res_c_pos[0] + K_neg * res_c_neg[0]
        except:
            val_f = complex(np.nan, np.nan)

        near_abs.append(abs(val_n))
        near_phase.append(np.angle(val_n))
        
        far_abs.append(abs(val_f))
        far_phase.append(np.angle(val_f))
    print(near_abs,far_abs)
    # 4. 绘图诊断
    fig, axes = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
    
    # 子图1: 幅度 (Log Scale)
    ax = axes[0]
    ax.semilogy(r_values, near_abs, 'r-', lw=2, label='Near (Hypergeo) Abs')
    ax.semilogy(r_values, far_abs, 'b--', lw=2, label='Far (Coulomb) Abs')
    # 标出理论匹配区域
    r_match_ideal = r_plus + 1.0 * kappa
    ax.axvline(r_match_ideal, color='g', ls=':', label='Ideal Match')
    ax.set_title("Amplitude |R| (Log Scale)")
    ax.legend()
    ax.grid(True)

    # 子图2: 相位 (Phase)
    ax = axes[1]
    #为了避免相位卷绕(wrapping)导致看来乱七八糟，使用 unwrap
    ax.plot(r_values, np.unwrap(near_phase), 'r-', lw=2, label='Near Phase')
    ax.plot(r_values, np.unwrap(far_phase), 'b--', lw=2, label='Far Phase')
    ax.axvline(r_match_ideal, color='g', ls=':')
    ax.set_title("Phase (Unwrapped)")
    ax.legend()
    ax.grid(True)

    # 子图3: 相对误差 (基于幅度)
    ax = axes[2]
    # 只计算非 NaN 且幅度不太小的区域
    diffs = []
    for i in range(len(r_values)):
        v_n = near_abs[i]
        v_f = far_abs[i]
        if np.isnan(v_n) or np.isnan(v_f) or v_n < 1e-20:
            diffs.append(np.nan)
        else:
            diffs.append(abs(v_n - v_f) / v_n)
            
    ax.semilogy(r_values, diffs, 'k-', label='Relative Amplitude Error')
    ax.axvline(r_match_ideal, color='g', ls=':')
    ax.set_ylabel("Error")
    ax.set_xlabel("r/M")
    ax.set_ylim(1e-4, 10)
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig("radial_diagnostic.png")
    print("Done. Check radial_diagnostic.png")

if __name__ == "__main__":
    test_full_matching()