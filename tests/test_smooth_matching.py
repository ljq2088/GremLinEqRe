import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import cmath

# 路径设置
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from GremLinEqRe import _core
except ImportError:
    print("❌ Error: 无法导入 C++ 模块")
    sys.exit(1)

def test_smooth_connection():
    print("=========================================================")
    print("Testing Evaluate_R_in Smooth Connection & Reset Logic")
    print("=========================================================")

    M, a, omega = 1.0, 0.5, 0.1
    
    # 模拟多模态循环 (这里演示两个模式，虽然只画第一个)
    modes = [(2, 2), (2, 1)] 

    for l, m in modes:
        print(f"\nProcessing Mode (l={l}, m={m})...")
        s = -2
        
        # 1. 物理参数准备
        swsh = _core.SWSH(s, l, m, a * omega)
        lambda_val = swsh.m_lambda
        
        tr = _core.TeukolskyRadial(M, a, omega, s, l, m, lambda_val)
        
        # --- 关键：每次处理新模式前，重置校准状态 ---
        # 如果是新建的 tr 对象，其实不需要重置（构造函数默认重置）。
        # 但如果是复用对象，或者参数中途改变，必须调用。
        tr.ResetCalibration() 
        
        # 2. 求解 nu 和系数
        nu = tr.solve_nu(complex(float(l), 0.0))
        n_max = 50 
        
        a_coeffs_pos = tr.ComputeSeriesCoefficients(nu, n_max)
        nu_neg = -nu - 1.0
        a_coeffs_neg = tr.ComputeSeriesCoefficients(nu_neg, n_max)
        
        K_pos = tr.k_factor(nu)
        K_neg = tr.k_factor(nu_neg)
        
        # 3. 设定匹配点
        kappa = np.sqrt(1.0 - a**2)
        r_plus = 1.0 + kappa
        r_match = r_plus + 0.8 * kappa # 比如 r ~ 2.9
        print(f"r_plus: {r_plus:.4f}, r_match used: {r_match:.4f}")

        # 4. 扫描 r_match 附近的点，检查连接是否平滑
        # 重点检查 r_match - epsilon 到 r_match + epsilon
        r_center = r_match
        span = 0.5
        r_values = np.linspace(r_center - span, r_center + span, 400)
        
        R_vals = []
        phases = []
        
        for r in r_values:
            # 直接调用智能的 Evaluate_R_in
            # 此时不需要我们在 Python 里写 if r < r_match else ...
            res = tr.Evaluate_R_in(r, nu, K_pos, K_neg, a_coeffs_pos, a_coeffs_neg, r_match)
            val = res[0]
            
            R_vals.append(abs(val))
            phases.append(np.angle(val))
            
        R_vals = np.array(R_vals)
        phases = np.unwrap(phases) # 解卷绕，看相位是否连续

        # 5. 数值检验：检查 r_match 前后两个点的差值
        idx_match = np.abs(r_values - r_match).argmin()
        val_pre = R_vals[idx_match-1]
        val_post = R_vals[idx_match+1]
        jump_rel = abs(val_pre - val_post) / val_pre
        
        print(f"Continuity Check at r_match:")
        print(f"  R({r_values[idx_match-1]:.4f}) = {val_pre:.6e}")
        print(f"  R({r_values[idx_match+1]:.4f}) = {val_post:.6e}")
        print(f"  Jump (Relative): {jump_rel:.4e}")
        
        if jump_rel < 1e-2:
            print("  ✅ Connection is Smooth!")
        else:
            print("  ❌ Connection has a jump!")

        # 6. 绘图 (只画第一个模式 l=2, m=2 的详情)
        if l == 2 and m == 2:
            plt.figure(figsize=(10, 8))
            
            plt.subplot(2, 1, 1)
            plt.plot(r_values, R_vals, 'k-', lw=2, label='Evaluate_R_in Output')
            plt.axvline(r_match, color='r', linestyle='--', label='r_match (Switch Point)')
            plt.title(f'Amplitude Continuity Check (l={l}, m={m})')
            plt.ylabel('|R|')
            plt.legend()
            plt.grid(True)
            
            plt.subplot(2, 1, 2)
            plt.plot(r_values, phases, 'b-', lw=2, label='Phase')
            plt.axvline(r_match, color='r', linestyle='--')
            plt.title('Phase Continuity Check')
            plt.ylabel('Phase (rad)')
            plt.xlabel('r/M')
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(f"smooth_connect_l{l}m{m}.png")
            print(f"Plot saved to smooth_connect_l{l}m{m}.png")

if __name__ == "__main__":
    test_smooth_connection()