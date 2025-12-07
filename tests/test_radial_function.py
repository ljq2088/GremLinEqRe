import sys
import os
import cmath

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from GremLinEqRe import _core
except ImportError as e:
    print("Error: 无法导入 C++ 模块。")
    sys.exit(1)

def test_radial_equation_check():
    print("\n[Test] Checking if R_in satisfies Teukolsky Equation")
    
    # 参数
    a = 0.5
    omega = 0.05
    s = -2
    l = 2
    m = 2
    # 注意: 为了严谨，lambda 应该由 spheroidal harmonic 模块计算得出
    # 但对于微分方程的一致性检查，只要 solve_nu 和 evaluate_ddR 使用同一个 lambda 即可
    lam = 4.0 
    
    tr = _core.TeukolskyRadial(a, omega, s, l, m, lam)
    
    # 1. 求解系数
    nu = tr.solve_nu(2.0+0j)
    print(f"nu: {nu}")
    tr.compute_coefficients(nu)
    
    # 2. 在收敛域内求值
    # 关键修改: MST 的超几何级数表示法在 r=10 处数值不稳定。
    # 应在视界附近测试 (r_plus approx 1.866 for a=0.5)
    r_test = 2.0 
    
    R, dR = tr.evaluate_R_in(r_test)
    print(f"R({r_test})  = {R}")
    print(f"dR({r_test}) = {dR}")
    
    # 3. 使用 evaluate_ddR 计算“理论上”的二阶导
    ddR_theoretical = tr.evaluate_ddR(r_test, R, dR)
    
    # 4. 使用数值差分验证 evaluate_ddR 是否正确
    h = 1e-6 #稍微减小步长以适应 r=2 的尺度
    R_p, dR_p = tr.evaluate_R_in(r_test + h)
    R_m, dR_m = tr.evaluate_R_in(r_test - h)
    
    ddR_numerical = (dR_p - dR_m) / (2*h)
    
    print(f"ddR (Theory): {ddR_theoretical}")
    print(f"ddR (Numeric): {ddR_numerical}")
    
    # 比较
    # 关键修改: 使用相对误差，因为 R 的幅值可能不是 1
    abs_val = abs(ddR_theoretical)
    if abs_val < 1e-15:
        diff_rel = abs(ddR_theoretical - ddR_numerical)
    else:
        diff_rel = abs(ddR_theoretical - ddR_numerical) / abs_val
        
    print(f"Relative Diff: {diff_rel:.2e}")
    
    # 相对误差应小于 1e-5 (取决于步长 h 和双精度限制)
    assert diff_rel < 1e-5
    print("✅ R_in satisfies the radial differential structure.")

if __name__ == "__main__":
    test_radial_equation_check()