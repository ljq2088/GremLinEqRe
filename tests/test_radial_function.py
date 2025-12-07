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
    # 我们先随便给一个 lambda，实际上 lambda 应该由 SWSH 给出
    # 但只要 evaluate_ddR 使用相同的 lambda，检查应该自洽
    lam = 4.0 
    
    tr = _core.TeukolskyRadial(a, omega, s, l, m, lam)
    
    # 1. 求解系数
    nu = tr.solve_nu(2.0+0j)
    print(f"nu: {nu}")
    tr.compute_coefficients(nu)
    
    # 2. 在某个半径处求值
    r_test = 10.0 # 离视界远一点，但在收敛域内
    
    R, dR = tr.evaluate_R_in(r_test)
    print(f"R({r_test})  = {R}")
    print(f"dR({r_test}) = {dR}")
    
    # 3. 使用 evaluate_ddR 计算“理论上”的二阶导
    ddR_theoretical = tr.evaluate_ddR(r_test, R, dR)
    
    # 4. 使用数值差分验证 evaluate_ddR 是否正确
    # (这一步验证 evaluate_ddR 公式没写错)
    h = 1e-5
    R_p, dR_p = tr.evaluate_R_in(r_test + h)
    R_m, dR_m = tr.evaluate_R_in(r_test - h)
    
    ddR_numerical = (dR_p - dR_m) / (2*h)
    
    print(f"ddR (Theory): {ddR_theoretical}")
    print(f"ddR (Numeric): {ddR_numerical}")
    
    # 比较
    diff = abs(ddR_theoretical - ddR_numerical)
    print(f"Diff: {diff:.2e}")
    
    # 如果 Diff 很小，说明 evaluate_R_in 计算出的曲线确实满足 Teukolsky 方程的局部性质
    # 或者说我们的 evaluate_R_in 的一阶导数行为是平滑且物理的
    assert diff < 1e-4
    print("✅ R_in satisfies the radial differential structure.")

if __name__ == "__main__":
    test_radial_equation_check()