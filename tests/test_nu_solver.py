import sys
import os
import math
import cmath

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from GremLinEqRe import _core
except ImportError as e:
    print("Error: 无法导入 C++ 模块。")
    sys.exit(1)

def test_schwarzschild_limit():
    """
    测试史瓦西极限 (a=0)
    物理预期: nu 应该非常接近整数 l (在低频下)
    """
    print("\n[Test 1] Schwarzschild Limit (a=0)")
    
    a = 0.0
    omega = 0.01 # 低频
    s = -2
    l = 2
    m = 2
    # Schwarzschild 极限下的 lambda (自旋加权球谐函数特征值)
    # lambda = (l-s)(l+s+1) = (2 - (-2)) * (2 + (-2) + 1) = 4 * 1 = 4
    # Wait, FT2004 Appendix A says lambda = l(l+1) - s(s+1)
    # lambda = 2*3 - (-2)*(-1) = 6 - 2 = 4. Correct.
    lam = 4.0 
    
    tr = _core.TeukolskyRadial(a, omega, s, l, m, lam)
    
    # 初始猜测 nu = l = 2
    guess = 2.0 + 0.0j
    
    print(f"Initial guess: {guess}")
    nu_sol = tr.solve_nu(guess)
    print(f"Solved nu    : {nu_sol}")
    
    residual = tr.calc_g(nu_sol)
    print(f"Residual |g(nu)|: {abs(residual):.2e}")
    
    # 验证残差是否足够小
    assert abs(residual) < 1e-10
    # 验证解是否接近 l (对于非常小的 omega)
    assert abs(nu_sol.real - l) < 0.1
    
    print("✅ Passed")

def test_kerr_convergence():
    """
    测试克尔情况 (a=0.9) 的收敛性
    """
    print("\n[Test 2] Kerr Convergence (a=0.9)")
    
    a = 0.9
    omega = 0.1
    s = -2
    l = 2
    m = 2
    # 这里 lambda 我们暂时用史瓦西的值近似，或者随便给一个合理值
    # 真实的 lambda 需要 SWSH 求解器，但这不影响 nu 求解器的数学收敛性
    lam = 4.0 
    
    tr = _core.TeukolskyRadial(a, omega, s, l, m, lam)
    
    guess = 2.0 + 0.1j # 给一点虚部作为扰动
    
    nu_sol = tr.solve_nu(guess)
    print(f"Solved nu    : {nu_sol}")
    
    residual = tr.calc_g(nu_sol)
    print(f"Residual |g(nu)|: {abs(residual):.2e}")
    
    assert abs(residual) < 1e-10
    print("✅ Passed")

if __name__ == "__main__":
    test_schwarzschild_limit()
    test_kerr_convergence()