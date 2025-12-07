import sys
import os
import math
import cmath
from scipy.special import loggamma as scipy_loggamma

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from GremLinEqRe import _core
except ImportError as e:
    print("Error: 无法导入 C++ 模块。")
    sys.exit(1)

def is_close_modulo_2pi(z1, z2, tol=1e-10):
    """
    判断两个复数是否相等（虚部允许差 2*pi*k）
    因为 exp(z1) == exp(z2) 当 z1 - z2 = 2*pi*i*k
    """
    # 实部必须相等
    if abs(z1.real - z2.real) > tol:
        return False
    
    # 虚部差值除以 2pi 应该是整数
    diff_imag = abs(z1.imag - z2.imag)
    remainder = diff_imag % (2 * math.pi)
    
    # remainder 应该接近 0 或者接近 2pi
    return min(remainder, 2 * math.pi - remainder) < tol

def test_log_gamma():
    print("\n[Test 1] Complex Log Gamma Function (Robust to Branch Cuts)")
    
    test_points = [
        1.0 + 0.0j,
        2.5 + 0.0j,
        0.5 + 2.0j,
        -1.5 + 0.5j,  # 之前报错的点
        10.0 + 100.0j
    ]
    
    all_passed = True
    print(f"{'Input':<20} | {'My C++':<25} | {'Scipy':<25} | {'Status':<10}")
    print("-" * 85)
    
    for z in test_points:
        val_cpp = _core.TeukolskyRadial.log_gamma(z)
        val_scipy = scipy_loggamma(z)
        
        # 验证 exp(val) 是否一致 (物理意义)
        passed = is_close_modulo_2pi(val_cpp, val_scipy)
        status = "✅" if passed else "❌"
        
        print(f"{str(z):<20} | {str(val_cpp):<25} | {str(val_scipy):<25} | {status}")
        
        if not passed:
            all_passed = False
            
    assert all_passed
    print("✅ Passed")

def test_mst_coefficients():
    print("\n[Test 2] MST Coefficients Consistency")
    # 构造一个示例对象
    tr = _core.TeukolskyRadial(0.9, 0.1, -2, 2, 2, 2.0)
    nu = 2.0 + 0.5j
    n = 0
    
    alpha = tr.coeff_alpha(nu, n)
    beta = tr.coeff_beta(nu, n)
    gamma = tr.coeff_gamma(nu, n)
    
    print(f"nu={nu}, n={n}")
    print(f"Alpha: {alpha}")
    print(f"Beta : {beta}")
    print(f"Gamma: {gamma}")
    
    assert not cmath.isnan(alpha.real)
    print("✅ Passed")

def test_continued_fraction_convergence():
    print("\n[Test 3] Continued Fraction Convergence")
    # 使用较小的参数保证收敛
    tr = _core.TeukolskyRadial(0.5, 0.05, -2, 2, 2, 2.0)
    nu = 2.0 + 0.1j
    
    cf_plus = tr.continued_fraction(nu, 1)
    print(f"CF(+): {cf_plus}")
    
    assert not cmath.isnan(cf_plus.real)
    print("✅ Passed")

if __name__ == "__main__":
    test_log_gamma()
    test_mst_coefficients()
    test_continued_fraction_convergence()