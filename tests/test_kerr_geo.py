import sys
import os
import math

# 路径设置
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from GremLinEqRe import _core
except ImportError as e:
    print("Error: 无法导入 C++ 模块。请先编译项目。")
    sys.exit(1)

def numerical_derivative(func, x, h=1e-5):
    """ 计算五点中心差分，精度 O(h^4) """
    return (-func(x + 2*h) + 8*func(x + h) - 8*func(x - h) + func(x - 2*h)) / (12 * h)

def numerical_derivative_2(func, x, h=1e-4):
    """ 计算二阶导数的中心差分 """
    return (func(x + h) - 2*func(x) + func(x - h)) / (h * h)

def test_circular_orbit_params():
    """回归测试：验证圆形轨道参数 (Schwarzschild ISCO)"""
    print("\n[Test 1] Circular Orbit Parameters (Schwarzschild ISCO)")
    geo = _core.KerrGeo.from_circular_equatorial(0.0, 6.0, True)
    
    expected_E = 2.0 * math.sqrt(2.0) / 3.0
    expected_Lz = 2.0 * math.sqrt(3.0)
    
    assert math.isclose(geo.energy, expected_E, rel_tol=1e-9)
    assert math.isclose(geo.angular_momentum, expected_Lz, rel_tol=1e-9)
    print("✅ Passed")

def test_analytical_derivatives():
    """
    核心测试：验证解析导数 vs 数值差分
    这能确保 KerrGeo.cpp 中的公式推导没有手误
    """
    print("\n[Test 2] Analytical vs Numerical Derivatives")
    
    # 设定一个一般的轨道参数 (非圆，非赤道，自旋非零)
    # 以确保所有项都参与计算
    a = 0.9
    E = 0.95
    Lz = 3.0
    Q = 2.0  # Carter constant != 0
    
    geo = _core.KerrGeo(a, E, Lz, Q)
    
    # 在某个一般半径处测试 (远离视界和奇点)
    r_test = 10.0
    
    # 1. 验证一阶导数 R'(r)
    val_dr_analytical = geo.diff_potential_r(r_test)
    val_dr_numerical = numerical_derivative(geo.potential_r, r_test, h=1e-4)
    
    diff_1 = abs(val_dr_analytical - val_dr_numerical)
    print(f"R'(r)  | Analytical: {val_dr_analytical:.8f} | Numerical: {val_dr_numerical:.8f} | Diff: {diff_1:.2e}")
    
    # 允许一定的数值误差 (通常在 1e-8 到 1e-10 之间)
    assert math.isclose(val_dr_analytical, val_dr_numerical, rel_tol=1e-6)

    # 2. 验证二阶导数 R''(r)
    val_ddr_analytical = geo.diff2_potential_r(r_test)
    val_ddr_numerical = numerical_derivative_2(geo.potential_r, r_test, h=1e-4)
    
    diff_2 = abs(val_ddr_analytical - val_ddr_numerical)
    print(f"R''(r) | Analytical: {val_ddr_analytical:.8f} | Numerical: {val_ddr_numerical:.8f} | Diff: {diff_2:.2e}")
    
    assert math.isclose(val_ddr_analytical, val_ddr_numerical, rel_tol=1e-5)
    
    print("✅ Passed")

def test_potential_theta():
    """验证 Theta 势函数的基本性质"""
    print("\n[Test 3] Theta Potential Consistency")
    a = 0.5
    E = 0.9
    Lz = 2.0
    Q = 5.0
    geo = _core.KerrGeo(a, E, Lz, Q)
    
    # 验证极轴处 (theta=0, z=1) 是否被截断为 0 (防止除零或负值)
    # 根据代码逻辑，z->1 时 Lz^2/(1-z^2) 会发散，但物理上束缚轨道无法到达这里
    # 除非 Lz=0。我们的代码里加了保护。
    val_pole = geo.potential_theta(1.0)
    print(f"Theta(z=1): {val_pole}")
    assert val_pole == 0.0
    
    print("✅ Passed")

if __name__ == "__main__":
    test_circular_orbit_params()
    test_analytical_derivatives()
    test_potential_theta()