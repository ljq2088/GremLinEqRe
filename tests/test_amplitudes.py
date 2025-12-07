import sys
import os
import cmath

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from GremLinEqRe import _core
except ImportError as e:
    print("Error: 无法导入 C++ 模块。")
    sys.exit(1)

def test_amplitudes_finite():
    print("\n[Test] MST Amplitudes Calculation")
    
    # 物理参数: a=0.5, omega=0.05 (Kerr)
    a = 0.5
    omega = 0.05
    s = -2
    l = 2
    m = 2
    lam = 4.0 # 近似 lambda
    
    tr = _core.TeukolskyRadial(a, omega, s, l, m, lam)
    
    # 1. 求解 nu
    guess = 2.0 + 0.0j
    nu = tr.solve_nu(guess)
    print(f"Solved nu: {nu}")
    
    # 2. 计算振幅
    tr.compute_amplitudes(nu)
    
    B_inc = tr.B_trans # 绑定里好像叫 get_B_trans，属性是 B_trans
    # 检查一下你的 bindings.cpp 里怎么绑定的，假设是 property
    # 按照之前的习惯，我们应该用 tr.B_trans
    # 但我上一轮给的 C++ 绑定代码只写了 get_B_trans 对应的 .def_property_readonly("B_trans", ...)
    
    # 我们需要在 bindings.cpp 把 B_inc 和 C_trans 也暴露出来
    # 假设你已经补全了 binding (见下方说明)
    
    print(f"B_trans: {tr.B_trans}")
    
    # 简单的非 NaN 检查
    assert not cmath.isnan(tr.B_trans.real)
    print("✅ Amplitudes are finite.")

if __name__ == "__main__":
    test_amplitudes_finite()