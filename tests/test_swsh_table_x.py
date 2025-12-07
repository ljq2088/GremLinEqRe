import sys
import os
import math

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from GremLinEqRe import _core
except ImportError as e:
    print("Error: 无法导入 C++ 模块。")
    sys.exit(1)

def test_table_x_verification():
    """
    严谨求证：复现 Fujita & Tagoshi (2004) Table X 的数据
    条件: M*omega = 0.1, s = -2
    """
    print("\n[Test] Verification against FT2004 Table X (M*w = 0.1, s = -2)")
    print(f"{'q':<5} {'l':<3} {'m':<3} | {'Calculated E':<15} | {'Table X Value':<15} | {'Diff':<10}")
    print("-" * 65)

    M_omega = 0.1
    s = -2
    
    # 数据来源: FT2004 Table X (p.32)
    # 格式: (q, l, m, expected_E)
    # 抽取部分典型数据点进行验证
    data_points = [
        # q = 0.9
        (0.9, 2, 2,  5.7540002160),
        (0.9, 2, 1,  5.8752937655),
        (0.9, 2, 0,  5.9957562220),
        (0.9, 2, -1, 6.1153623140),
        (0.9, 2, -2, 6.2340859091),
        
        # q = -0.9 (注意: GremlinEq 代码逻辑中，逆行通常处理为负 m 或负 a)
        # 这里我们按物理参数输入 a = -0.9
        (-0.9, 2, 2, 6.2340859091), # Table X: l=2, m=2, q=-0.9 -> 6.234...
        (-0.9, 2, 1, 6.1153623140),
        (-0.9, 3, 3, 12.175986227), # Higher l
        
        # 验证 lambda 在 a=0 时的退化
        (0.0, 2, 2, 6.0) # E = l(l+1) = 6.0
    ]

    all_passed = True
    
    for q, l, m, expected in data_points:
        a_omega = q * M_omega # aw = (a/M) * (M*omega)
        
        solver = _core.SWSH(s, l, m, a_omega)
        calc_E = solver.E
        
        diff = abs(calc_E - expected)
        status = "✅" if diff < 1e-8 else "❌"
        
        print(f"{q:<5} {l:<3} {m:<3} | {calc_E:<15.8f} | {expected:<15.8f} | {diff:.2e} {status}")
        
        if diff > 1e-7:
            all_passed = False

    assert all_passed
    print("\n✅ Verification Successful: SWSH Eigenvalues match literature.")

if __name__ == "__main__":
    test_table_x_verification()