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

def test_swsh_evaluation():
    print("\n[Test] SWSH Function Evaluation")
    
    # 参数: Schwarzschild case
    s = -2
    l = 2
    m = 2
    a_omega = 0.0
    
    swsh = _core.SWSH(s, l, m, a_omega)
    
    # 赤道面 x=0 (theta=pi/2)
    # 对于 s=-2, l=2, m=2 的球谐函数 Y_{2,2}
    # Y_{2,2} ~ sin^2(theta) * e^{2iphi}
    # 在 theta=pi/2, sin=1. 模应该是常数。
    
    val_eq = swsh.evaluate_S(0.0)
    print(f"S(x=0) [a*w=0]: {val_eq}")
    
    # 算符测试
    val_op = swsh.evaluate_L2dag_S(0.0)
    print(f"L2dag S(x=0): {val_op}")
    
    # 简单的非零检查
    assert abs(val_eq) > 1e-5
    print("✅ SWSH Evaluation basic sanity check passed.")

if __name__ == "__main__":
    test_swsh_evaluation()