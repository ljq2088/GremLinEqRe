import sys
import os
import numpy as np
import mpmath
import cmath

# 确保能导入 C++ 模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from GremLinEqRe import _core
except ImportError:
    print("❌ Error: 无法导入 C++ 模块 GremLinEqRe._core")
    print("请确保你已经编译了项目 (pip install . 或 cmake build)")
    sys.exit(1)

def compare_hyp2f1(a, b, c, z, label=""):
    """
    对比 C++ (Arb) 和 mpmath 的 2F1 计算结果
    """
    # 1. 实例化 TeukolskyRadial 对象 (仅用于访问 hyp2f1 包装器)
    # 参数随意，不影响静态计算
    tr = _core.TeukolskyRadial(1.0,0.9, 0.1, -2, 2, 2, 4.0)

    # 2. 设置 mpmath 精度 (作为基准，我们可以设得比 double 高，比如 30 位)
    mpmath.mp.dps = 30

    # 3. 调用 C++ (Arb)
    try:
        # C++ 绑定接受 Python complex 类型
        val_arb = tr.hyp2f1(complex(a), complex(b), complex(c), complex(z))
    except Exception as e:
        print(f"[{label}] ❌ C++ Call Failed: {e}")
        return

    # 4. 调用 mpmath (作为 Ground Truth)
    try:
        # mpmath.hyp2f1 接受 mpmath 类型的复数
        val_mp = mpmath.hyp2f1(a, b, c, z)
        # 将高精度结果转回标准 complex 以便比较
        val_ref = complex(val_mp)
    except Exception as e:
        print(f"[{label}] ⚠️ mpmath calculation failed: {e}")
        return

    # 5. 计算相对误差
    diff = abs(val_arb - val_ref)
    mean_val = (abs(val_arb) + abs(val_ref)) / 2.0
    
    # 避免除以零
    if mean_val < 1e-15:
        rel_diff = diff
    else:
        rel_diff = diff / mean_val

    # 6. 打印结果
    print(f"--- Case: {label} ---")
    print(f"Parameters: a={a}, b={b}, c={c}, z={z}")
    print(f"  C++ (Arb) : {val_arb:.16g}")
    print(f"  Py (mpmath): {val_ref:.16g}")
    print(f"  Rel Diff   : {rel_diff:.4e}")
    
    # 判定标准：C++ 返回的是 double，精度极限约 1e-15
    if rel_diff < 1e-13:
        print("✅ Match (Excellent)")
    elif rel_diff < 1e-10:
        print("✅ Match (Good)")
    else:
        print("❌ MISMATCH")
        print(f"   (mpmath raw: {val_mp})") # 打印 mpmath 原始高精度值以供调试
    print("")

def run_tests():
    print("=========================================================")
    print("Testing C++ Arb Wrapper vs Python mpmath (hyp2f1)")
    print("=========================================================\n")

    # Case 1: 实数参数，单位圆内 (基础检查)
    compare_hyp2f1(1.0, 2.0, 3.0, 0.5, "Real, |z| < 1")

    # Case 2: 复数参数，单位圆内
    compare_hyp2f1(1+1j, 2-0.5j, 3+2j, 0.5+0.5j, "Complex, |z| < 1")

    # Case 3: 单位圆外 (|z| > 1) - 测试 MST 关键的解析延拓
    # 这是 MST 方法中最常见的区域 (x 很大且为负)
    compare_hyp2f1(0.5+2j, -1.0-1j, 1.5+0j, -5.0+0.1j, "Complex, |z| > 1 (MST regime)")

    # Case 4: 接近分支点 z=1
    compare_hyp2f1(0.5, 0.5, 1.5, 0.99+0.01j, "Near branch point z=1")

    # Case 5: 接近奇点 (MST 的参数 c 有时会接近整数，测试稳定性)
    # Arb 基于球算术，通常能很好地处理这种情况
    compare_hyp2f1(10.0, -10.0, 0.0001+0.0j, 0.3, "Small c (Near singularity)")

    # Case 6: 高阶 MST 参数模拟 (模拟 n 很大时的情况)
    n = 20
    compare_hyp2f1(n + 2j, -n - 2j, 1.0 - 2j, -0.5, "Large parameters (High n)")
    
    # Case 7: 纯虚数大参数 (模拟高频 limit)
    compare_hyp2f1(100j, -100j, 1.0+5j, 0.5+0.5j, "Large Imaginary params")

if __name__ == "__main__":
    run_tests()