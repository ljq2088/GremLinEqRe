import numpy as np
import sys
import os
import math
import cmath
try:
    from GremLinEqRe import _core
except ImportError:
    print("❌ Error: 无法导入 C++ 模块 GremLinEqRe")
    sys.exit(1)

import matplotlib.pyplot as plt

def test_spectral_accuracy():
    print("=== Testing SWSH Spectral Implementation ===")
    
    # 参数设置
    s = -2
    l = 2
    m = 2
    a_spin = 0.9
    omega = 0.5
    a_omega = a_spin * omega
    
    print(f"Parameters: s={s}, l={l}, m={m}, aw={a_omega}")
    
    # 初始化 SWSH (新版会自动求解特征值)
    swsh = _core.SWSH(s, l, m, a_omega)
    
    E = swsh.get_E()
    Lambda = swsh.get_lambda()
    print(f"Eigenvalue E = {E:.12f}")
    print(f"Lambda     = {Lambda:.12f}")
    
    # 测试点 (避开极点)
    thetas = np.linspace(0.1, np.pi-0.1, 100)
    xs = np.cos(thetas)
    
    # --- 验证 1: L_{-2}^dag S ---
    # 理论算符: L^dag = d/dtheta - m/sin - s*cot + aw*sin
    # 我们用数值差分 d/dtheta 来检查谱方法结果
    
    errs_L2 = []
    
    h = 1e-5
    for x, th in zip(xs, thetas):
        sin_th = np.sin(th)
        cos_th = np.cos(th)
        cot_th = cos_th/sin_th
        
        # 1. 谱方法直接计算结果 (包含 aw*sin)
        val_spectral = swsh.evaluate_L2dag_S(x)
        
        # 2. 数值差分验证
        # 计算 S(x)
        S_plus  = swsh.evaluate_S(np.cos(th + h))
        S_minus = swsh.evaluate_S(np.cos(th - h))
        dS_dth_num = (S_plus - S_minus) / (2*h)
        
        # S(x) 当前值
        S_val = swsh.evaluate_S(x)
        
        # 手动构建算符: dS/dth + (-m/sin - s*cot + aw*sin) S
        potential = -m/sin_th - s*cot_th + a_omega * sin_th
        val_numerical = dS_dth_num + potential * S_val
        
        # 误差
        err = np.abs(val_spectral - val_numerical)
        errs_L2.append(err)

    print(f"L2dag Max Error (vs finite diff): {np.max(errs_L2):.2e}")
    
    # --- 验证 2: 归一化 ---
    # int |S|^2 sin theta dtheta = int |S|^2 dx = 1
    # 使用高斯积分或简单的梯形法则
    S_vals = [swsh.evaluate_S(x) for x in xs]
    integrand = np.abs(np.array(S_vals))**2
    norm = np.trapz(integrand, x=-xs) # 注意 x 是递减的，或者用 abs
    print(f"Normalization Check (Trapz): {norm:.5f} (Expect ~1.0)")
    
    if np.max(errs_L2) < 1e-4: # 差分本身有截断误差，1e-5左右正常
        print(">>> Test PASSED: Spectral operator matches numerical limit.")
    else:
        print(">>> Test FAILED: Mismatch too large.")

if __name__ == "__main__":
    test_spectral_accuracy()