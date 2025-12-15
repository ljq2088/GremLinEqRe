import sys
import os
import math
import cmath
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from GremLinEqRe import _core

def test_source_terms():
    print("\n[Test] Source Terms Calculation (A coefficients)")
    
    # 1. 物理参数 Setup
    M = 1.0
    a = 0.9
    r_orbit = 5.0 # 圆轨道半径
    omega_phi = 1.0 / (r_orbit**1.5 + a) # 顺行圆轨道频率
    
    # 波参数
    l = 2
    m = 2
    omega = 2 * omega_phi # 主频
    s = -2
    
    print(f"Orbit: r={r_orbit}, a={a}, omega_orb={omega_phi:.4f}")
    print(f"Wave : l={l}, m={m}, omega={omega:.4f}")

    # 2. 初始化各个模块
    # (A) 几何
    geo = _core.KerrGeo.from_circular_equatorial(a, r_orbit, True)
    
    # (B) SWSH (为了计算 S 和 LdagS)
    print("Initializing SWSH...")
    swsh = _core.SWSH(s, l, m, a * omega)
    
    # (C) Source Calculator
    print("Initializing TeukolskySource...")
    ts = _core.TeukolskySource(a, omega,s,l, m)
    
    # 3. 构造粒子状态 (State)
    # 对于赤道面圆轨道:
    # t, r, theta, phi
    # u^t, u^r, u^theta, u^phi
    state = _core.KerrGeo.State()
    state.x = [0.0, r_orbit, math.pi/2.0, 0.0] # Theta = pi/2
    
    # 计算四速度 (u^t, u^phi)
    # u^t = E / (1 - 2M/r) ... (简化公式仅适用Schwarzschild，Kerr需完整公式)
    # 我们利用 KerrGeo 内部逻辑或直接根据公式算。
    # 简便起见，这里手动算一下 u^t 和 u^phi
    v = 1.0 / math.sqrt(r_orbit)
    # u^t for circular equatorial
    denom = 1.0 - 3.0*v**2 + 2.0*a*v**3
    ut = (1.0 + a*v**3) / (math.sqrt(denom) * math.sqrt(1.0 - 2.0*v**2 + a**2*v**4) ) # 近似或直接用 geo.energy 反推
    
    # 更严谨的做法：利用 E 和 Lz 反解 u^t, u^phi
    # Sigma * u^t = ...
    # 在赤道面 Sigma = r^2
    # Delta = r^2 - 2r + a^2
    Delta = r_orbit**2 - 2.0*r_orbit + a**2
    Sigma = r_orbit**2
    E = geo.energy
    Lz = geo.angular_momentum
    
    # u^t = 1/Sigma * [ (r^2+a^2)^2/Delta - a^2 ] * E + ... (复杂)
    # 让我们用简单物理关系: u^phi = Omega * u^t
    # u_mu u^mu = -1 => -1 = g_tt (u^t)^2 + 2 g_tphi u^t u^phi + g_phiphi (u^phi)^2
    # => -1 = (u^t)^2 [ g_tt + 2 Omega g_tphi + Omega^2 g_phiphi ]
    
    g_tt = -(1.0 - 2.0/r_orbit)
    g_tphi = -2.0*a/r_orbit
    g_phiphi = (r_orbit**2 + a**2 + 2.0*a**2/r_orbit) # sin=1
    
    Gamma_sq = - (g_tt + 2*omega_phi*g_tphi + omega_phi**2*g_phiphi)
    ut = 1.0 / math.sqrt(Gamma_sq)
    uphi = omega_phi * ut
    
    state.u = [ut, 0.0, 0.0, uphi] # u^r=0, u^theta=0
    
    print(f"4-Velocity: ut={ut:.4f}, uphi={uphi:.4f}")

    # 4. 计算投影
    print("Computing Projections...")
    proj = ts.ComputeProjections(state, geo, swsh)
    
    # 5. 验证结果
    print("\n[Results]")
    print(f"A_nn0       : {proj.A_nn0}")
    print(f"A_mbarn0    : {proj.A_mbarn0}")
    print(f"A_mbarmbar0 : {proj.A_mbarmbar0}")
    
    # 赤道面检查：
    # 对于 l=2, m=2, s=-2 的模式，S(theta) 在 pi/2 处是对称的（偶宇称）还是反对称？
    # l-max(|m|,|s|) = 2-2 = 0 (偶数)。 S 在 pi/2 处非零，dS/dtheta 为 0。
    # 预期 A_mbarn0 应该包含 dS 项？
    # 只要数值不是 NaN 且量级合理即可。
    
    assert not cmath.isnan(proj.A_nn0.real)
    assert abs(proj.A_nn0) > 1e-10 # 应该有值
    
    print("✅ Source terms computed successfully.")

if __name__ == "__main__":
    test_source_terms()