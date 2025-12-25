import numpy as np
import sys
import os

# 路径设置
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '..'))
if root_dir not in sys.path:
    sys.path.append(root_dir)

try:
    from GremLinEqRe import _core
except ImportError as e:
    print(f"Error: {e}")
    sys.exit(1)

def check_lrr_angular_eq15_residual(swsh, n_theta=4001):
    """
    验证 LRR Eq(15) 残差。
    【关键】：由于 C++ 内部计算的是 -m 模式 (e^{-im\phi})，
    这里的势能公式必须使用 m_eff = -m 才能匹配函数的形状。
    """
    aomega = swsh.aw
    m_phys = swsh.m
    m_eff = m_phys  
    
    s = swsh.s
    lam = swsh.m_lambda
    
    # 构造 LRR 分离常数 A_Teukolsky (Eq 15 constant term)
    # lambda = E - 2*m_phys*aw + aw^2 - s(s+1)
    # A_Teuk = E - s(s+1) = lambda + 2*m_phys*aw - aw^2
    A_teuk = lam + 2*m_phys*aomega - aomega**2

    thetas = np.linspace(1e-4, np.pi - 1e-4, n_theta)
    h = thetas[1] - thetas[0]

    S = np.array([swsh.evaluate_S(np.cos(th)) for th in thetas], dtype=np.complex128)
    
    # 差分导数
    S_th = (S[2:] - S[:-2]) / (2*h)
    S_th2 = (S[2:] - 2*S[1:-1] + S[:-2]) / (h*h)

    th = thetas[1:-1]
    st = np.sin(th)
    ct = np.cos(th)
    cot = ct / st

    # Teukolsky Potential (Standard Form)
    # V = (aw cos)^2 - 2 aw s cos + s + A - (m + s cos)^2 / sin^2
    # 注意这里 m 用 m_eff (-m)
    V_standard = (
        (aomega * ct)**2 
        - 2*aomega*s*ct 
        + s + A_teuk 
        - ((m_eff + s*ct)**2)/(st**2)
    )

    resid = S_th2 + cot * S_th + V_standard * S[1:-1]
    return float(np.max(np.abs(resid)))

def check_Ldag_eq34(swsh, n_theta=2001):
    """
    验证算符 L^dag。
    同样需要使用 m_eff = -m 来构建数值算符。
    """
    aomega = swsh.aw
    m_eff = swsh.m 
    s = swsh.s

    thetas = np.linspace(1e-4, np.pi - 1e-4, n_theta)
    h = thetas[1] - thetas[0]
    S = np.array([swsh.evaluate_S(np.cos(th)) for th in thetas], dtype=np.complex128)
    S_th = (S[2:] - S[:-2]) / (2*h)

    th = thetas[1:-1]
    st = np.sin(th)
    ct = np.cos(th)

    # LRR Eq 34 (Applied to m_eff)
    # L^dag = d/dth - m/sin + s cot + aw sin
    L_num = S_th - (m_eff/st)*S[1:-1] + aomega*st*S[1:-1] + s*(ct/st)*S[1:-1]

    L_cpp = np.array([swsh.evaluate_L2dag_S(np.cos(th_i)) for th_i in th], dtype=np.complex128)

    return float(np.max(np.abs(L_cpp - L_num)))

def check_theta_normalization(swsh):
    thetas = np.linspace(1e-6, np.pi - 1e-6, 4001)
    S = np.array([swsh.evaluate_S(np.cos(th)) for th in thetas], dtype=np.complex128)
    return np.trapz(np.abs(S)**2 * np.sin(thetas), thetas)

if __name__ == "__main__":
    s = -2
    l = 2
    m = 2
    aomega = 0.1

    print(f"Testing SWSH s={s}, l={l}, m={m}, aw={aomega}")
    swsh = _core.SWSH(s, l, m, aomega)
    
    print(f"E      = {swsh.E}")
    print(f"Lambda = {swsh.m_lambda}")

    norm = check_theta_normalization(swsh)
    print(f"Normalization: {norm:.6f}")

    res = check_lrr_angular_eq15_residual(swsh)
    print(f"ODE Residual:  {res:.3e}")

    errL = check_Ldag_eq34(swsh)
    print(f"Op Error L^dag:{errL:.3e}")