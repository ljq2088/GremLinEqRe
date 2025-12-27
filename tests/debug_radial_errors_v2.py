import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import cmath

# =========================================================================
# 0. 环境设置与导入
# =========================================================================
# 尝试定位并导入 C++ 模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../python')))

try:
    from GremLinEqRe import _core
except ImportError:
    print("❌ Error: 无法导入 C++ 模块 GremLinEqRe._core")
    print("请确保在编译目录下运行或正确设置了 PYTHONPATH")
    sys.exit(1)

# =========================================================================
# 1. 辅助工具: 因子 P 计算 (用于从 R 中分离出 p_in 和 f)
# =========================================================================

def calc_P_near(r, M, a, omega, s, epsilon, tau, kappa):
    """
    计算近场分解因子 P_in(x). 
    对应 LRR-2003-6 Eq. 116: R_in = P_in * p_in(x)
    """
    # 坐标变换变量 x
    r_plus = 1.0 + np.sqrt(1.0 - (a/M)**2)
    x = (r_plus - r) / (2.0 * kappa)
    
    if abs(x) < 1e-16: return 0.0 + 0.0j
    
    i = 1.0j
    # 指数参数
    alpha = -float(s) - i * (epsilon + tau) / 2.0
    beta  = i * (epsilon - tau) / 2.0
    
    # P_in = e^{i * eps * kappa * x} * (-x)^alpha * (1-x)^beta
    term1 = i * epsilon * kappa * x
    term2 = alpha * cmath.log(-x)      # 注意: r > r_+ 时 x < 0，log(-x) 为实数对数
    term3 = beta * cmath.log(1.0 - x)
    
    return cmath.exp(term1 + term2 + term3)

def calc_P_far(r, M, a, omega, s, epsilon, tau, kappa):
    """
    计算远场分解因子 P_C(z).
    对应 LRR-2003-6 Eq. 139: R_C = P_C * f(z)
    """
    # 坐标变换变量 z
    # 注意: LRR 中 z_hat = omega * (r - r_-)
    q = a 
    r_minus = 1.0 - np.sqrt(1.0 - q*q)
    z_hat = omega * (r - r_minus)
    
    if abs(z_hat) < 1e-16: return 0.0 + 0.0j

    i = 1.0j
    factor1 = -1.0 - float(s)
    factor2 = -float(s) - i * (epsilon + tau) / 2.0
    
    term_brace = 1.0 - epsilon * kappa / z_hat
    
    # P_C = z^{factor1} * (1 - eps*kappa/z)^{factor2}
    LogP = factor1 * cmath.log(z_hat) + factor2 * cmath.log(term_brace)
    
    return cmath.exp(LogP)

# =========================================================================
# 2. 核心诊断: 方程残差检查 (用户填空区)
# =========================================================================

def check_eq119_residual(r, p_in, dp_in_dr, d2p_in_dr2, params):
    """
    验证近场级数部分 p_in(x) 是否满足 LRR Eq. 119。
    输入:
        r: 当前半径
        p_in: p_in(x) 的数值
        dp_in_dr: p_in 对 r 的一阶导 (数值差分得到)
        d2p_in_dr2: p_in 对 r 的二阶导 (数值差分得到)
        params: 物理参数元组
    """
    M, a, omega, s, l, m, epsilon, kappa, tau, nu,m_lambda = params
    i = 1.0j

    # --- 1. 坐标变换准备 ---
    # r -> x
    r_plus = 1.0 + np.sqrt(1.0 - (a/M)**2)
    x = (r_plus - r) / (2.0 * kappa)
    
    # 链式法则因子: d/dr = (dx/dr) * d/dx
    dx_dr = -1.0 / (2.0 * kappa)
    
    # 转换为对 x 的导数
    dy_dx = dp_in_dr / dx_dr
    d2y_dx2 = d2p_in_dr2 / (dx_dr**2)
    
    y = p_in

    # --- 2. 构造方程 (请在此处填入 LRR Eq. 119) ---
    # 目标方程形式: Coeff2 * y'' + Coeff1 * y' + Coeff0 * y = 0
    

    # coeff_2 = x * (1.0 - x)
    # coeff_1 = ... 
    # coeff_0 = ...
    
    coeff_2 = x*(1-x) # 
    coeff_1 = 1-s-i*epsilon-i*tau-(2-2*i*tau)*x
    coeff_0 = i*tau*(1-i*tau)+nu*(nu+1)
    
    # 计算方程左边 (LHS)
    lhs = coeff_2 * d2y_dx2 + coeff_1 * dy_dx + coeff_0 * y
    rhs=2*i*epsilon*kappa*(-x*(1.0-x)*dy_dx+(1-s+i*epsilon-i*tau)*x*y)+(nu*(nu+1)-m_lambda-s*(s+1)+epsilon**2-i*epsilon*kappa*(1-2*s))*y
    # 归一化残差 (防止 y 过小时除零)
    scale = abs(y) if abs(y) > 1e-15 else 1.0
    return (rhs - lhs) / scale


def check_eq140_residual(r, f_val, df_dr, d2f_dr2, params):
    """
    验证远场级数部分 f(z) 是否满足 LRR Eq. 140。
    输入: 
        r, f, df/dr, d^2f/dr^2 (同上)
    """
    M, a, omega, s, l, m, epsilon, kappa, tau, nu,m_lambda = params
    i = 1.0j

    # --- 1. 坐标变换准备 ---
    # r -> z
    r_minus = 1.0 - np.sqrt(1.0 - (a/M)**2)
    z = omega * (r - r_minus)
    
    # 链式法则因子: d/dr = (dz/dr) * d/dz
    dz_dr = omega
    
    # 转换为对 z 的导数
    dy_dz = df_dr / dz_dr
    d2y_dz2 = d2f_dr2 / (dz_dr**2)
    
    y = f_val

    # --- 2. 构造方程 (请在此处填入 LRR Eq. 140) ---
    # 目标方程形式: Coeff2 * y'' + Coeff1 * y' + Coeff0 * y = 0
    

    q=a
    coeff_2 = z*(z-epsilon*kappa)
    coeff_1 = -epsilon*kappa*(s-1+2*i*epsilon)
    coeff_0 = z**2+(2*i*s+2*epsilon)*z-m_lambda-s*(s+1)-epsilon*kappa*z+epsilon*(kappa-i*(epsilon-m*q))*(s+i*epsilon-1)/z-(-2*epsilon*epsilon+epsilon*m*q+kappa*(epsilon*epsilon+i*epsilon*s))
    
    # 计算方程左边 (LHS)
    lhs = coeff_2 * d2y_dz2 + coeff_1 * dy_dz + coeff_0 * y
    
    scale = abs(y) if abs(y) > 1e-15 else 1.0
    return lhs / scale

# =========================================================================
# 3. 主分析脚本
# =========================================================================

def analyze_residuals_full():
    print("=========================================================")
    print("   Teukolsky Radial Residual Analysis (LRR Exact Eq)")
    print("=========================================================")

    # 1. 物理参数
    M, a, omega = 1.0, 0.9, 0.3
    s, l, m = -2, 2, 2
    
    print(f"Params: a={a}, w={omega}, s={s}, l={l}, m={m}")

    # 2. 求解器初始化
    swsh = _core.SWSH(s, l, m, a * omega)
    lambda_val = swsh.m_lambda
    tr = _core.TeukolskyRadial(M, a, omega, s, l, m, lambda_val)
    
    # 3. 准备中间变量
    epsilon = 2 * omega
    kappa = np.sqrt(1.0 - a**2)
    tau = (epsilon - m*a) / kappa
    
    # 4. 求解特征值 nu
    print("Solving nu...")
    nu = tr.solve_nu(complex(float(l), 0.1))
    print(f"nu = {nu}")
    
    # 5. 计算级数系数
    n_max = 60
    a_coeffs_pos = tr.ComputeSeriesCoefficients(nu, n_max)
    
    # 打包参数供 residual 函数使用
    params = (M, a, omega, s, l, m, epsilon, kappa, tau, nu,lambda_val)

    # -----------------------------------------------------------
    # A. 检查近场方程 (Eq. 119)
    # -----------------------------------------------------------
    print("\nChecking Near Field (Eq. 119)...")
    r_plus = 1.0 + np.sqrt(1.0 - a**2)
    # 扫描范围: r_+ 到收敛半径附近
    r_grid_near = np.linspace(r_plus + 0.001, r_plus + 1.2*kappa, 200)
    
    p_in_list = []
    
    # A1. 计算 p_in(x) 数值
    for r in r_grid_near:
        try:
            # R_near from C++
            res = tr.Evaluate_Hypergeometric(r, nu, a_coeffs_pos)
            R_val = res[0]
            # P_in from Helper
            P_val = calc_P_near(r, M, a, omega, s, epsilon, tau, kappa)
            
            if abs(P_val) > 1e-20:
                p_in_list.append(R_val / P_val)
            else:
                p_in_list.append(np.nan)
        except:
            p_in_list.append(np.nan)
            
    p_in_arr = np.array(p_in_list)
    
    # A2. 数值差分计算导数 (对 r)
    dr_near = r_grid_near[1] - r_grid_near[0]
    dp_in_dr = np.gradient(p_in_arr, dr_near)
    d2p_in_dr2 = np.gradient(dp_in_dr, dr_near)
    
    # A3. 代入方程检查残差
    resid_119 = []
    for i, r in enumerate(r_grid_near):
        if np.isnan(p_in_arr[i]):
            resid_119.append(np.nan)
        else:
            res = check_eq119_residual(r, p_in_arr[i], dp_in_dr[i], d2p_in_dr2[i], params)
            resid_119.append(res)

    # -----------------------------------------------------------
    # B. 检查远场方程 (Eq. 140)
    # -----------------------------------------------------------
    print("Checking Far Field (Eq. 140)...")
    # 扫描范围: 远场区域
    r_grid_far = np.linspace(r_plus + 1.5*kappa, r_plus + 8.0, 200)
    
    f_list = []
    
    # B1. 计算 f(z) 数值
    for r in r_grid_far:
        try:
            # R_far (Coulomb) from C++
            res = tr.Evaluate_Coulomb(r, nu, a_coeffs_pos)
            R_C_val = res[0]
            # P_C from Helper
            P_C_val = calc_P_far(r, M, a, omega, s, epsilon, tau, kappa)
            
            if abs(P_C_val) > 1e-20:
                f_list.append(R_C_val / P_C_val)
            else:
                f_list.append(np.nan)
        except:
            f_list.append(np.nan)
            
    f_arr = np.array(f_list)
    
    # B2. 数值差分计算导数 (对 r)
    dr_far = r_grid_far[1] - r_grid_far[0]
    df_dr = np.gradient(f_arr, dr_far)
    d2f_dr2 = np.gradient(df_dr, dr_far)
    
    # B3. 代入方程检查残差
    resid_140 = []
    for i, r in enumerate(r_grid_far):
        if np.isnan(f_arr[i]):
            resid_140.append(np.nan)
        else:
            res = check_eq140_residual(r, f_arr[i], df_dr[i], d2f_dr2[i], params)
            resid_140.append(res)

    # -----------------------------------------------------------
    # C. 绘图结果
    # -----------------------------------------------------------
    plt.figure(figsize=(12, 10))
    
    # Plot 1: Near Field Residual
    plt.subplot(2, 1, 1)
    plt.title("Residual of LRR Eq. 119 (Near Field p_in)")
    plt.plot(r_grid_near, np.abs(resid_119), 'k-', label='|LHS|')
    plt.plot(r_grid_near, np.real(resid_119), 'r--', alpha=0.5, label='Real(LHS)')
    plt.plot(r_grid_near, np.imag(resid_119), 'b--', alpha=0.5, label='Imag(LHS)')
    plt.yscale('log')
    plt.xlabel('r/M')
    plt.ylabel('Residual')
    plt.legend()
    plt.grid(True)
    
    # Plot 2: Far Field Residual
    plt.subplot(2, 1, 2)
    plt.title("Residual of LRR Eq. 140 (Far Field f)")
    plt.plot(r_grid_far, np.abs(resid_140), 'k-', label='|LHS|')
    plt.plot(r_grid_far, np.real(resid_140), 'r--', alpha=0.5, label='Real(LHS)')
    plt.plot(r_grid_far, np.imag(resid_140), 'b--', alpha=0.5, label='Imag(LHS)')
    plt.yscale('log')
    plt.xlabel('r/M')
    plt.ylabel('Residual')
    plt.legend()
    plt.grid(True)
    
    outfile = "radial_residual_check_full.png"
    plt.tight_layout()
    plt.savefig(outfile)
    print(f"\nAnalysis complete. Plot saved to {outfile}")

if __name__ == "__main__":
    analyze_residuals_full()