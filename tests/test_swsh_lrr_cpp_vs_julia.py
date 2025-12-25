import numpy as np
import sys
import os
# ===== Julia reference (SpinWeightedSpheroidalHarmonics.jl) =====
JULIA_AVAILABLE = True
try:
    from juliacall import Main as jl
except Exception as e:
    JULIA_AVAILABLE = False
    jl = None
    print(f"[Julia] juliacall not available: {e}")

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


# -----------------------------
# 小工具：更稳的 trapezoid
# -----------------------------
def trapz(y, x):
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x)
    return np.trapz(y, x)


# -----------------------------
# 读取 swsh 属性（兼容不同绑定名）
# -----------------------------
def get_attr(swsh, name, fallback=None):
    if hasattr(swsh, name):
        return getattr(swsh, name)
    if fallback is not None and hasattr(swsh, fallback):
        return getattr(swsh, fallback)
    raise AttributeError(f"SWSH object has neither '{name}' nor '{fallback}'")


def get_params(swsh):
    # 按你给的脚本习惯：swsh.aw, swsh.m, swsh.s, swsh.m_lambda, swsh.E
    aomega = get_attr(swsh, "aw", "aomega")
    m = get_attr(swsh, "m")
    s = get_attr(swsh, "s")
    lam = get_attr(swsh, "m_lambda", "lambda_")
    E = get_attr(swsh, "E", "m_E")  # 有些绑定会叫 E 或 m_E
    return s, m, aomega, lam, E


# -----------------------------
# 有限差分：一阶/二阶 θ 导数
# -----------------------------
def finite_diff_theta(S, h):
    # central differences for interior points
    S_th = (S[2:] - S[:-2]) / (2.0 * h)
    S_th2 = (S[2:] - 2.0 * S[1:-1] + S[:-2]) / (h * h)
    return S_th, S_th2


# -----------------------------
# 计算 A (LRR/Julia angular_sep_const) 的一致性检查
# -----------------------------
def compute_A_from_lambda(lam, m_eff, aomega):
    # Julia/LRR convention: lambda = A + c^2 - 2 m c , c=aomega
    # => A = lambda - c^2 + 2 m c
    c = aomega
    return lam - c * c + 2.0 * m_eff * c


def compute_A_from_E_internal(E_internal, s):
    # E_internal = A + s(s+1)
    return E_internal - s * (s + 1)


# -----------------------------
# LRR Eq.(15) 残差（θ-only 形式）
# -----------------------------
def check_lrr_angular_eq15_residual(swsh, m_eff, n_theta=4001, eps=1e-6):
    r"""
    验证 LRR Eq.(15) 的 ODE 残差（对 θ-only 的 {}_sS_{lm}(θ) ）：
        S'' + cotθ S' + [ (c cosθ)^2 - 2 c s cosθ + s + A - (m + s cosθ)^2/sin^2θ ] S = 0

    其中 A 为 angular separation constant（Julia: angular_sep_const）。
    注意：m_eff 的符号必须与你的 C++/Julia 在 φ 相位约定一致。
    """
    s, m_phys, aomega, lam, E_internal = get_params(swsh)

    # 用 lambda 推回 A（这是最不容易混的）
    A = compute_A_from_lambda(lam, m_eff, aomega)

    thetas = np.linspace(eps, np.pi - eps, n_theta)
    h = thetas[1] - thetas[0]

    xs = np.cos(thetas)
    S = np.array([swsh.evaluate_S(x) for x in xs], dtype=np.complex128)

    S_th, S_th2 = finite_diff_theta(S, h)

    th = thetas[1:-1]
    st = np.sin(th)
    ct = np.cos(th)
    cot = ct / st

    V = (
        (aomega * ct) ** 2
        - 2.0 * aomega * s * ct
        + s
        + A
        - ((m_eff + s * ct) ** 2) / (st ** 2)
    )

    resid = S_th2 + cot * S_th + V * S[1:-1]
    imax = np.argmax(np.abs(resid))
    print("theta@maxres =", th[imax], "maxres =", resid[imax], "S =", S[1:-1][imax], "V =", V[imax])

    return float(np.max(np.abs(resid)))


def pick_m_eff_sign(swsh, n_theta=2001):
    s, m, aomega, lam, E_internal = get_params(swsh)

    r1 = check_lrr_angular_eq15_residual(swsh, m_eff=+m, n_theta=n_theta)
    r2 = check_lrr_angular_eq15_residual(swsh, m_eff=-m, n_theta=n_theta)

    if r1 <= r2:
        return +1, r1, r2
    return -1, r2, r1


# -----------------------------
# LRR Eq.(34) 算符测试：L_sop^dag
# -----------------------------
def Ldag_num(S, S_th, th, m_eff, aomega, sop):
    # LRR Eq.(34): L_sop^dag = d/dθ - m/sinθ + c sinθ + sop cotθ
    st = np.sin(th)
    ct = np.cos(th)
    return S_th - (m_eff / st) * S + aomega * st * S + sop * (ct / st) * S


def check_Ldag_eq34_L2(swsh, m_eff, n_theta=2001, eps=1e-6):
    r"""
    测试 C++ 的 evaluate_L2dag_S 是否等于数值实现的 L_2^dag S。
    这里 sop=2（不是 swsh.s=-2！）
    """
    s, m_phys, aomega, lam, E_internal = get_params(swsh)

    thetas = np.linspace(eps, np.pi - eps, n_theta)
    h = thetas[1] - thetas[0]
    xs = np.cos(thetas)

    S = np.array([swsh.evaluate_S(x) for x in xs], dtype=np.complex128)
    S_th = (S[2:] - S[:-2]) / (2.0 * h)

    th = thetas[1:-1]
    L_num = Ldag_num(S[1:-1], S_th, th, m_eff=m_eff, aomega=aomega, sop=2)

    # C++ evaluate_L2dag_S
    L_cpp = np.array([swsh.evaluate_L2dag_S(np.cos(t)) for t in th], dtype=np.complex128)

    return float(np.max(np.abs(L_cpp - L_num)))


def check_L1L2_eq34(swsh, m_eff, n_theta=2001, eps=1e-6):
    r"""
    测试 C++ evaluate_L1dag_L2dag_S 是否等于数值实现的 L_1^dag L_2^dag S。
    数值实现方式：先用差分算 S'，构造 L2_num；再差分 L2_num 得到 (L2_num)'，构造 L1(L2_num)。
    """
    s, m_phys, aomega, lam, E_internal = get_params(swsh)

    thetas = np.linspace(eps, np.pi - eps, n_theta)
    h = thetas[1] - thetas[0]
    xs = np.cos(thetas)

    S = np.array([swsh.evaluate_S(x) for x in xs], dtype=np.complex128)

    # S' on interior
    S_th = (S[2:] - S[:-2]) / (2.0 * h)
    th = thetas[1:-1]

    # L2_num on same interior grid
    L2_num = Ldag_num(S[1:-1], S_th, th, m_eff=m_eff, aomega=aomega, sop=2)

    # Now take derivative of L2_num (need second interior: indices 1..-2)
    L2_th = (L2_num[2:] - L2_num[:-2]) / (2.0 * h)
    th2 = th[1:-1]  # aligned with L2_th

    # Apply L1^dag to L2_num on th2 grid
    # L1^dag g = g' - m/sin g + c sin g + 1 cot g
    st2 = np.sin(th2)
    ct2 = np.cos(th2)
    L1L2_num = L2_th - (m_eff / st2) * L2_num[1:-1] + aomega * st2 * L2_num[1:-1] + 1.0 * (ct2 / st2) * L2_num[1:-1]

    # C++ evaluate_L1dag_L2dag_S on th2 grid
    L1L2_cpp = np.array([swsh.evaluate_L1dag_L2dag_S(np.cos(t)) for t in th2], dtype=np.complex128)

    return float(np.max(np.abs(L1L2_cpp - L1L2_num)))


# -----------------------------
# 由 C++ 算符反解 θ 导数（便于对比 Julia）
# -----------------------------
def theta_derivatives_from_cpp(swsh, m_eff, thetas):
    r"""
    给定 θ 网格，使用 C++ 的 evaluate_S / evaluate_L2dag_S / evaluate_L1dag_L2dag_S
    反解得到 S', S''（全部是 θ-only 定义）。
    公式：
      L2 = L_2^dag S = S' + P2 S
      => S' = L2 - P2 S
      其中 P2 = -m/sinθ + c sinθ + 2 cotθ

      L1L2 = L_1^dag(L_2^dag S) = S'' + (P1+P2)S' + (P2' + P1 P2)S
      => S'' = L1L2 - (P1+P2)S' - (P2' + P1 P2)S
      P1 = -m/sinθ + c sinθ + 1 cotθ
      P2' = m cscθ cotθ + c cosθ - 2 csc^2θ
    """
    s, m_phys, aomega, lam, E_internal = get_params(swsh)
    c = aomega

    xs = np.cos(thetas)
    S = np.array([swsh.evaluate_S(x) for x in xs], dtype=np.complex128)
    L2 = np.array([swsh.evaluate_L2dag_S(x) for x in xs], dtype=np.complex128)
    L1L2 = np.array([swsh.evaluate_L1dag_L2dag_S(x) for x in xs], dtype=np.complex128)

    st = np.sin(thetas)
    ct = np.cos(thetas)
    inv_st = 1.0 / st
    cot = ct * inv_st
    csc2 = inv_st * inv_st

    P2 = -m_eff * inv_st + c * st + 2.0 * cot
    P1 = -m_eff * inv_st + c * st + 1.0 * cot
    P2p = m_eff * inv_st * cot + c * ct - 2.0 * csc2

    S_th = L2 - P2 * S
    S_thth = L1L2 - (P1 + P2) * S_th - (P2p + P1 * P2) * S

    return S, S_th, S_thth


def check_theta_derivatives(swsh, m_eff, n_theta=4001, eps=1e-6):
    r"""
    对比：
      - 差分得到的 S', S''  vs
      - 由 C++ 算符反解得到的 S', S''
    """
    thetas = np.linspace(eps, np.pi - eps, n_theta)
    h = thetas[1] - thetas[0]
    xs = np.cos(thetas)

    S = np.array([swsh.evaluate_S(x) for x in xs], dtype=np.complex128)
    S_th_fd, S_thth_fd = finite_diff_theta(S, h)  # on interior

    # For fair compare, evaluate cpp-derived derivatives on the same interior points
    th_int = thetas[1:-1]
    S_int, S_th_cpp, S_thth_cpp = theta_derivatives_from_cpp(swsh, m_eff, th_int)

    err1 = float(np.max(np.abs(S_th_cpp - S_th_fd)))
    err2 = float(np.max(np.abs(S_thth_cpp - S_thth_fd)))
    return err1, err2


# -----------------------------
# 归一化（θ-only）
# -----------------------------
def check_theta_normalization(swsh, n_theta=4001, eps=1e-8):
    thetas = np.linspace(eps, np.pi - eps, n_theta)
    xs = np.cos(thetas)
    S = np.array([swsh.evaluate_S(x) for x in xs], dtype=np.complex128)
    val = trapz(np.abs(S) ** 2 * np.sin(thetas), thetas)
    return float(val)


# -----------------------------
# 输出便于 Julia 对比的本征值信息
# -----------------------------
def print_eigenvalue_report(swsh, m_eff):
    s, m, aomega, lam, E_internal = get_params(swsh)

    A_from_lam = compute_A_from_lambda(lam, m_eff, aomega)
    A_from_E = compute_A_from_E_internal(E_internal, s)

    print("=== Eigenvalue report (for Julia/LRR compare) ===")
    print(f"s={s}, m={m}, aω={aomega}")
    print(f"lambda (LRR/Teuk) = {lam:.15g}")
    print(f"E_internal        = {E_internal:.15g}")
    print(f"A from lambda     = {A_from_lam:.15g}   (Julia: angular_sep_const)")
    print(f"A from E_internal = {A_from_E:.15g}   (should match A from lambda)")
    print(f"|A(lam)-A(E)|      = {abs(A_from_lam - A_from_E):.3e}")
    print("==============================================")


# -----------------------------
# 主程序
# -----------------------------
if __name__ == "__main__":
    s = -2
    l = 2
    m = 2
    aomega = 0.1

    print(f"Testing SWSH s={s}, l={l}, m={m}, aw={aomega}")
    swsh = _core.SWSH(s, l, m, aomega)

    # 打印绑定对象的核心量
    s0, m0, aw0, lam0, E0 = get_params(swsh)
    print(f"E_internal (swsh.E)      = {E0}")
    print(f"lambda     (swsh.m_lambda)= {lam0}")

    # 自动选择 m_eff 符号
    sign, r_best, r_other = pick_m_eff_sign(swsh, n_theta=2001)
    m_eff = sign * m0
    print("\n=== m convention check ===")
    print(f"Residual with m_eff = +m : {check_lrr_angular_eq15_residual(swsh, +m0, n_theta=2001):.3e}")
    print(f"Residual with m_eff = -m : {check_lrr_angular_eq15_residual(swsh, -m0, n_theta=2001):.3e}")
    print(f"--> choose m_eff = {m_eff} (sign={sign:+d})")
    print("==========================\n")
    # ===== Julia compare block (aligned with chosen m_eff) =====
    if JULIA_AVAILABLE:
        # 1) load package
        jl.seval("using SpinWeightedSpheroidalHarmonics")

        # 2) IMPORTANT: use m_jl = m_eff to align operator sign convention
        s_jl = int(s0)
        l_jl = int(l)
        m_jl = int(m_eff)      # <-- align with C++ effective m
        c_jl = float(aw0)      # a*omega

        # 建议：把 N 设大一些做收敛（Julia 默认 N 很小）
        N_JL = 80  # 可从 40/60/80/120 试，看 A/E 是否收敛到 C++

        # Julia 返回的是 Teukolsky_lambda_const = A + c^2 - 2 m c
        lam_teuk_jl = float(jl.spin_weighted_spheroidal_eigenvalue(s_jl, l_jl, m_jl, c_jl, N=N_JL))

        # 还原 angular_sep_const A
        A_jl = lam_teuk_jl - c_jl**2 + 2.0*m_jl*c_jl

        # LRR 里趋于 l(l+1) 的本征值（你 C++ 的 E_internal 就是这个）
        E_lrr_jl = A_jl + s_jl*(s_jl + 1)

        print("\n=== Julia reference (SpinWeightedSpheroidalHarmonics.jl) ===")
        print(f"Julia params: s={s_jl}, l={l_jl}, m={m_jl}, c={c_jl}, N={N_JL}")
        print(f"Julia lambda_Teuk (Teukolsky_lambda_const) = {lam_teuk_jl:.15g}")
        print(f"Julia A (angular_sep_const)               = {A_jl:.15g}")
        print(f"Julia E_LRR (=A+s(s+1), ->l(l+1))         = {E_lrr_jl:.15g}")

        

        # build callable harmonic
        swsh_jl = jl.spin_weighted_spheroidal_harmonic(s_jl, l_jl, m_jl, c_jl, N=N_JL)


        # 3) mapping: Julia returns full S(θ,φ) normalized on sphere:
        #      ∫ |S|^2 dΩ = 1
        #    Your C++ (LRR-aligned theta-only) expects:
        #      ∫_0^π |S(θ)|^2 sinθ dθ = 1
        #    If S_full(θ,φ)=Sθ(θ)e^{imφ}/sqrt(2π), then:
        #      S_theta_only(θ) = sqrt(2π) * S_full(θ, 0)
        SQRT_2PI = np.sqrt(2.0*np.pi)

        def jl_theta_only(theta):
            return SQRT_2PI * np.complex128(swsh_jl(theta, 0.0))

        def jl_theta_only_dtheta(theta):
            return SQRT_2PI * np.complex128(swsh_jl(theta, 0.0, theta_derivative=1, phi_derivative=0))

        def jl_full(theta, phi):
            return np.complex128(swsh_jl(theta, phi))

        def jl_full_dphi(theta, phi):
            return np.complex128(swsh_jl(theta, phi, theta_derivative=0, phi_derivative=1))

        # 4) compare pointwise (same theta as you already print, or pick new ones)
        th_cmp = np.pi/6
        phi_cmp = np.pi/3

        S_cpp = np.complex128(swsh.evaluate_S(np.cos(th_cmp)))
        S_jl_th = jl_theta_only(th_cmp)

        # θ derivative:
        # reuse your existing helper (recommended): theta_derivatives_from_cpp(swsh, m_eff, thetas)
        # If you didn't keep it, you can infer from L2dag via LRR Eq.(34):
        st = np.sin(th_cmp); ct = np.cos(th_cmp)
        P2 = -m_eff/st + aw0*st + 2.0*(ct/st)
        L2_cpp = np.complex128(swsh.evaluate_L2dag_S(np.cos(th_cmp)))
        dS_cpp = L2_cpp - P2*S_cpp

        dS_jl = jl_theta_only_dtheta(th_cmp)
        # phase = S_cpp / S_jl_th   # 复数整体相位因子
        # # 之后所有 Julia 值都乘 phase 再比较
        # S_jl_th *= phase
        # dS_jl   *= phase


        print("\n=== Pointwise compare (theta-only) ===")
        print(f"theta={th_cmp}, (phi used only for Julia full test: {phi_cmp})")
        print(f"S_cpp(theta)         = {S_cpp}")
        print(f"S_jl_thetaOnly(theta)= {S_jl_th}")
        print(f"|S_cpp-S_jl|          = {abs(S_cpp-S_jl_th):.3e}")
        print(f"dS/dθ_cpp(theta)     = {dS_cpp}")
        print(f"dS/dθ_jl(theta)      = {dS_jl}")
        print(f"|dS_cpp-dS_jl|        = {abs(dS_cpp-dS_jl):.3e}")

        # 5) φ derivative on Julia full function (should satisfy ∂φ S = i m S for e^{imφ})
        val0 = jl_full(th_cmp, phi_cmp)
        val_dphi = jl_full_dphi(th_cmp, phi_cmp)
        print("\n=== Julia full S(theta,phi) phase check ===")
        print(f"S_jl_full(theta,phi)     = {val0}")
        print(f"∂φ S_jl_full(theta,phi)  = {val_dphi}")
        print(f"i*m*S                    = {1j*m_jl*val0}")
        print(f"|∂φS - i m S|            = {abs(val_dphi - 1j*m_jl*val0):.3e}")

        # 6) eigenvalue consistency against C++ derived A
        # C++: A = lambda - c^2 + 2 m_eff c
        A_cpp = lam0 - aw0**2 + 2.0*m_eff*aw0
        print("\n=== Eigenvalue compare ===")
        print(f"lambda_Teuk_cpp (swsh.m_lambda) = {lam0:.15g}")
        print(f"lambda_Teuk_jl                  = {lam_teuk_jl:.15g}")
        print(f"|Δlambda_Teuk|                  = {abs(lam0-lam_teuk_jl):.3e}")

        print(f"A_cpp (from lambda_Teuk)        = {A_cpp:.15g}")
        print(f"A_jl (angular_sep_const)        = {A_jl:.15g}")
        print(f"|ΔA|                            = {abs(A_cpp-A_jl):.3e}")

        print(f"E_LRR_cpp (swsh.E)              = {E0:.15g}")
        print(f"E_LRR_jl (=A+s(s+1))            = {E_lrr_jl:.15g}")
        print(f"|ΔE_LRR|                        = {abs(E0-E_lrr_jl):.3e}")

        print("=========================================\n")
            # ===== Grid-level compare for theta-only S and dS/dθ =====
        n_grid = 2001
        eps = 1e-6
        thetas_g = np.linspace(eps, np.pi-eps, n_grid)

        # C++: infer S and dS/dθ from operator at grid
        xs_g = np.cos(thetas_g)
        S_cpp_g = np.array([swsh.evaluate_S(x) for x in xs_g], dtype=np.complex128)

        st_g = np.sin(thetas_g); ct_g = np.cos(thetas_g)
        P2_g = -m_eff/st_g + aw0*st_g + 2.0*(ct_g/st_g)
        L2_cpp_g = np.array([swsh.evaluate_L2dag_S(x) for x in xs_g], dtype=np.complex128)
        dS_cpp_g = L2_cpp_g - P2_g*S_cpp_g

        # Julia theta-only
        S_jl_g = np.array([jl_theta_only(th) for th in thetas_g], dtype=np.complex128)
        dS_jl_g = np.array([jl_theta_only_dtheta(th) for th in thetas_g], dtype=np.complex128)

        print("=== Grid compare (theta-only) ===")
        print(f"max|S_cpp - S_jl|      = {np.max(np.abs(S_cpp_g - S_jl_g)):.3e}")
        print(f"max|dS_cpp - dS_jl|    = {np.max(np.abs(dS_cpp_g - dS_jl_g)):.3e}")
        print("================================")

    # 归一化
    norm = check_theta_normalization(swsh)
    print(f"[1] theta-normalization  ∫|S|^2 sinθ dθ = {norm:.12f}")

    # 本征值一致性报告（便于对 Julia）
    print_eigenvalue_report(swsh, m_eff)

    # LRR Eq.(15) 残差
    res = check_lrr_angular_eq15_residual(swsh, m_eff=m_eff, n_theta=4001)
    print(f"[2] LRR Eq.(15) max residual = {res:.3e}")  
    

    # LRR Eq.(34) 算符误差：L2dag
    errL2 = check_Ldag_eq34_L2(swsh, m_eff=m_eff, n_theta=2001)
    print(f"[3] LRR Eq.(34)  L_2^dag error = {errL2:.3e}")

    # LRR Eq.(34) 嵌套算符误差：L1dag L2dag
    errL1L2 = check_L1L2_eq34(swsh, m_eff=m_eff, n_theta=2001)
    print(f"[4] LRR Eq.(34)  L_1^dag L_2^dag error = {errL1L2:.3e}")

    # 单独测试 θ 导数（FD vs 由算符反解）
    derr1, derr2 = check_theta_derivatives(swsh, m_eff=m_eff, n_theta=4001)
    print(f"[5] dS/dθ   error (cpp-inferred vs FD) = {derr1:.3e}")
    print(f"[6] d2S/dθ2 error (cpp-inferred vs FD) = {derr2:.3e}")

    # 给一个固定角度点的数值，方便你和 Julia 打印对比
    th0 = np.pi / 3
    S0 = complex(swsh.evaluate_S(np.cos(th0)))
    L20 = complex(swsh.evaluate_L2dag_S(np.cos(th0)))
    L1L20 = complex(swsh.evaluate_L1dag_L2dag_S(np.cos(th0)))
    print("\n=== Pointwise values for Julia compare (theta=pi/3) ===")
    print(f"S(theta)         = {S0}")
    print(f"L2dag S(theta)   = {L20}")
    print(f"L1dagL2dag S     = {L1L20}")
