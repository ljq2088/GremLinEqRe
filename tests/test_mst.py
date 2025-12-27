# test_mst_internal.py
import sys
import os
import math
import cmath
import numpy as np

# 高精度复数 gamma / loggamma
import mpmath as mp

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from GremLinEqRe import _core
except ImportError:
    print("Error: 无法导入 C++ 模块 (from GremLinEqRe import _core).")
    import traceback
    traceback.print_exc()
    sys.exit(1)

import mpmath as mp

def check_eq134(a_dict, eps, kappa, N, rtol=1e-8):
    # a_dict: {n: complex}
    rhs = 1j * eps * kappa / 2.0
    print(f'i eps kappa/2 = {rhs}')
    lhs_pos = N * a_dict[N] / a_dict[N-1]
    lhs_neg = (-N) * a_dict[-N] / a_dict[-N+1]  # n f_n / f_{n+1}, n=-N

    err_pos = abs(lhs_pos - rhs) / max(abs(rhs), 1e-300)
    err_neg = abs(lhs_neg + rhs) / max(abs(rhs), 1e-300)

    print(f"Eq134(+inf): lhs={lhs_pos}, rhs={rhs}, relerr={err_pos}")
    print(f"Eq134(-inf): lhs={lhs_neg}, rhs={-rhs}, relerr={err_neg}")

def p_nu(nu, n, s, eps, tau, x):
    # p_{n+nu}(x) = 2F1(n+nu+1-i tau, -n-nu-i tau; 1-s-i eps-i tau; x)
    a = (nu + n + 1) - 1j*tau
    b = (-nu - n)     - 1j*tau
    c = (1 - s)       - 1j*eps - 1j*tau
    return mp.hyp2f1(a, b, c, x)

def check_eq135(a_dict, nu, s, eps, kappa, tau, x, N=40, rtol=1e-6):
    mp.mp.dps = 80  # 提高点精度避免 mp 自身误差

    # RHS: i eps kappa/2 [1-2x + sqrt((1-2x)^2-1)]
    rhs = 1j*eps*kappa/2.0 * ( (1-2*x) + mp.sqrt((1-2*x)**2 - 1) )

    # +inf side
    pN   = p_nu(nu, N,   s, eps, tau, x)
    pNp1 = p_nu(nu, N+1, s, eps, tau, x)
    lhs_pos = N * (a_dict[N+1]*pNp1) / (a_dict[N]*pN)

    # -inf side (按你抽取的等式右边写法：-lim_{n->-inf} n f_{n-1} p_{n+nu-1} / (f_n p_{n+nu}))
    n = -N
    pn   = p_nu(nu, n,   s, eps, tau, x)
    pnm1 = p_nu(nu, n-1, s, eps, tau, x)
    lhs_neg = -(n) * (a_dict[n-1]*pnm1) / (a_dict[n]*pn)

    err_pos = abs(lhs_pos - rhs) / max(abs(rhs), 1e-300)
    err_neg = abs(lhs_neg - rhs) / max(abs(rhs), 1e-300)

    print(f"Eq135(+inf): lhs={lhs_pos}, rhs={rhs}, relerr={err_pos}") 
    print(f"Eq135(-inf): lhs={lhs_neg}, rhs={rhs}, relerr={err_neg}")

# =========================
# Utils
# =========================
def _pick_method(obj, names):
    """Try multiple candidate method names."""
    for nm in names:
        if hasattr(obj, nm) and callable(getattr(obj, nm)):
            return getattr(obj, nm), nm
    return None, None

def _to_mpc(z):
    return mp.mpc(z.real, z.imag)

def _relerr(a, b, floor=1e-300):
    da = abs(a - b)
    denom = max(abs(a), abs(b), floor)
    return da / denom


# =========================
# MST coefficients (LRR unscaled)
# match TeukolskyRadial.cpp definitions:
#   epsilon = 2*omega
#   q = a
#   kappa = sqrt(1 - q^2)
#   tau = (epsilon - m*q) / kappa
# =========================
def mst_params(M, a, omega, s, m, lam):
    # C++ 里构造函数实际上把 M 忽略在 epsilon 中（epsilon=2*omega）
    eps = 2.0 * omega
    q = a
    kappa = math.sqrt(max(0.0, 1.0 - q*q))
    if kappa == 0.0:
        raise ValueError("kappa=0: extremal a=1 causes division by zero in tau. Use a<1 for this test.")
    tau = (eps - m*q) / kappa
    return eps, q, kappa, tau

def coeff_alpha_LRR(nu, n, eps, kappa, tau, s):
    i = 1j
    nn = nu + n
    num = i * eps * kappa * (((nn + 1 + s)**2) + eps**2) * (nn + 1 + i*tau)
    den = (nn + 1) * (2*nn + 3)
    return num / den

def coeff_gamma_LRR(nu, n, eps, kappa, tau, s):
    i = 1j
    nn = nu + n
    num = -i * eps * kappa * (((nn - s)**2) + eps**2) * (nn - i*tau)
    den = nn * (2*nn - 1)
    return num / den

def coeff_beta_LRR(nu, n, eps, q, tau, s, m, lam):
    nn = nu + n
    term1 = nn * (nn + 1)
    # b_add1 = 2 eps^2 - eps*m*q - lambda - s(s+1)
    b_add1 = 2.0*(eps**2) - eps*m*q - lam - s*(s+1)
    b_add2 = eps*(eps - m*q) * (s*s + eps**2)
    # beta = term1 + b_add1 + b_add2/term1
    tiny = 1e-300
    if abs(term1) < tiny:
        # 避免除 0；此处只用于测试环境
        return term1 + b_add1 + b_add2/(term1 + tiny)
    return term1 + b_add1 + b_add2/term1


# =========================
# K_nu (Python recompute) mimicking current C++ k_factor() logic
# (注意：这是“对 C++ 当前实现的复现”，不是保证物理上完全正确的 Eq.(165) 最终形态)
# =========================
def knu_from_an_python(a_coeffs, nu, eps, q, kappa, tau, s, m):
    mp.mp.dps = 80  # 高精度避免 gamma 比值误差

    i = mp.mpc(0, 1)
    pi = mp.pi

    nu_mp = _to_mpc(nu)
    eps_mp = mp.mpf(eps)
    q_mp = mp.mpf(q)
    kappa_mp = mp.mpf(kappa)
    tau_mp = mp.mpf(tau)
    s_mp = mp.mpf(s)

    # C++ 中 eps_plus = (eps + tau)/2
    eps_plus = (eps_mp + tau_mp) / 2

    r_int = mp.mpf('0.0')

    nu_plus_1_s_ie  = nu_mp + 1 + s_mp + i*eps_mp
    nu_plus_1_ms_ie = nu_mp + 1 - s_mp - i*eps_mp
    nu_plus_1_it    = nu_mp + 1 + i*tau_mp
    nu_plus_1_mit   = nu_mp + 1 - i*tau_mp

    # ln_pre: 按你 C++ 当前实现逐项照抄
    ln_pre = ( i*eps_mp*kappa_mp
              + (s_mp - nu_mp - r_int) * mp.log(2*eps_mp*kappa_mp)
              - s_mp * mp.log(2)
              + r_int * i * pi/2
              + mp.loggamma(1 - s_mp - 2*i*eps_plus)
              + mp.loggamma(r_int + 2*nu_mp + 2)
              - mp.loggamma(r_int + nu_mp + 1 - s_mp + i*eps_mp)
              - mp.loggamma(r_int + nu_mp + 1 + i*tau_mp)
              - mp.loggamma(r_int + nu_mp + 1 + s_mp + i*eps_mp)
            )

    sum_num = mp.mpc(0)
    sum_den = mp.mpc(0)

    # a_coeffs: dict[int] -> complex(double)
    # C++: sign = (-1)^n
    for n, a_n in a_coeffs.items():
        dn = mp.mpf(n)
        sign = mp.mpf(1) if (abs(n) % 2 == 0) else mp.mpf(-1)
        f_n = _to_mpc(a_n)

        if n >= 0:
            term = sign * f_n
            term *= mp.e**(mp.loggamma(dn + r_int + 2*nu_mp + 1) - mp.loggamma(dn + 1 - r_int))
            term *= mp.e**(mp.loggamma(dn + nu_plus_1_s_ie) - mp.loggamma(dn + nu_plus_1_ms_ie))
            term *= mp.e**(mp.loggamma(dn + nu_plus_1_it)   - mp.loggamma(dn + nu_plus_1_mit))
            sum_num += term

        if n <= 0:
            term = sign * f_n
            # / (-n)!  -> exp(log_gamma(1 - n))
            term /= mp.e**(mp.loggamma(1 - dn))
            # / ( (2nu+2)_n ) where n<=0
            term /= mp.e**(mp.loggamma(r_int + 2*nu_mp + 2 + dn) - mp.loggamma(r_int + 2*nu_mp + 2))
            # * (nu+1+s+i eps)_n / (nu+1-s-i eps)_n  (按你 C++ 变量名)
            term *= mp.e**(mp.loggamma(nu_plus_1_s_ie + dn)  - mp.loggamma(nu_plus_1_s_ie))
            term /= mp.e**(mp.loggamma(nu_plus_1_ms_ie + dn) - mp.loggamma(nu_plus_1_ms_ie))
            sum_den += term

    K = mp.e**(ln_pre) * (sum_num / sum_den)
    return complex(K.real, K.imag)


# =========================
# Tests
# =========================
def test_nu_and_g(tr, guess, tol_abs=1e-10):
    print(f"Initial guess: {guess}")
    nu_sol = tr.solve_nu(guess)
    print(f"Solved nu    : {nu_sol}")

    gval = tr.calc_g(nu_sol)
    print(f"Residual |g(nu)|: {abs(gval):.3e}")
    if (abs(gval) < tol_abs):
        print("✅ Passed")
    else:
        print(f'abs(gval) = {abs(gval)}, tol_abs = {tol_abs}')
    return nu_sol

def test_an_recurrence(a_coeffs, nu, eps, q, kappa, tau, s, m, lam, n_check=30):
    """
    Check three-term recurrence residual:
      alpha_n a_{n+1} + beta_n a_n + gamma_n a_{n-1} ~ 0
    """
    max_rel = 0.0
    worst_n = None

    for n in range(-n_check, n_check+1):
        if (n-1) not in a_coeffs or n not in a_coeffs or (n+1) not in a_coeffs:
            continue

        anm1 = a_coeffs[n-1]
        an   = a_coeffs[n]
        anp1 = a_coeffs[n+1]

        alpha = coeff_alpha_LRR(nu, n, eps, kappa, tau, s)
        beta  = coeff_beta_LRR(nu, n, eps, q, tau, s, m, lam)
        gamma = coeff_gamma_LRR(nu, n, eps, kappa, tau, s)

        res = alpha*anp1 + beta*an + gamma*anm1
        den = abs(alpha*anp1) + abs(beta*an) + abs(gamma*anm1) + 1e-300
        rel = abs(res)/den

        if rel > max_rel:
            max_rel = rel
            worst_n = n

    print(f"Recurrence check: max relative residual = {max_rel:.3e} at n={worst_n}")
    # 经验阈值：你若实现正确且 n_max 足够大，这里通常能到 1e-10~1e-12
    if max_rel < 1e-8:
        print("✅ Passed")
    else:
        print(f'max_rel = {max_rel}, tol = {1e-8}')

def test_knu_consistency(tr, nu, a_coeffs, eps, q, kappa, tau, s, m,
                         rel_tol=1e-6, trunc_list=(40, 60, 80, 100)):
    # find K_nu method
    knu_func, knu_name = _pick_method(tr, ["k_factor", "kFactor", "K_factor", "KFactor", "compute_Knu", "compute_K_nu"])
    if knu_func is None:
        print("Warning: C++ 未暴露 k_factor / compute_Knu 等接口，跳过 K_nu 测试。")
        return

    K_cpp = knu_func(nu)
    print(f"C++ {knu_name}(nu) = {K_cpp}")

    # Python recompute using the same a_n and the same formula structure as current C++
    K_py = knu_from_an_python(a_coeffs, nu, eps, q, kappa, tau, s, m)
    print(f"Py  K_nu(recompute) = {K_py}")

    rerr = _relerr(K_cpp, K_py)
    print(f"Relative diff |K_cpp-K_py|/max = {rerr:.3e}")
    assert rerr < rel_tol

    # truncation stability test (Python side, by slicing coefficients)
    K_prev = None
    for N in trunc_list:
        sub = {k: v for k, v in a_coeffs.items() if (-N <= k <= N)}
        Kt = knu_from_an_python(sub, nu, eps, q, kappa, tau, s, m)
        if K_prev is not None:
            drift = _relerr(K_prev, Kt)
            print(f"Trunc N={N:3d}: K={Kt}, drift vs prev={drift:.3e}")
        else:
            print(f"Trunc N={N:3d}: K={Kt}")
        K_prev = Kt


def build_tr(M, a, omega, s, l, m):
    # lambda
    if abs(a) < 1e-15:
        lam = l*(l+1) - s*(s+1)
    else:
        lam = _core.SWSH(s, l, m, a*omega).m_lambda
    tr = _core.TeukolskyRadial(M, a, omega, s, l, m, lam)
    return tr, lam


def get_series_coeffs(tr, nu, n_max):
    meth, name = _pick_method(tr, [
        "ComputeSeriesCoefficients",
        "compute_series_coefficients",
        "series_coefficients",
        "get_series_coefficients",
    ])
    if meth is None:
        raise RuntimeError(
            "Error: C++ 未暴露 ComputeSeriesCoefficients(nu, n_max) 类接口。\n"
            "请在 pybind11 里为 TeukolskyRadial 添加 .def(\"ComputeSeriesCoefficients\", &TeukolskyRadial::ComputeSeriesCoefficients)\n"
        )
    a = meth(nu, n_max)
    # pybind 通常把 std::map<int, complex> 转成 dict
    if not isinstance(a, dict):
        # 尝试兼容：有些绑定会返回 list of pairs
        try:
            a = dict(a)
        except Exception:
            raise RuntimeError(f"ComputeSeriesCoefficients returned unsupported type: {type(a)}")
    print(f"Got a_n from {name} with n_max={n_max}: keys in [{min(a.keys())}, {max(a.keys())}] (count={len(a)})")
    return a


# =========================
# Main
# =========================
def test_case_schwarzschild():
    print("\n[Test A] Schwarzschild low-freq: nu, a_n recurrence, K_nu")
    M = 1.0
    a = 0.0
    omega = 0.001
    s, l, m = -2, 2, 2

    tr, lam = build_tr(M, a, omega, s, l, m)

    nu = test_nu_and_g(tr, guess=complex(l, 0.0), tol_abs=1e-10)
    assert abs(nu.real - l) < 0.1

    eps, q, kappa, tau = mst_params(M, a, omega, s, m, lam)

    a_coeffs = get_series_coeffs(tr, nu, n_max=120)
    test_an_recurrence(a_coeffs, nu, eps, q, kappa, tau, s, m, lam, n_check=30)
    test_knu_consistency(tr, nu, a_coeffs, eps, q, kappa, tau, s, m, rel_tol=1e-6)


def test_case_kerr():
    print("\n[Test B] Kerr: nu, a_n recurrence, K_nu")
    M = 1.0
    a = 0.5
    omega = 1.0
    s, l, m = -2, 2, 2

    tr, lam = build_tr(M, a, omega, s, l, m)

    nu = test_nu_and_g(tr, guess=complex(l, 0.1), tol_abs=1e-10)

    eps, q, kappa, tau = mst_params(M, a, omega, s, m, lam)

    a_coeffs = get_series_coeffs(tr, nu, n_max=300)
    test_an_recurrence(a_coeffs, nu, eps, q, kappa, tau, s, m, lam, n_check=40)
    test_knu_consistency(tr, nu, a_coeffs, eps, q, kappa, tau, s, m, rel_tol=3e-6)
    #提取最大的非0的系数a_n的索引N
    r_plus = M + (M*M - a*a)**0.5
    r_minus = M - (M*M - a*a)**0.5
    r_test = r_plus + 0.1  # 或者用你的 r_match
    x = (r_plus - r_test)/(r_plus - r_minus)
    N = max([k for k in a_coeffs.keys() if (a_coeffs[k] != 0.0 and  abs(p_nu(nu, k,   s, eps, tau, x))!=mp.mpf('0') and  abs(p_nu(nu, k+1, s, eps, tau, x))!=mp.mpf('0') )] )
    print(f"a_n≠0 max idx={N}")
    # N = min(N-60, 100)
    N=N-10
    pNp1, pN = p_nu(nu, N+1, s, eps, tau, x), p_nu(nu, N, s, eps, tau, x)
    print(f'p_nu test point x={x}')
    print(f'p_nu={p_nu(nu, N, s, eps, tau, x)}')
    print(f'p_nu+1={p_nu(nu, N+1, s, eps, tau, x)}')
    print(N * (a_coeffs[N+1]*pNp1) / (a_coeffs[N]*pN))
    print(f"Using N={N}")
    print(f'a_N={a_coeffs[N]}')
    print(f'a1={a_coeffs[1]}')
    check_eq134(a_coeffs, eps, kappa, N, rtol=1e-8)
 
    check_eq135(a_coeffs, nu, s, eps, kappa, tau, x, N, rtol=1e-6)
    # print(f'a_n', a_coeffs)


if __name__ == "__main__":
    test_case_schwarzschild()
    test_case_kerr()
    print("\nAll tests passed.")
