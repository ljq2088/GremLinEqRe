import sys, os, math, cmath
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from GremLinEqRe import _core


def teuk_residual(M, a, omega, s, m, lamb, r, R, Rp, Rpp):
    """Residual of: Δ R'' + (s+1)Δ' R' + H R = 0"""
    Delta = r*r - 2.0*M*r + a*a
    dDelta = 2.0*(r - M)
    K = (r*r + a*a)*omega - m*a
    H = (K*K - 2j*s*(r - M)*K)/Delta + 4j*s*omega*r - lamb
    return Delta*Rpp + (s+1)*dDelta*Rp + H*R


def central_second_derivative(evalR, r, h):
    Rm, _  = evalR(r - h)
    R0, Rp = evalR(r)
    Rp1, _ = evalR(r + h)
    Rpp = (Rp1 - 2*R0 + Rm) / (h*h)
    return R0, Rp, Rpp


def main():
    # -------- case config (你可以换成你真正出问题的参数) --------
    M = 1.0
    a = 0.9
    omega = 0.3
    s = -2
    l = 2
    m = 2

    # Schwarzschild(aω=0) 时常用：lambda = l(l+1)-s(s+1)
    # Kerr 时你应传入 spheroidal eigenvalue 相关的那一套 lambda（与你 C++ 约定一致）
    lamb = _core.SWSH(s,l,m,a*omega).m_lambda

    n_max = 300
    kappa = math.sqrt(1.0 - a*a)
    r_plus=1.0 + kappa
    r_match= r_plus+2.0*kappa

    tr = _core.TeukolskyRadial(M, a, omega, s, l, m, lamb)

    # ---- ensure bindings exist ----
    need = ["Evaluate_R_in", "Evaluate_Hypergeometric", "Compute_Raw_Coulomb_Combo", "ComputeSeriesCoefficients", "k_factor", "solve_nu"]
    missing = [name for name in need if not hasattr(tr, name)]
    if missing:
        print("[ERROR] Missing bindings:", missing)
        print("请在 pybind11 里把这些方法 .def 暴露出来，然后重编译。")
        sys.exit(1)
    print("s,l,m,a,omega =", s, l, m, a, omega)
    # ---- solve nu ----
    guess = complex(l, 0.1)   # 你的经验初始化
    nu = tr.solve_nu(guess)
    nu_neg = -nu - 1.0
    g_val=tr.calc_g(nu)

    print("\n[nu]")
    print("  guess =", guess)
    print("  nu    =", nu)
    print("  nu_neg=", nu_neg)
    print("  g(nu) =", g_val)

    # ---- a_n, K ----
    a_pos = tr.ComputeSeriesCoefficients(nu, n_max)
    a_neg = tr.ComputeSeriesCoefficients(nu_neg, n_max)

    K_nu  = tr.k_factor(nu)
    K_neg = tr.k_factor(nu_neg)

    print("\n[a_n]")
    print("  len(a_pos) =", len(a_pos), " n in [", min(a_pos.keys()), ",", max(a_pos.keys()), "]")
    print("  len(a_neg) =", len(a_neg), " n in [", min(a_neg.keys()), ",", max(a_neg.keys()), "]")

    print("\n[K]")
    print("  K_nu  =", K_nu)
    print("  K_neg =", K_neg)

    # ---- eval wrappers ----
    def eval_near(r):
        return tr.Evaluate_Hypergeometric(r, nu, a_pos)

    def eval_far(r):
        return tr.Compute_Raw_Coulomb_Combo(r, nu, K_nu, K_neg, a_pos, a_neg)

    def eval_Rin(r):
        return tr.Evaluate_R_in(r, nu, K_nu, K_neg, a_pos, a_neg, r_match)

    # ---- check near/far ratio around match ----
    print("\n[match check near vs far]")
    r_list=r_match+np.linspace(-1.0, 20.0, 20)*kappa
    print(f'r_plus={r_plus: .6f} r_match={r_match: .6f} kappa={kappa: .6f}')
    for rr in r_list:
        Rn, dRn = eval_near(rr)
        Rf, dRf = eval_far(rr)
        ratio = Rn / Rf if abs(Rf) > 0 else complex('nan')

        # how well derivative matches after scaling by ratio
        rel = abs( dRn/Rn - dRf/Rf )


        print(f"  r={rr: .6f} Rn={Rn: .16e} Rf={Rf: .16e} ratio= {ratio: .16e}  rel(dR)= {rel: .3e}")

    # ---- ODE residual scan ----
    print("\n[ODE residual scan using finite-diff R'']")
    r_plus = M + math.sqrt(M*M - a*a)
    rs = r_list

    for rr in rs:
        h = 1e-5 * max(1.0, rr)  # step scales with r
        R0, Rp0, Rpp = central_second_derivative(eval_Rin, rr, h)
        res = teuk_residual(M, a, omega, s, m, lamb, rr, R0, Rp0, Rpp)

        denom = abs(Rpp)*(rr*rr) + abs(Rp0)*rr + abs(R0) + 1e-300
        rel = abs(res) / denom
        print(f"  r={rr: .6f}  |res|={abs(res): .3e}  rel~{rel: .3e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
