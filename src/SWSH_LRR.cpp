#include "SWSH_LRR.h"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace {
constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr double kTwoPi = 2.0 * kPi;
constexpr double kTiny = 1e-300;

inline int lmin_from_s_m(int s, int m) {
    return std::max(std::abs(s), std::abs(m));
}

// Safe acos clamp.
inline double acos_clamped(double x) {
    if (x > 1.0) x = 1.0;
    if (x < -1.0) x = -1.0;
    return std::acos(x);
}

inline double log_factorial_int(int n) {
    // n >= 0
    return std::lgamma(static_cast<double>(n) + 1.0);
}

inline int pow_minus_one_int(int n) {
    return (n % 2 == 0) ? +1 : -1;
}

// ============================
// Julia spectral.jl coefficients
// ============================

inline double Fslm_jl(int s, int l, int m) {
    // Julia: (l == -1 && abs(m) == 0 && abs(s) == 0) ? 0 : ...
    if (l == -1 && std::abs(m) == 0 && std::abs(s) == 0) return 0.0;
    
    double ld = static_cast<double>(l);
    double sd = static_cast<double>(s);
    double md = static_cast<double>(m);
    
    double num1 = (ld + 1.0) * (ld + 1.0) - md * md;
    double den1 = (2.0 * ld + 3.0) * (2.0 * ld + 1.0);
    double term1 = std::sqrt(num1 / den1);
    
    double num2 = (ld + 1.0) * (ld + 1.0) - sd * sd;
    double den2 = (ld + 1.0) * (ld + 1.0);
    double term2 = std::sqrt(num2 / den2);
    
    return term1 * term2;
}

inline double Gslm_jl(int s, int l, int m) {
    if (l == 0) return 0.0;
    double ld = static_cast<double>(l);
    double sd = static_cast<double>(s);
    double md = static_cast<double>(m);
    
    double num1 = ld * ld - md * md;
    double den1 = 4.0 * ld * ld - 1.0;
    double term1 = std::sqrt(num1 / den1);
    
    double num2 = ld * ld - sd * sd;
    double den2 = ld * ld;
    double term2 = std::sqrt(num2 / den2);
    
    return term1 * term2;
}

inline double Hslm_jl(int s, int l, int m) {
    if (l == 0 || s == 0) return 0.0;
    return -static_cast<double>(m * s) / (static_cast<double>(l) * (static_cast<double>(l) + 1.0));
}

// Composite helpers strictly following spectral.jl
inline double Aslm_jl(int s, int l, int m) {
    return Fslm_jl(s, l, m) * Fslm_jl(s, l + 1, m);
}

inline double Bslm_jl(int s, int l, int m) {
    double h = Hslm_jl(s, l, m);
    return Fslm_jl(s, l, m) * Gslm_jl(s, l + 1, m) + 
           Gslm_jl(s, l, m) * Fslm_jl(s, l - 1, m) + 
           h * h;
}

inline double Cslm_jl(int s, int l, int m) {
    return Gslm_jl(s, l, m) * Gslm_jl(s, l - 1, m);
}

inline double Dslm_jl(int s, int l, int m) {
    return Fslm_jl(s, l, m) * (Hslm_jl(s, l + 1, m) + Hslm_jl(s, l, m));
}

inline double Eslm_jl(int s, int l, int m) {
    return Gslm_jl(s, l, m) * (Hslm_jl(s, l - 1, m) + Hslm_jl(s, l, m));
}

} // namespace

// ============================
// SWSH implementation
// ============================

SWSH::SWSH(int s, int l, int m, double a_omega)
    : m_s(s), m_l(l), m_m(m), m_c(a_omega),
      m_A(0.0), m_lambda(0.0), m_E_internal(0.0),
      m_lmin(lmin_from_s_m(s, m)), m_N(0), m_lmax(0) {
    solve_spectral();
}

void SWSH::solve_spectral() {
    // Match utils.determine_N in Julia:
    // N = (l - lmin + 1) + 10
    const int lmin = m_lmin;
    const int N = (m_l - lmin + 1) + 10;
    if (N <= 0) {
        throw std::runtime_error("SWSH::solve_spectral: invalid truncation N");
    }
    m_N = N;
    m_lmax = lmin + N - 1;

    // Build symmetric spectral matrix (real for real c).
    Eigen::MatrixXd M = Eigen::MatrixXd::Zero(N, N);

    for (int i = 0; i < N; ++i) {
        const int l = lmin + i;
        double ld = static_cast<double>(l);
        double sd = static_cast<double>(m_s);
        double c = m_c;

        // Diagonal (l' = l)
        // Julia: eigenvalue_Schwarzschild(s, l) - c^2 * Bslm + 2*c*s*Hslm
        double diag_schw = ld * (ld + 1.0) - sd * (sd + 1.0);
        double term_B = - c * c * Bslm_jl(m_s, l, m_m);
        double term_H = 2.0 * c * sd * Hslm_jl(m_s, l, m_m);
        M(i, i) = diag_schw + term_B + term_H;

        // Superdiagonal (l' = l+1) -> corresponds to Julia's lprime = l+1 case
        // Julia: -c^2 * Eslm(l+1) + 2*c*s*Gslm(l+1)
        if (i + 1 < N) {
            int l_next = l + 1; // This is l'
            double val = - c * c * Eslm_jl(m_s, l_next, m_m) 
                         + 2.0 * c * sd * Gslm_jl(m_s, l_next, m_m);
            
            M(i, i + 1) = val;
            M(i + 1, i) = val;
        }

        // Second superdiagonal (l' = l+2) -> corresponds to Julia's lprime = l+2 case
        // Julia: -c^2 * Cslm(l+2)
        if (i + 2 < N) {
            int l_next2 = l + 2; // This is l'
            double val = - c * c * Cslm_jl(m_s, l_next2, m_m);
            
            M(i, i + 2) = val;
            M(i + 2, i) = val;
        }
    }

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(M);
    if (solver.info() != Eigen::Success) {
        throw std::runtime_error("SWSH::solve_spectral: Eigen solver failed");
    }

    const int idx = m_l - lmin;
    if (idx < 0 || idx >= N) {
        throw std::runtime_error("SWSH::solve_spectral: requested l out of range");
    }

    const auto& evals = solver.eigenvalues();     // ascending
    const auto& evecs = solver.eigenvectors();    // columns

    // Julia: angular_sep_const = eigenvalues[index_of_l]
    m_A = evals(idx);

    // Coefficients b_l' come from eigenvector column idx.
    Eigen::VectorXd v = evecs.col(idx);

    // Julia: make coefficient at l itself positive, normalize.
    double ref = v(idx);
    if (std::abs(ref) < 1e-16) {
        // fallback: choose largest magnitude component
        int imax = 0;
        double vmax = 0.0;
        for (int i = 0; i < N; ++i) {
            if (std::abs(v(i)) > vmax) {
                vmax = std::abs(v(i));
                imax = i;
            }
        }
        if (vmax < 1e-30) {
            throw std::runtime_error("SWSH::solve_spectral: zero eigenvector");
        }
        if (v(imax) < 0.0) v = -v;
    } else {
        if (ref < 0.0) v = -v;
    }

    // v /= v[idx], then normalize
    ref = v(idx);
    if (std::abs(ref) < 1e-16) ref = 1.0;
    v /= ref;

    const double norm = std::sqrt(v.squaredNorm());
    if (!std::isfinite(norm) || norm <= 0.0) {
        throw std::runtime_error("SWSH::solve_spectral: invalid normalization");
    }
    v /= norm;

    m_b.resize(N);
    for (int i = 0; i < N; ++i) m_b[i] = v(i);

    // LRR-aligned constants
    m_lambda = m_A + m_c * m_c - 2.0 * static_cast<double>(m_m) * m_c;
    m_E_internal = m_A + static_cast<double>(m_s) * (static_cast<double>(m_s) + 1.0);

    // Precompute spherical harmonic caches for lmin..lmax
    m_Ycache.clear();
    m_Ycache.reserve(N);
    for (int i = 0; i < N; ++i) {
        m_Ycache.push_back(build_Ylm_cache(lmin + i));
    }
}

SWSH::YlmCache SWSH::build_Ylm_cache(int l) const {
    // Direct-eval cache mirroring harmonic.jl (direct method), at phi=0.
    // It builds:
    //   {}_sY_{lm}(theta,0) = overall * sum_r coeff_r * ct2^{p_r} st2^{q_r}
    // with ct2 = cos(theta/2), st2 = sin(theta/2).

    if (l < m_lmin) {
        throw std::runtime_error("build_Ylm_cache: l below lmin");
    }

    const int s = m_s;
    const int m = m_m;

    // Prefactor: (-1)^m * e^{imφ} * [if s<0 then (-1)^s] * sqrt((l-m)!(l+m)!/(l-s)!(l+s)!)
    const int phase_m = pow_minus_one_int(m);
    const int phase_s = (s < 0) ? pow_minus_one_int(s) : +1;

    const double log_ratio =
        0.5 * (log_factorial_int(l - m) + log_factorial_int(l + m)
             - log_factorial_int(l - s) - log_factorial_int(l + s));
    const double ratio = std::exp(log_ratio);
    const double pref = static_cast<double>(phase_m * phase_s) * ratio;

    // log_norm in Julia:
    const double log_norm =
        0.5 * std::log((2.0 * l + 1.0) / (4.0 * kPi))
        + 0.5 * (log_factorial_int(l - s) + log_factorial_int(l + s)
               + log_factorial_int(l - m) + log_factorial_int(l + m));

    // r-range
    const int r_min = std::max(0, m - s);
    const int r_max = std::min(l - s, l + m);

    // compute max log_pref for stability
    double max_log_pref = -std::numeric_limits<double>::infinity();
    struct RawTerm { int r; double log_pref; int sign; int p; int q; };
    std::vector<RawTerm> raw;
    raw.reserve(std::max(0, r_max - r_min + 1));

    for (int r = r_min; r <= r_max; ++r) {
        const int a1 = r;
        const int a2 = l + m - r;
        const int a3 = l - s - r;
        const int a4 = r + s - m;
        if (a1 < 0 || a2 < 0 || a3 < 0 || a4 < 0) continue;

        const double lp = - (log_factorial_int(a1) + log_factorial_int(a2)
                           + log_factorial_int(a3) + log_factorial_int(a4));

        max_log_pref = std::max(max_log_pref, lp);

        const int sign = pow_minus_one_int(l - r - s); // (-1)^(l-r-s)
        const int p = 2 * r + s - m;
        const int q = 2 * l - 2 * r - s + m;

        raw.push_back({r, lp, sign, p, q});
    }

    if (raw.empty()) {
        throw std::runtime_error("build_Ylm_cache: empty r-sum");
    }

    const double overall_scale = std::exp(log_norm + max_log_pref) * pref;

    YlmCache cache;
    cache.l = l;
    cache.overall = Complex(overall_scale, 0.0);
    cache.terms.reserve(raw.size());

    for (const auto& t : raw) {
        const double cterm = static_cast<double>(t.sign) * std::exp(t.log_pref - max_log_pref);
        cache.terms.push_back({t.p, t.q, cterm});
    }

    return cache;
}

void SWSH::eval_Y_and_theta_derivs_from_cache(
    const YlmCache& cache,
    double ct2, double st2,
    const std::vector<double>& ct2_pows,
    const std::vector<double>& st2_pows,
    double& y, double& dy, double& d2y) {

    // Evaluate sum_r coeff * ct2^p st2^q and its θ-derivatives up to 2nd.
    // For each term f = ct2^p st2^q:
    //   u = ln f = p ln ct2 + q ln st2
    //   u'  = p ct2'/ct2 + q st2'/st2  with ct2'=-0.5 st2, st2'=0.5 ct2
    //   u'' = -0.25*(p sec^2(θ/2) + q csc^2(θ/2))
    //   f'  = f u',  f'' = f (u'' + u'^2)

    const double inv_ct2 = 1.0 / std::max(std::abs(ct2), std::sqrt(kTiny)) * (ct2 >= 0 ? 1.0 : -1.0);
    const double inv_st2 = 1.0 / std::max(std::abs(st2), std::sqrt(kTiny)) * (st2 >= 0 ? 1.0 : -1.0);

    const double sec2 = inv_ct2 * inv_ct2;
    const double csc2 = inv_st2 * inv_st2;

    y = 0.0;
    dy = 0.0;
    d2y = 0.0;

    for (const auto& t : cache.terms) {
        const int p = t.p_ct2;
        const int q = t.q_st2;
        if (p < 0 || q < 0) continue;
        if (p >= static_cast<int>(ct2_pows.size())) continue;
        if (q >= static_cast<int>(st2_pows.size())) continue;

        const double f = ct2_pows[p] * st2_pows[q];
        const double u1 = (-0.5 * static_cast<double>(p)) * (st2 * inv_ct2)
                        + (0.5 * static_cast<double>(q)) * (ct2 * inv_st2);
        const double u2 = -0.25 * (static_cast<double>(p) * sec2 + static_cast<double>(q) * csc2);
        const double f1 = f * u1;
        const double f2 = f * (u2 + u1 * u1);

        const double c = t.coeff;
        y   += c * f;
        dy  += c * f1;
        d2y += c * f2;
    }
}

void SWSH::eval_spheroidal_theta_only(double x, double& S, double& S_th, double& S_thth) const {
    const double theta = acos_clamped(x);
    const double ct2 = std::cos(0.5 * theta);
    const double st2 = std::sin(0.5 * theta);

    // Determine max power needed for pows.
    const int max_pow = 2 * m_lmax + std::abs(m_s) + std::abs(m_m) + 4;

    std::vector<double> ct2_pows(max_pow + 1);
    std::vector<double> st2_pows(max_pow + 1);
    ct2_pows[0] = 1.0;
    st2_pows[0] = 1.0;
    for (int i = 1; i <= max_pow; ++i) {
        ct2_pows[i] = ct2_pows[i - 1] * ct2;
        st2_pows[i] = st2_pows[i - 1] * st2;
    }

    // Accumulate full-sphere normalized S(θ,φ=0) and its θ-derivatives.
    double s0 = 0.0, s1 = 0.0, s2 = 0.0;

    for (int i = 0; i < m_N; ++i) {
        const double bi = m_b[i];
        if (std::abs(bi) < 1e-18) continue;

        const auto& cache = m_Ycache[i];
        double y, dy, d2y;
        eval_Y_and_theta_derivs_from_cache(cache, ct2, st2, ct2_pows, st2_pows, y, dy, d2y);

        // multiply by cache.overall (real)
        const double overall = cache.overall.real();
        s0 += bi * overall * y;
        s1 += bi * overall * dy;
        s2 += bi * overall * d2y;
    }

    // Convert to theta-only normalization (LRR): S(θ) = sqrt(2π) * S(θ,φ=0).
    const double theta_norm = std::sqrt(kTwoPi);
    S      = theta_norm * s0;
    S_th   = theta_norm * s1;
    S_thth = theta_norm * s2;
}

Complex SWSH::evaluate_S(double x) const {
    double S, S_th, S_thth;
    eval_spheroidal_theta_only(x, S, S_th, S_thth);
    return Complex(S, 0.0);
}

Complex SWSH::evaluate_L2dag_S(double x) const {
    // LRR Eq.(34): L_s^\dagger = d/dθ - m/sinθ + c sinθ + s cotθ.
    // Here we need L_2^\dagger acting on {}_{-2}S.
    double S, S_th, S_thth;
    eval_spheroidal_theta_only(x, S, S_th, S_thth);

    const double theta = acos_clamped(x);
    const double st = std::sin(theta);
    const double ct = std::cos(theta);
    const double inv_st = 1.0 / std::max(std::abs(st), 1e-14) * (st >= 0 ? 1.0 : -1.0);
    const double cot = ct * inv_st;

    const double P2 = -static_cast<double>(m_m) * inv_st + m_c * st + 2.0 * cot;

    const double out = S_th + P2 * S;
    return Complex(out, 0.0);
}

Complex SWSH::evaluate_L1dag_L2dag_S(double x) const {
    // L_1^\dagger L_2^\dagger S = S'' + (P1+P2)S' + (P2' + P1 P2)S
    double S, S_th, S_thth;
    eval_spheroidal_theta_only(x, S, S_th, S_thth);

    const double theta = acos_clamped(x);
    const double st = std::sin(theta);
    const double ct = std::cos(theta);
    const double inv_st = 1.0 / std::max(std::abs(st), 1e-14) * (st >= 0 ? 1.0 : -1.0);
    const double csc2 = inv_st * inv_st;
    const double cot = ct * inv_st;

    const double P2 = -static_cast<double>(m_m) * inv_st + m_c * st + 2.0 * cot;
    const double P1 = -static_cast<double>(m_m) * inv_st + m_c * st + 1.0 * cot;

    const double P2p = static_cast<double>(m_m) * inv_st * cot + m_c * ct - 2.0 * csc2;

    const double out = S_thth + (P1 + P2) * S_th + (P2p + P1 * P2) * S;
    return Complex(out, 0.0);
}

Complex SWSH::spin_weighted_Y(int s, int l, int m, double x) {
    // Convenience: full-sphere normalized {}_sY_{lm}(theta,phi=0).
    // This is mainly for debugging/testing.
    const double theta = acos_clamped(x);
    const double ct2 = std::cos(0.5 * theta);
    const double st2 = std::sin(0.5 * theta);

    // Use a temporary SWSH to reuse the exact cache builder (costly but rarely used).
    SWSH tmp(s, l, m, 0.0);

    const int lmin = lmin_from_s_m(s, m);
    const int idx = l - lmin;
    if (idx < 0 || idx >= static_cast<int>(tmp.m_Ycache.size())) {
        throw std::runtime_error("spin_weighted_Y: cache index out of range");
    }
    const auto& cache = tmp.m_Ycache[idx];

    int max_pow = 0;
    for (const auto& t : cache.terms) {
        max_pow = std::max(max_pow, std::max(t.p_ct2, t.q_st2));
    }
    std::vector<double> ct2_pows(max_pow + 1), st2_pows(max_pow + 1);
    ct2_pows[0] = 1.0;
    st2_pows[0] = 1.0;
    for (int i = 1; i <= max_pow; ++i) {
        ct2_pows[i] = ct2_pows[i - 1] * ct2;
        st2_pows[i] = st2_pows[i - 1] * st2;
    }

    double y, dy, d2y;
    eval_Y_and_theta_derivs_from_cache(cache, ct2, st2, ct2_pows, st2_pows, y, dy, d2y);
    return cache.overall * Complex(y, 0.0);
}
