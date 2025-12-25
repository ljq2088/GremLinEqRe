#ifndef SWSH_LRR_H
#define SWSH_LRR_H

#include <complex>
#include <vector>

using Complex = std::complex<double>;

/**
 * @file SWSH_LRR.h
 * @brief Spin-weighted spheroidal harmonics solver aligned with LRR (lrr-2003-6).
 *
 * Conventions (LRR, Sec. 2):
 *   - Angular eigenfunction is the theta-dependent piece {}_sS_{lm}(theta).
 *   - lambda is the Teukolsky separation constant used in the radial equation:
 *       lambda = A_{slm}(c) + c^2 - 2 m c ,  with c = a*omega.
 *   - We additionally expose an internal E used in some codes/tests:
 *       E_internal = A_{slm}(c) + s(s+1).
 *
 * Normalization:
 *   Julia’s spectral eigenvectors normalize the full-sphere function
 *     ∫ |S(θ,φ)|^2 dΩ = 1.
 *   LRR often uses theta-only normalization
 *     ∫_0^π |S(θ)|^2 sinθ dθ = 1,
 *   with S(θ) being the θ-dependent piece (no e^{imφ}). If the full-sphere
 *   normalization holds, then S(θ) = sqrt(2π) * S(θ,φ=0).
 *   This implementation returns theta-only normalized values by multiplying
 *   the full-sphere basis sum at φ=0 with sqrt(2π).
 */
class SWSH {
public:
    SWSH(int s, int l, int m, double a_omega);

    // LRR-aligned outputs
    double get_E() const { return m_E_internal; }  // = A + s(s+1)
    double get_lambda() const { return m_lambda; } // = A + c^2 - 2 m c
    double get_A() const { return m_A; }           // angular separation constant
    double get_m() const { return m_m; }
    double get_l() const { return m_l; }
    double get_s() const { return m_s; }
    double get_aw() const { return m_c; }
    // Evaluate {}_sS_{lm}(theta) where input x = cos(theta).
    Complex evaluate_S(double x) const;

    // LRR Eq.(34): L_s^\dagger = d/dθ - m/sinθ + c sinθ + s cotθ.
    // Here: return L_2^\dagger ( {}_{-2}S_{lm} ).
    Complex evaluate_L2dag_S(double x) const;

    // Return L_1^\dagger L_2^\dagger ( {}_{-2}S_{lm} ).
    Complex evaluate_L1dag_L2dag_S(double x) const;

    // Convenience spherical harmonic at phi=0 (full-sphere normalized).
    static Complex spin_weighted_Y(int s, int l, int m, double x);

private:
    // Parameters
    int m_s;
    int m_l;
    int m_m;
    double m_c; // a*omega

    // Eigenvalues (LRR aligned)
    double m_A;          // angular separation constant A_{slm}(c) (Julia angular_sep_const)
    double m_lambda;     // LRR lambda = A + c^2 - 2 m c
    double m_E_internal; // = A + s(s+1)

    // Spectral basis range and coefficients (Julia spectral eigenvector)
    int m_lmin;
    int m_N;
    int m_lmax;
    std::vector<double> m_b; // real coefficients for real c

    struct Term {
        int p_ct2;    // power of cos(theta/2)
        int q_st2;    // power of sin(theta/2)
        double coeff; // precomputed term coefficient (real)
    };
    struct YlmCache {
        int l;
        Complex overall;      // includes normalization and phase (phi=0)
        std::vector<Term> terms; // r-sum terms
    };
    std::vector<YlmCache> m_Ycache;

    // Build spectral eigenpair (A and eigenvector)
    void solve_spectral();

    // Build cached direct-eval representation for spin-weighted spherical harmonic at phi=0.
    YlmCache build_Ylm_cache(int l) const;

    // Evaluate cached spherical harmonic and its theta derivatives up to 2nd order.
    // Returns (Y, dY/dθ, d2Y/dθ2) in double (real), without multiplying cache.overall.
    static void eval_Y_and_theta_derivs_from_cache(
        const YlmCache& cache,
        double ct2, double st2,
        const std::vector<double>& ct2_pows,
        const std::vector<double>& st2_pows,
        double& y, double& dy, double& d2y);

    // Evaluate spheroidal S and its θ-derivatives at given x=cosθ (theta-only normalized).
    void eval_spheroidal_theta_only(double x, double& S, double& S_th, double& S_thth) const;
};

#endif // SWSH_LRR_H
