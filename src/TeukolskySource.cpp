#include "TeukolskySource.h"
#include "SWSH_LRR.h"  
#include <cmath>
#include <complex>
using Complex = std::complex<double>;
using namespace std::complex_literals; // 允许使用 1.0i
TeukolskySource::TeukolskySource(Real a_spin, Real omega, int s, int l, int m) : m_a(a_spin), m_omega(omega),m_s(s), m_l(l), m_m(m) {
    double a_omega=m_a * m_omega;
}
inline Complex get_Vs(int s, int m, double a_omega, double sin_th, double cos_th) {
    if (std::abs(sin_th) < 1e-10) sin_th = 1e-10; // 极轴保护
    double cot_th = cos_th / sin_th;
    return -(double)m / sin_th + a_omega * sin_th + (double)s * cot_th;
}
SourceProjections TeukolskySource::ComputeProjections(
    const KerrGeo::State& st, 
    const KerrGeo& geo_obj,
    SWSH& swsh) 
{
    SourceProjections proj;

    // ==========================================================
    // 1. 提取物理量与坐标
    // ==========================================================
    // 坐标: Boyer-Lindquist (t, r, theta, phi)
    double r = st.x[1];
    double theta = st.x[2];
    
    // 四速度: u^t, u^r, u^theta, u^phi (注意：这是对固有时的导数 d/d_tau)
    double ut = st.u[0];   // dt/d_tau
    double ur = st.u[1];   // dr/d_tau
    double uth = st.u[2];  // d_theta/d_tau
    // double uphi = st.u[3]; // unused directly in C coeffs if Lz is used

    // 守恒量 (Particle Constants)
    double E_part = geo_obj.energy();
    double Lz_part = geo_obj.angular_momentum();
    // Carter constant Q 用于校验，计算中主要使用 E 和 Lz

    // 参数
    double M = 1.0;
    double a = m_a;
    double omega = m_omega;
    int m = m_m;
    double a_omega = a * omega;

    // ==========================================================
    // 2. 计算几何辅助量
    // ==========================================================
    double sin_th = std::sin(theta);
    double cos_th = std::cos(theta);
    double sin2 = sin_th * sin_th;
    double Sigma = r*r + a*a * cos_th*cos_th;
    double Delta = r*r - 2.0*M*r + a*a;
    
    Complex i = 1.0i;

    // 变量定义: rho (LRR Eq. 6 definition: rho = (r - i a cos)^-1)
    // 注意：很多文献(Teukolsky 73)定义 rho = -1/(r-ia cos)。
    // LRR 2003-6 Eq 6 明确写的是 (r - i a cos)^-1。请务必确认这里的一致性。
    // 如果 swsh 的算符是基于标准定义的，这里符号可能会影响。
    // 暂时遵循 LRR 论文原文定义。
    Complex rho = 1.0 / (r - i * a * cos_th);
    Complex rho_bar = 1.0 / (r + i * a * cos_th);   
    Complex rho2 = rho * rho;
    Complex rho3 = rho2 * rho;
    Complex rho_bar2 = rho_bar * rho_bar;
    
    // 场变量 K (LRR Eq. 16)
    // K = (r^2 + a^2) * omega - m * a
    double K_val = (r*r + a*a) * omega - m * a;
    Complex K_div_Delta = K_val / Delta;
    // ==========================================================
    // 3. 计算 SWSH 函数值及其导数算符
    // ==========================================================

    double x_cos = cos_th;
    

    Complex S = swsh.evaluate_S(x_cos);
    
    
    Complex L_2_dag_S = swsh.evaluate_L2dag_S(x_cos);
    

    Complex L1L2S_full = swsh.evaluate_L1dag_L2dag_S(x_cos);

    // ==========================================================
    // 辅助导数项提取 (用于后续 A_nn0 等系数的 Chain Rule)
    // ==========================================================
    
    // 我们需要 dS/dtheta。
    // 利用关系: L_{2}^dag S = dS/dtheta + V_{2} S
    // 其中 V_{2} = -m/sin - (2)cot + a*omega*sin
    // 因此: dS/dtheta = L_{2}^dag S - V_{2} * S
    
    // 注意：get_Vs 函数定义在 TeukolskySource.cpp 顶部，确保它包含 aw*sin
    // inline Complex get_Vs(...) { return -m/sin + aw*sin + s*cot; }
    Complex V_2 = get_Vs(2, m, a_omega, sin_th, cos_th);
    Complex dS_dtheta = L_2_dag_S - V_2 * S;

    // 我们需要 d(L_{2}^dag S)/dtheta (记为 d(L2S))。
    // 利用关系: L_{1}^dag (L2S) = d(L2S)/dtheta + V_{1} * (L2S)
    // 因此: d(L2S)/dtheta = L1L2S_full - V_{1} * L_minus2_dag_S
    
    Complex V_1 = get_Vs(1, m, a_omega, sin_th, cos_th);
    Complex d_L_2_dag_S_dtheta = L1L2S_full - V_1 * L_2_dag_S;
    // ==========================================================
    // 4. 计算 C 系数 (Tetrad projections of T_munu)
    // 依据 LRR Eq. (30), (31), (32)
    // ==========================================================
    
    // 辅助项
    // Term 1: E(r^2 + a^2) - a Lz + Sigma * dr/dtau
    double term_rad = (E_part * (r*r + a*a) - a * Lz_part + Sigma * ur)/(2.0*Sigma);
    
    // Term 2: i sin(theta) [ a E - Lz/sin^2(theta) ] + Sigma * dtheta/dtau
    Complex term_ang =-rho* i * sin_th * (a * E_part - Lz_part / sin2) + Sigma * uth/(std::sqrt(2.0));

    // [Eq. 30] C_nn
    // Coeff = 1 / (4 * Sigma^3 * ut)
    Complex C_nn = (term_rad * term_rad)/(Sigma*ut);

    // [Eq. 31] C_mbarn
    // Coeff = -rho / (2 * sqrt(2) * Sigma^2 * ut)
    // Note: LRR Eq 31 has a minus sign.
    Complex C_mbarn = term_rad * term_ang/(Sigma*ut);

    // [Eq. 32] C_mbarmbar
    // Coeff = rho^2 / (2 * Sigma * ut)
    Complex C_mbarmbar =  (term_ang * term_ang)/(Sigma*ut);

    // ==========================================================
    // 5. 计算 A 系数 (Source terms coefficients)
    // 依据 LRR Eq. (37) - (42)
    // ==========================================================
    
    // 预计算常用的因子
    Complex factor_sqrt_pi = 1.0 / std::sqrt(M_PI);
    Complex factor_sqrt_2pi = 1.0 / std::sqrt(2.0 * M_PI);
    Complex factor_2_sqrt_pi_Delta = 2.0 / (std::sqrt(M_PI) * Delta);
    
    // ----------------------------------------------------------
    // A_nn0 (Eq. 37)
    // A_nn0 = -2/(sqrt(2pi)*Delta^2) * rho^-2 * rhobar^-1 * C_nn * L_1^dag [ rho^-4 L_2^dag (rho^3 S) ]
    // ----------------------------------------------------------
    
    // --- 准备 Op_nn0 计算所需的公共项 ---
    // 我们需要计算 L_1^dag [ rho^-4 * L_2^dag (rho^3 S) ]
    // 其中 S 的 spin weight 为 -2
    
    // 1. 计算 Inner = L_2^dag (rho^3 S)
    // L_2^dag = partial_theta + V_2
 
    // Complex V_2 = get_Vs(2, m, a_omega, sin_th, cos_th);
    
    // d(rho)/dtheta = -rho^2 * (-i a (-sin)) = -i a rho^2 sin
    Complex d_rho_dth = -i * a * rho2 * sin_th;
    Complex d_rho3_dth = 3.0 * rho2 * d_rho_dth; // = -3 i a rho^4 sin
    
    // Inner = d(rho^3 S) + V_2 * rho^3 S
    //       = (d_rho3) S + rho^3 (dS) + V_2 rho^3 S
    //       = rho^3 [ dS + (V_2 - 3 i a rho sin) S ]
    Complex term_inner_bracket = dS_dtheta + (V_2 - 3.0 * i * a * rho * sin_th) * S;
    Complex Inner = rho3 * term_inner_bracket; // L_2^dag (rho^3 S)
    
    // 为了下一步求导，我们需要 d(Inner)/dtheta
    // d(Inner) = d(rho^3) * bracket + rho^3 * d(bracket)
    // d(bracket) = d(dS) + d(V_2)*S + V_2*dS - d(3ia rho sin S)
    
    // d(dS) = d(L_{2}^dag S - V_{2} S)
    //       = d(L_{2}^dag S) - d(V_{2}) S - V_{2} dS
    Complex d_V_2_dth = m * cos_th / sin2 + a_omega * cos_th - (2.0) / sin2; // d/dth(-2 cot) = +2/sin^2
    Complex d2S_dtheta2 = d_L_2_dag_S_dtheta - d_V_2_dth * S - V_2 * dS_dtheta;
    
    // d(V_2)/dtheta
    // Complex d_V2_dth = m * cos_th / sin2 + a_omega * cos_th - (2.0) / sin2;
    
    // d(rho sin) = d_rho sin + rho cos = (-i a rho^2 sin) sin + rho cos
    Complex d_rho_sin_dth = -i * a * rho2 * sin2 + rho * cos_th;
    
    Complex d_bracket_dth = d2S_dtheta2 
                          + d_V_2_dth * S + V_2 * dS_dtheta 
                          - 3.0 * i * a * (d_rho_sin_dth * S + rho * sin_th * dS_dtheta);
                          
    Complex d_Inner_dth = d_rho3_dth * term_inner_bracket + rho3 * d_bracket_dth;

    // 2. 计算 Outer = L_1^dag [ rho^-4 * Inner ]
    // Outer = d(rho^-4 Inner) + V_1 * (rho^-4 Inner)
    //       = d(rho^-4) Inner + rho^-4 d(Inner) + V_1 rho^-4 Inner
    // d(rho^-4) = -4 rho^-5 d_rho = -4 rho^-5 (-i a rho^2 sin) = 4 i a rho^-3 sin
    // Complex V_1 = get_Vs(1, m, a_omega, sin_th, cos_th);
    
    Complex Op_nn0_val = 4.0 * i * a * std::pow(rho, -3) * sin_th * Inner
                       + std::pow(rho, -4) * d_Inner_dth
                       + V_1 * std::pow(rho, -4) * Inner;
    proj.A_nn0 = (-2.0 *factor_sqrt_2pi/( Delta * Delta)) 
                 * std::pow(rho, -2) * std::pow(rho_bar, -1) * C_nn 
                 * Op_nn0_val;

    // ----------------------------------------------------------
    // A_mbarn0 (Eq. 38)
    // A_mbarn0 = 2/(sqrt(pi)*Delta) * rho^-3 * C_mbarn * [ (L_2^dag S)(iK/Delta + rho + rhobar) - a sin S (K/Delta)(rhobar - rho) ]
    // ----------------------------------------------------------
    
    // --- A_mbarn0 (Eq. 38) ---
    // Term = (L_2^dag S)(iK/Delta + rho + rhobar) - a sin S (K/Delta)(rhobar - rho)
    // L_2^dag S = dS + V_2 S
    Complex L2_dag_S = dS_dtheta + V_2 * S;
    Complex term_mbarn0 = L2_dag_S * (i * K_div_Delta + rho + rho_bar)
                        - a * sin_th * S * K_div_Delta * (rho_bar - rho);
    
    proj.A_mbarn0 = factor_2_sqrt_pi_Delta * std::pow(rho, -3) * C_mbarn * term_mbarn0;

    // --- A_mbarn1 (Eq. 40) ---
    // Term = L_2^dag S + i a sin(rhobar - rho) S
    Complex term_mbarn1 = L2_dag_S + i * a * sin_th * (rho_bar - rho) * S;
    
    proj.A_mbarn1 = factor_2_sqrt_pi_Delta * std::pow(rho, -3) * C_mbarn * term_mbarn1;

    // --- A_mbarmbar0 (Eq. 39) ---
    // Term = -i (K/Delta),r - (K/Delta)^2 + 2i rho (K/Delta)
    // (K/Delta),r = (K' Delta - K Delta') / Delta^2
    // K' = 2rw, Delta' = 2(r-M)
    double dK_dr = 2.0 * r * omega;
    double dDelta_dr = 2.0 * (r - M);
    Complex d_K_div_Delta_dr = (dK_dr * Delta - K_val * dDelta_dr) / (Delta * Delta);
    
    Complex term_mm0 = -i * d_K_div_Delta_dr - K_div_Delta * K_div_Delta + 2.0 * i * rho * K_div_Delta;
    
    proj.A_mbarmbar0 = -factor_sqrt_2pi * std::pow(rho, -3) * rho_bar * C_mbarmbar * S * term_mm0;

    // --- A_mbarmbar1 (Eq. 41) ---
    // Term = i K/Delta + rho
    Complex term_mm1 = i * K_div_Delta + rho;
    
    proj.A_mbarmbar1 = (-2.0 * factor_sqrt_2pi) * std::pow(rho, -3) * rho_bar * C_mbarmbar * S * term_mm1;

    // --- A_mbarmbar2 (Eq. 42) ---
    proj.A_mbarmbar2 = -factor_sqrt_2pi * std::pow(rho, -3) * rho_bar * C_mbarmbar * S;

    return proj;
}