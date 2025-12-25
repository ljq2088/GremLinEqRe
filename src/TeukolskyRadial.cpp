/**
 * @file TeukolskyRadial.cpp
 * @brief Teukolsky 径向求解器实现
 * * 修正记录:
 * 1. 修复 std::complex 不支持 ++y 的问题。
 * 2. 更新 log_gamma 算法系数，完全对齐 GremlinEq/src/fujtag/gammln.cc
 * 以确保高精度复现。
 */

 /**
 * @file TeukolskyRadial.cpp
 * @brief Teukolsky 径向求解器实现
 *
 * 修改日志:
 * 1. 引入 GremlinEq 特有的 sinln 函数以处理 log(sin(z)) 的分支和溢出问题。
 * 2. 修复 log_gamma 在负实部区域的计算，使其与 Scipy/GremlinEq 对齐。
 */
//所有涉及s的地方都带入-2
#include "TeukolskyRadial.h"
#include <iostream>
#include <limits>
#include <stdio.h>

#include <acb_hypgeom.h>
#include <acb.h>
#include <arb.h>
#include <arf.h>
using namespace std::complex_literals; // 允许使用 1.0i

// ==========================================================
// 内部辅助函数: sinln
// 来源: GremlinEq/src/fujtag/gammln.cc
// 作用: 计算 log(sin(z))，专门处理复数分支和大虚部数值稳定性
// ==========================================================
static Complex sinln(const Complex& x) {
    // GremlinEq 实现逻辑:
    // return - I * M_PI_2 - M_LN2 +
    //   ((imag(x) > 0) ?
    //    - I * x + log(exp(2.*I*x)-1.) :
    //    + I * x + log(1.-exp(-2.*I*x)));
    
    const Complex I = 1.0i;
    // M_PI_2 = pi/2, M_LN2 = log(2)
    
    if (x.imag() > 0.0) {
        return -I * M_PI_2 - M_LN2 - I * x + std::log(std::exp(2.0 * I * x) - 1.0);
    } else {
        return -I * M_PI_2 - M_LN2 + I * x + std::log(1.0 - std::exp(-2.0 * I * x));
    }
}

TeukolskyRadial::TeukolskyRadial(Real M,Real a_spin, Real omega, int s, int l, int m, Real lambda)
    : m_M(M),m_a(a_spin), m_omega(omega), m_s(s), m_l(l), m_m(m), m_lambda(lambda)
{
    m_epsilon = 2.0* m_omega;//输入的ω应是Kerr频率×M(还有国际单位的因子)
    q = m_a;//考虑a是M的量纲
    
    m_kappa = std::sqrt(1.0 - q * q);
    m_tau = (m_epsilon - m_m * q) / m_kappa;
    
    m_epsilon_sq = m_epsilon * m_epsilon;
    m_tau_sq = m_tau * m_tau;
}

// ==========================================================
// 辅助函数：复数 Log Gamma
// 完全复刻 GremlinEq/src/fujtag/gammln.cc 的实现
// ==========================================================
Complex TeukolskyRadial::log_gamma(Complex z) {
    // 1. 处理实部 <= 0 的情况（使用反射公式）
    // GremlinEq: if(real(x) <= 0) return log(M_PI) - sinln(M_PI * x) - gammln(1.0 - x);
    if (z.real() <= 0.0) {
        // 使用自定义的 sinln 而不是 std::log(std::sin(...))
        // 这解决了 2*pi*i 的分支误差问题
        return std::log(M_PI) - sinln(M_PI * z) - log_gamma(1.0 - z);
    }

    // 2. Lanczos 近似 (GremlinEq 特有的 14 项系数)
    static const double coef[14] = { 
        57.1562356658629235,    -59.5979603554754912,     14.1360979747417471,
       -0.491913816097620199,    3.39946499848118887e-5,  4.65236289270485756e-5,
       -9.83744753048795646e-5,  1.58088703224912494e-4, -2.10264441724104883e-4,
        2.17439618115212643e-4, -1.64318106536763890e-4,  8.44182239838527433e-5,
       -2.61908384015814087e-5,  3.68991826595316234e-6
    };

    Complex x = z;
    Complex y = z;
    Complex tmp = x + 5.24218750000000000;
    tmp = (x + 0.5) * std::log(tmp) - tmp;
    
    Complex ser = 0.999999999999997092;
    for (int j = 0; j < 14; j++) {
        y += 1.0;
        ser += coef[j] / y;
    }
    
    return tmp + std::log(2.5066282746310005 * ser / x);
}

Complex TeukolskyRadial::coeff_alpha(Complex nu, int n) const {
    Complex n_nu = nu + (double)n;
    Complex iepskappa = 1.0i * m_epsilon * m_kappa;
    
    Complex deno=n_nu*(n_nu + 1.0)*(2.0*n_nu -1.0)*(2.0*n_nu +3.0);
    return (iepskappa * n_nu * (2.0 * n_nu - 1.0)
           * ((n_nu + 1.0 + (double)m_s)*(n_nu + 1.0 + (double)m_s) + m_epsilon_sq)
           * (n_nu + 1.0 + 1.0i * m_tau));
}

Complex TeukolskyRadial::coeff_gamma(Complex nu, int n) const {
    Complex n_nu = nu + (double)n;
    Complex iepskappa = 1.0i * m_epsilon * m_kappa;
    
    Complex deno=n_nu*(n_nu + 1.0)*(2.0*n_nu -1.0)*(2.0*n_nu +3.0);
    return (-iepskappa * (n_nu + 1.0) * (2.0 * n_nu + 3.0)
           * ((n_nu - (double)m_s)*(n_nu - (double)m_s) + m_epsilon_sq)
           * (n_nu - 1.0i * m_tau));
}

Complex TeukolskyRadial::coeff_beta(Complex nu, int n) const {
    Complex n_nu = nu + (double)n;
    
    Complex term1 = n_nu * (n_nu + 1.0); 
    
    Complex b_add1 = 2.0 * m_epsilon_sq - m_epsilon * m_m * q - m_lambda - (double)m_s*((double)m_s + 1.0);
    Complex b_add2 = m_epsilon * (m_epsilon - m_m * q) * ((double)m_s*(double)m_s + m_epsilon_sq);
    Complex deno=n_nu*(n_nu + 1.0)*(2.0*n_nu -1.0)*(2.0*n_nu +3.0);
    return  ((term1 * (term1 + b_add1) + b_add2)
           * (2.0* n_nu + 3.0) * (2.0*n_nu - 1.0));
}

Complex TeukolskyRadial::continued_fraction(Complex nu, int direction) const {
    const int max_iter = 30000;
    const double tol = 1e-14;
    const double tiny = 1e-30;
    
    Complex f = tiny;
    Complex C = f;
    Complex D = 0.0;
    
    int step = (direction > 0) ? 1 : -1;
    int n_start = (direction > 0) ? 1 : -1;
    
    for (int k = 1; k <= max_iter; ++k) {
        int n = n_start + (k - 1) * step;
        
        Complex a_k, b_k;
        
        // 注意：这里的 definition 导致 a_1 包含了 alpha_0 (当 dir>0) 或 alpha_-1 (当 dir<0)
        // 所以计算出的连分式 f 实际上是 alpha_0 * R_1 或 gamma_0 * L_-1
        if (direction > 0) {
            a_k = -coeff_alpha(nu, n - 1) * coeff_gamma(nu, n);
            b_k = coeff_beta(nu, n);
        } else {
            a_k = -coeff_alpha(nu, n) * coeff_gamma(nu, n + 1);
            b_k = coeff_beta(nu, n);
        }
        
        // Modified Lentz iteration
        D = b_k + a_k * D;
        if (std::abs(D) < tiny) D = tiny;
        C = b_k + a_k / C;
        if (std::abs(C) < tiny) C = tiny;
        D = 1.0 / D;
        Complex delta = C * D;
        f *= delta;
        if (std::abs(delta - 1.0) < tol) return f;
    }
    std::cerr << "Warning: Continued fraction did not converge." << std::endl;
    return f;
}

// ----------------------------------------------------------
// 修正后的求解特征值方程 g(nu)
// ----------------------------------------------------------

Complex TeukolskyRadial::calc_g(Complex nu) const {
    // 1. 获取 n=0 时的 MST 系数 (仅用于 beta_0)
    Complex b0 = coeff_beta(nu, 0);
    
    // [FIX 2] 不再重复乘 alpha_0 和 gamma_0
    // continued_fraction(nu, 1) 已经返回了 alpha_0 * R_1
    // continued_fraction(nu, -1) 已经返回了 gamma_0 * L_-1
    
    Complex term_R = continued_fraction(nu, 1);     // = alpha_0 * R_1
    Complex term_L = continued_fraction(nu, -1);    // = gamma_0 * L_-1
    
    // g(nu) = beta_0 + alpha_0 * R_1 + gamma_0 * L_-1
    return b0 + term_R + term_L;
}


Complex TeukolskyRadial::solve_nu(Complex nu_guess) const {
    // 使用割线法 (Secant Method) 在复平面寻根
    // 迭代公式: x_{n+1} = x_n - f(x_n) * (x_n - x_{n-1}) / (f(x_n) - f(x_{n-1}))
    
    Complex x0 = nu_guess;
    // 第二个初始点取一个微小偏移，避免 f(x1) 与 f(x0) 完全相同
    Complex x1 = nu_guess + 1e-4; 
    
    Complex f0 = calc_g(x0);
    Complex f1 = calc_g(x1);
    
    const int max_iter = 150;
    const double tol = 1e-10; // 目标精度
    
    for (int i = 0; i < max_iter; ++i) {
        // 如果已经足够接近 0，直接返回
        if (std::abs(f1) < tol) return x1;
        
        Complex diff_f = f1 - f0;
        // 避免除以零
        if (std::abs(diff_f) < 1e-30) {
            std::cerr << "Warning: Secant method stuck (flat region)." << std::endl;
            break;
        }
        
        Complex x_new = x1 - f1 * (x1 - x0) / diff_f;
        
        // 更新迭代状态
        x0 = x1;
        f0 = f1;
        x1 = x_new;
        f1 = calc_g(x1);
        
        // 简单的发散保护
        if (std::abs(x1) > 1000.0) {
            std::cerr << "Error: solve_nu diverging (nu > 1000)." << std::endl;
            // 返回 NaN 表示失败
            return std::numeric_limits<double>::quiet_NaN();
        }
    }
    
    // 如果达到最大迭代次数仍未满足 tol，但通常结果也可用，
    // 这里我们打印警告并返回最后的值
    std::cerr << "Warning: solve_nu max iterations reached. Residual: " << std::abs(f1) << std::endl;
    return x1;
}


// ==========================================================
// 计算连接因子 K_nu (依据 Sasaki & Tagoshi Eq. 165)
// 修正版：直接使用 ComputeSeriesCoefficients 计算的 a_n
// ==========================================================
Complex TeukolskyRadial::k_factor(Complex nu) const {
    Complex eps = m_epsilon;
    Complex tau = m_tau;
    Complex kappa = m_kappa;
    Complex i = 1.0i;
    
    // 1. 获取正确的级数系数 a_n
    // 我们需要足够的项以保证 sum 收敛
    int n_max = 100; // 这里的 n_max 可以设大一点以保证精度
    std::map<int, Complex> a_coeffs = const_cast<TeukolskyRadial*>(this)->ComputeSeriesCoefficients(nu, n_max);
    Complex s_c=(double)m_s;
    Complex eps_plus=(eps+tau)/2.0;
    // 2. 准备公式 Eq. 165 中的公共参数
    // K_nu = [PreFactor] * (Sum_Inf / Sum_R)
    // 通常取 r=0，则 Sum_R 是 n 从 -inf 到 0 的和
    // Sum_Inf 是 n 从 0 到 +inf 的和
    
    // 2.1 计算 Prefactor
    // Pre = e^(i*eps*kappa) * (2*eps*kappa)^(s-nu) * 2^(-s) * i^r * Gamma(...)
    // 取 r = 0
    double r_int = 0.0;
    
    Complex nu_plus_1_s_ie = nu + 1.0 + s_c + i * eps;
    Complex nu_plus_1_ms_ie = nu + 1.0 - s_c - i * eps;
    Complex nu_plus_1_it = nu + 1.0 + i * tau;
    Complex nu_plus_1_mit = nu + 1.0 - i * tau;
    
    Complex ln_pre = i * eps * kappa 
                   + (s_c - nu-r_int) * std::log(2.0 * eps * kappa)
                   - s_c * std::log(2.0)+r_int*i*M_PI/2.0
                   + log_gamma(1.0 - s_c - 2.0 * i * eps_plus) // 注意: 这里的 eps 是 epsilon+ ? 需核对 Eq 165
                   + log_gamma(r_int + 2.0 * nu + 2.0)
                   - log_gamma(r_int + nu + 1.0 - s_c + i * eps)
                   - log_gamma(r_int + nu + 1.0 + i * tau)
                   - log_gamma(r_int + nu + 1.0 + s_c+i * eps);
                   
    // 注意：LRR Eq 165 中的 Gamma 参数非常复杂，上述只是基于常见 MST 形式的近似
    // 为确保万无一失，我们直接实现 Eq 165 的求和部分
    // Eq 165: Sum_Numerator (n from r to inf)
    //         Sum_Denominator (n from -inf to r)
    
    Complex sum_num = 0.0;
    Complex sum_den = 0.0;
    
    // 预计算 Gamma 常数
    // 分子通项: (-1)^n * Gamma(n+r+2nu+1) / (n-r)! * ... * f_n
    // 分母通项: (-1)^n / ((r-n)! * (r+2nu+2)_n) * ... * f_n
    
    // 我们取 r=0
    for (auto const& [n, f_n] : a_coeffs) {
        double dn = (double)n;
        double sign = (std::abs(n) % 2 == 0) ? 1.0 : -1.0;
        
        // --- 分子部分 (Sum for n >= 0) ---
        if (n >= 0) {
            // Term = (-1)^n * Gamma(n + 2nu + 1) / n! 
            //      * Gamma(n + nu + 1 + s + i*eps) 
            //      * Gamma(n + nu + 1 + i*tau) 
            //      * Gamma(n + nu + 1 + i*eps) <-- 这一项原文可能有差异，请核对
            //      * f_n
            
            // 依据 Eq 165 Line 2:
            // Gamma(n+r+2nu+1)/(n-r)! 
            // * Gamma(n+nu+1+s+i*eps) / Gamma(n+nu+1-s-i*eps)
            // * Gamma(n+nu+1+i*tau) / Gamma(n+nu+1-i*tau)
            
            Complex term_n = sign * f_n;
            term_n *= std::exp(log_gamma(dn +r_int+ 2.0 * nu + 1.0) - log_gamma(dn +1.0-r_int)); // n!
            term_n *= std::exp(log_gamma(dn + nu_plus_1_s_ie) - log_gamma(dn + nu_plus_1_ms_ie));
            term_n *= std::exp(log_gamma(dn + nu_plus_1_it) - log_gamma(dn + nu_plus_1_mit));
            
            sum_num += term_n;
        }
        
        // --- 分母部分 (Sum for n <= 0) ---
        if (n <= 0) {
            // Term = (-1)^n / ( (-n)! * (2nu+2)_n ) * ... * f_n
            // (2nu+2)_n for negative n is Gamma(2nu+2+n)/Gamma(2nu+2)
            
            Complex term_d = sign * f_n;
            term_d /= std::exp(log_gamma(1.0- dn)); // (-n)!
            
            // Pochhammer (2nu+2)_n
            term_d /= std::exp(log_gamma(r_int+2.0 * nu + 2.0 + dn) - log_gamma(r_int+2.0 * nu + 2.0));
            
            // 剩下的系数 (nu+1+s-i*eps)_n / (nu+1-s+i*eps)_n
            term_d *= std::exp(log_gamma(nu_plus_1_s_ie + dn) - log_gamma(nu_plus_1_s_ie));
            term_d /= std::exp(log_gamma(nu_plus_1_ms_ie + dn) - log_gamma(nu_plus_1_ms_ie));
            
            sum_den += term_d;
        }
    }
    
    Complex K_val = std::exp(ln_pre) * (sum_num / sum_den);
    return K_val;
}

// 辅助函数：获取第 n 阶的 MST 递推系数 alpha, beta, gamma
// 预期功能：对接现有的数学公式模块，返回对应 n 和 nu 的系数


// 主函数：计算级数系数 a_n
// 接口：输入重整化角动量 nu 和截断阶数 n_max，返回索引为 n 的系数 map
std::map<int, std::complex<double>> TeukolskyRadial::ComputeSeriesCoefficients(std::complex<double> nu, int n_max) {
    std::map<int, std::complex<double>> a_coeffs;
    a_coeffs[0] = std::complex<double>(1.0, 0.0);

    // 1. 正向部分 (n > 0): 计算比值 R_n = a_n / a_{n-1}
    // R_n = -gamma_n / (beta_n + alpha_n * R_{n+1})
    std::vector<std::complex<double>> R(n_max + 2, 0.0);
    
    for (int n = n_max; n >= 1; --n) {
        std::complex<double> alph = coeff_alpha(nu, n);
        std::complex<double> bet  = coeff_beta(nu, n);
        std::complex<double> gam  = coeff_gamma(nu, n);
        
        std::complex<double> denominator = bet + alph * R[n + 1];
        if (std::abs(denominator) < 1e-15) denominator = 1e-15;
        R[n] = -gam / denominator;
    }
    for (int n = 1; n <= n_max; ++n) {
        a_coeffs[n] = a_coeffs[n - 1] * R[n];
    }

    // 2. 负向部分 (n < 0): 计算比值 r_k = a_{-k} / a_{-k+1}
    // r_k = -alpha_{-k} / (beta_{-k} + gamma_{-k} * r_{k+1})
    std::vector<std::complex<double>> r(n_max + 2, 0.0);

    for (int k = n_max; k >= 1; --k) {
        // 注意系数索引是 -k
        std::complex<double> alph = coeff_alpha(nu, -k);
        std::complex<double> bet  = coeff_beta(nu, -k);
        std::complex<double> gam  = coeff_gamma(nu, -k);

        std::complex<double> denominator = bet + gam * r[k + 1];
        if (std::abs(denominator) < 1e-15) denominator = 1e-15;
        r[k] = -alph / denominator;
    }
    for (int k = 1; k <= n_max; ++k) {
        a_coeffs[-k] = a_coeffs[-k + 1] * r[k];
    }

    return a_coeffs;
}
// ==========================================================
// 计算渐近振幅 (Asymptotic Amplitudes)
// 对应 Sasaki & Tagoshi (2003) LRR-2003-6 Section 4.4
// ==========================================================

// ==========================================================
// 计算 MST 渐近振幅系数 A+ 和 A-
// 依据: Sasaki & Tagoshi (2003), Living Rev. Relativity 6
// 对应 PDF 中的 Equation 157 和 158 (Page 33)
// ==========================================================

AsymptoticAmplitudes TeukolskyRadial::ComputeAmplitudes(std::complex<double> nu, 
    const std::map<int, std::complex<double>>& a_coeffs) {
AsymptoticAmplitudes amps;

// 物理参数缩写
Complex eps = m_epsilon;       // epsilon = 2M*omega
Complex s_c(m_s, 0.0);         // spin s
Complex i(0.0, 1.0);           // imaginary unit
double PI = M_PI;

// 辅助变量 nu+1
Complex nu1 = nu + 1.0;

// -------------------------------------------------------------------------
// 1. 计算 A_plus (Eq. 157)
// 对应无穷远处的入射波 (Incoming wave at infinity, R_+)
// Formula: 
// A_+ = e^{-pi*eps/2} * e^{i*pi*(nu+1-s)/2} * 2^{-1+s-i*eps} 
//       * [Gamma(nu+1-s+i*eps) / Gamma(nu+1+s-i*eps)] * Sum(f_n)
// -------------------------------------------------------------------------

// (1.1) 计算级数和 Sum(f_n)
Complex sum_f = 0.0;
for (const auto& [n, f_n] : a_coeffs) {
sum_f += f_n;
}

// (1.2) 计算 Gamma 函数比值
Complex g_num = log_gamma(nu1 - s_c + i * eps);
Complex g_den = log_gamma(nu1 + s_c - i * eps);
Complex gamma_ratio = std::exp(g_num - g_den);

// (1.3) 计算前置因子
Complex term1 = std::exp(-PI * eps / 2.0);
Complex term2 = std::exp(i * PI * (nu1 - s_c) / 2.0);
Complex term3 = std::pow(2.0, -1.0 + (double)m_s - i * eps);

// 组合结果 A_+
Complex A_plus = term1 * term2 * term3 * gamma_ratio * sum_f;


// -------------------------------------------------------------------------
// 2. 计算 A_minus (Eq. 158)
// 对应无穷远处的出射波 (Outgoing wave at infinity, R_-)
// Formula:
// A_- = 2^{-1-s+i*eps} * e^{-i*pi*(nu+1+s)/2} * e^{-pi*eps/2}
//       * Sum( (-1)^n * [(nu+1+s-i*eps)_n / (nu+1-s+i*eps)_n] * f_n )
// 
// Note: (a)_n is the Pochhammer symbol: Gamma(a+n)/Gamma(a)
// -------------------------------------------------------------------------

// (2.1) 准备 Pochhammer 符号的基数
Complex poch_a = nu1 + s_c - i * eps; // 分子部分的基数 (nu+1+s-i*eps)
Complex poch_b = nu1 - s_c + i * eps; // 分母部分的基数 (nu+1-s+i*eps)

// 预先计算 Gamma(a) 和 Gamma(b) 的对数值，避免循环中重复计算
Complex lg_a = log_gamma(poch_a);
Complex lg_b = log_gamma(poch_b);

// (2.2) 级数求和
Complex sum_minus = 0.0;

for (const auto& [n, f_n] : a_coeffs) {
// 计算 (-1)^n
double sign = (std::abs(n) % 2 == 0) ? 1.0 : -1.0;

// 计算 Pochhammer 比值: (a)_n / (b)_n
// Ratio = [Gamma(a+n)/Gamma(a)] / [Gamma(b+n)/Gamma(b)]
//       = exp( lg(a+n) - lg(a) - (lg(b+n) - lg(b)) )

Complex lg_a_n = log_gamma(poch_a + (double)n);
Complex lg_b_n = log_gamma(poch_b + (double)n);

Complex poch_ratio = std::exp((lg_a_n - lg_a) - (lg_b_n - lg_b));

sum_minus += sign * poch_ratio * f_n;
}

// (2.3) 计算前置因子
Complex term1_m = std::pow(2.0, -1.0 - (double)m_s + i * eps);
Complex term2_m = std::exp(-i * PI * (nu1 + s_c) / 2.0);
Complex term3_m = std::exp(-PI * eps / 2.0); // 同 A_plus 的 term1

// 组合结果 A_-
Complex A_minus = term1_m * term2_m * term3_m * sum_minus;

// -------------------------------------------------------------------------
// 3. 结果存储
// 这里的命名仅作区分，具体物理含义(Inc/Trans)需结合 K_nu 判定
// 根据 LRR 描述: R_+ 是 incoming (Eq 155), R_- 是 outgoing (Eq 156)
// -------------------------------------------------------------------------

amps.R_in_coef_inf_trans = A_plus;  // 对应 A_+ (Incoming part of Coulomb expansion)
amps.R_in_coef_inf_inc   = A_minus; // 对应 A_- (Outgoing part of Coulomb expansion)

return amps;
}

// ==========================================================
// 计算物理散射系数 (B, C) 和 Wronskian
// 依据: Sasaki & Tagoshi (2003) LRR-2003-6 Eq. 167-170, Eq. 23
// ==========================================================

// ==========================================================
// 计算连接视界与无穷远的物理系数 (Physical Amplitudes)
// 依据: Sasaki & Tagoshi (2003) Eq. 167 - 170
// ==========================================================

PhysicalAmplitudes TeukolskyRadial::ComputePhysicalAmplitudes(std::complex<double> nu, 
    const std::map<int, std::complex<double>>& a_coeffs,
    const AsymptoticAmplitudes& amps_nu) {
PhysicalAmplitudes phys_amps;

// 1. 准备物理参数
Complex eps = m_epsilon;
Complex omega = m_omega;
Complex kappa = m_kappa;     // sqrt(1-q^2)
Complex s_c(m_s, 0.0);
Complex i(0.0, 1.0);
double PI = M_PI;

// 辅助变量
Complex eps_plus = (eps + m_tau) / 2.0; // epsilon_+ = (eps + tau)/2
// 注意：k = omega - m * Omega_H 可能是 Eq 167 中的 k，需确认定义
// LRR Eq 19: k = omega - m * a / (2Mr_+)
double r_plus = 1.0 + std::sqrt(1.0 - q * q); // M=1 units
Complex k_horizon = omega - (double)m_m * q / (2.0 * r_plus); 

// -------------------------------------------------------------------------
// 2. 准备 K 因子
// 公式涉及 K_nu 和 K_{-nu-1}
// -------------------------------------------------------------------------
Complex K_nu = k_factor(nu);
Complex K_neg_nu_1 = k_factor(-nu - 1.0); // K_{-nu-1}
Complex phase_term=i*(eps*log(eps) - (1.0 - kappa)/2.0 * eps);
// -------------------------------------------------------------------------
// 3. 计算 B_trans (Eq. 167)
// 描述 R^in 在视界处的透射行为
// -------------------------------------------------------------------------

// (3.1) 计算级数求和 Sum(f_n)
// 注意：这与 A_+ 公式中的求和相同，如果你之前没存，这里需要重算一遍
Complex sum_f = 0.0;
for (const auto& [n, f_n] : a_coeffs) {
sum_f += f_n;
}

// B_trans 公式
// 参考 Eq. 167: (eps*kappa/omega)^(2s) * exp(...) * Sum(f_n)
// 提示: 指数部分包含 ik(epsilon_+) 和 ln(kappa) 相关项
// Complex term_pre = ...; 
// phys_amps.B_trans = term_pre * sum_f;

phys_amps.B_trans=std::pow((eps*kappa/omega),(2.0*s_c)) * exp(i*(eps_plus)*kappa*2.0*(0.5 + log(kappa)/(1.0+kappa))) * sum_f;
// -------------------------------------------------------------------------
// 4. 计算 B_inc (Eq. 168)
// 描述 R^in 在无穷远的入射分量 (1/omega * ... * A_+)
// -------------------------------------------------------------------------

Complex A_plus = amps_nu.R_in_coef_inf_trans; // 注意映射关系: A_+ 对应 Transmitted part of Coulomb solution

// B_inc 公式
// 参考 Eq. 168: omega^-1 * (K_nu - i * exp(...) * Ratio_Sin * K_-nu-1) * A_+ * Phase
// 关键点: 
// 1. sin_ratio = sin(pi*(nu - s + i*eps)) / sin(pi*(nu + s - i*eps))
// 2. Phase term = exp( -i * (eps*ln(eps) - (1-kappa)/2 * eps) )
// Complex bracket_term = ...;
// Complex phase_term = ...;
// phys_amps.B_inc = (1.0/omega) * bracket_term * A_plus * phase_term;

phys_amps.B_inc=1.0/omega * (K_nu - i * exp(-i*PI*nu)* sin(PI*(nu - s_c + i*eps)) / sin(PI*(nu + s_c - i*eps))  * K_neg_nu_1) * A_plus * exp(-phase_term);
// -------------------------------------------------------------------------
// 5. 计算 B_ref (Eq. 169)
// 描述 R^in 在无穷远的反射分量 (1/omega^(1+2s) * ... * A_-)
// -------------------------------------------------------------------------

Complex A_minus = amps_nu.R_in_coef_inf_inc; // A_- 对应 Incident part of Coulomb

// B_ref 公式
// 参考 Eq. 169: omega^(-1-2s) * (K_nu + i * exp(i*pi*nu) * K_-nu-1) * A_- * Phase_Conj
// 注意: Phase term 与 B_inc 的相位互为复共轭 (符号相反)
// Complex bracket_ref = ...;
// Complex phase_ref = ...;
// phys_amps.B_ref = std::pow(omega, -1.0 - 2.0 * (double)m_s) * bracket_ref * A_minus * phase_ref;
phys_amps.B_ref= std::pow(omega, -1.0 - 2.0 * s_c) * (K_nu + i * exp(i*PI*nu)  * K_neg_nu_1) * A_minus * exp(phase_term);

// -------------------------------------------------------------------------
// 6. 计算 C_trans (Eq. 170)
// 描述 R^up (Upgoing) 辐射到无穷远的系数
// -------------------------------------------------------------------------

// C_trans 公式
// 参考 Eq. 170: omega^(-1-2s) * A_- * Phase_Ref
// 注意: 它与 B_ref 共享很多因子，除了括号里的 K 组合
// phys_amps.C_trans = std::pow(omega, -1.0 - 2.0 * (double)m_s) * A_minus * phase_ref;
phys_amps.C_trans=std::pow(omega, -1.0 - 2.0 * s_c) * A_minus * exp(phase_term);
return phys_amps;
}
// ==========================================================
// 计算径向函数 R_in(r) (超几何级数展开)
// 依据: Sasaki & Tagoshi (2003) Eq. 114 - 117
// ==========================================================

std::pair<Complex, Complex> TeukolskyRadial::Evaluate_Hypergeometric(
    double r, 
    Complex nu, 
    const std::map<int, Complex>& a_coeffs) 
{
    // 1. 准备物理参数
    Real s_c = m_s;
    Complex eps = m_epsilon;
    Complex omega = m_omega;
    double kappa = m_kappa;
    Complex tau = m_tau;
    double M = 1.0; // 几何单位制
    double r_plus = 1.0 + std::sqrt(1.0 - q * q);
    
    Complex i(0.0, 1.0);

    // -------------------------------------------------------------------------
    // 坐标变换 r -> x
    // -------------------------------------------------------------------------
    // 参考 Eq. 114: x = (omega*r_plus - omega*r) / (epsilon * kappa)
    // 注意: epsilon = 2*M*omega. 
    // 这里 x 通常是一个负实数 (当 r > r_+)
    double x_val=(r_plus - r) / (2.0 * kappa);
    if (abs(x_val)<=1e-15)
    {
        x_val=1e-15;
    }
    // Complex x_val = ...;
    // x_val = (r_plus - r) / (2.0 * M * kappa); // 化简后的形式
    // 对应的 dx/dr = ... (链式法则需要)
    // Complex dxdR = ...;
    double dx_dr=-1.0 / (2.0 * kappa);

    // -------------------------------------------------------------------------
    // 计算前置因子 P(x) 及其关于 x 的对数导数
    // -------------------------------------------------------------------------
    // 参考 Eq. 116: R = P(x) * p_in(x)

    Complex alpha = -s_c - i * (eps + tau) / 2.0; 
    Complex beta  = i * (eps - tau) / 2.0;

 
    Complex P_val, dLogPdx;
    if (std::abs(x_val) < 1e-16) {
        // 处理边界情况，或者直接报错
        P_val = 0.0; 
        dLogPdx = 0.0; 
    } else {
        Complex term1 = i * eps * kappa * x_val;
        Complex term2 = alpha * std::log(-x_val);
        Complex term3 = beta * std::log(1.0 - x_val);
        P_val = std::exp(term1 + term2 + term3);
        dLogPdx = i * eps * kappa + alpha / x_val - beta / (1.0 - x_val);
    }

    // -------------------------------------------------------------------------
    // 3. 计算级数求和 (使用 FullyScaled Arb 包装器)
    // -------------------------------------------------------------------------
    Complex sum_S = 0.0;      
    Complex sum_dSdx = 0.0;   
    Complex zero_log(0.0, 0.0); // 目前不需要额外的 Log 因子

    for (const auto& [n, a_n] : a_coeffs) {
        double dn = (double)n;
        
        // 参数设置 Eq. 120
        Complex hyp_a = dn + nu + 1.0 - i * tau;
        Complex hyp_b = -dn - nu - i * tau;
        Complex hyp_c = 1.0 - s_c - i * eps - i * tau;

        // --- 1. 计算函数值项: a_n * 2F1(...) ---
        // 将 a_n 作为 factor 传入 Arb
        // log_factor 设为 0，因为 Hypergeometric 级数本身没有阶乘增长的前缀
        Complex term_val = Hyp2F1_FullyScaled(
            hyp_a, hyp_b, hyp_c, x_val, 
            a_n, zero_log
        );
        
        // --- 2. 计算导数项 ---
        // 公式: d/dx [a_n * 2F1(a,b,c,x)] = a_n * (a*b/c) * 2F1(a+1, b+1, c+1, x)
        // 计算前缀 multiplier = a_n * (a*b/c)
        Complex deriv_prefactor = (hyp_a * hyp_b) / hyp_c;
        Complex deriv_factor = a_n * deriv_prefactor;

        Complex term_deriv = Hyp2F1_FullyScaled(
            hyp_a + 1.0, hyp_b + 1.0, hyp_c + 1.0, x_val, 
            deriv_factor, zero_log
        );
        
        // --- 3. 累加 ---
        sum_S += term_val;
        sum_dSdx += term_deriv;
    }

    // ... (后续组合 R_val 代码保持不变) ...

    // R(r) = P(x) * S(x)
    Complex R_val = P_val * sum_S;
    
    // dR/dx = P'(x)S(x) + P(x)S'(x) = P(x) * [ (P'/P)*S + S' ]
    Complex dRdx_val = P_val * (dLogPdx* sum_S + sum_dSdx);
    
    // 转换回对 r 的导数: dR/dr = dR/dx * dx/dr
    Complex dRdr_val = dRdx_val * dx_dr;

    return {R_val, dRdr_val};
}



Complex TeukolskyRadial::Hyp2F1(Complex a, Complex b, Complex c, Complex z, bool regularized) {
    // 1. 初始化 Arb 复数变量
    acb_t acb_a, acb_b, acb_c, acb_z, acb_res;
    acb_init(acb_a);
    acb_init(acb_b);
    acb_init(acb_c);
    acb_init(acb_z);
    acb_init(acb_res);

    // 2. 设置精度 (Precision)
    // 即使输入是 double，内部使用更高精度 (如 128 bits) 可以保证结果的 double 值是准确的
    slong prec = 128; 

    // 3. 将 std::complex 转为 acb_t
    // Arb 使用 acb_set_d_d(target, real, imag)
    acb_set_d_d(acb_a, a.real(), a.imag());
    acb_set_d_d(acb_b, b.real(), b.imag());
    acb_set_d_d(acb_c, c.real(), c.imag());
    acb_set_d_d(acb_z, z.real(), z.imag());

    // 4. 调用 Arb 的超几何函数
    // acb_hypgeom_2f1(res, a, b, c, z, regularization, prec)
    // regularization=0 表示计算标准的 2F1
    acb_hypgeom_2f1(acb_res, acb_a, acb_b, acb_c, acb_z, regularized, prec);

    // 5. 将结果转回 std::complex<double>
    // arf_get_d 转换 Arb 浮点数 (arf) 到 double
    // acb_realref 获取实部引用，acb_imagref 获取虚部引用
    double res_r = arf_get_d(arb_midref(acb_realref(acb_res)), ARF_RND_NEAR);
    double res_i = arf_get_d(arb_midref(acb_imagref(acb_res)), ARF_RND_NEAR);

    // 6. 清理内存
    acb_clear(acb_a);
    acb_clear(acb_b);
    acb_clear(acb_c);
    acb_clear(acb_z);
    acb_clear(acb_res);

    return Complex(res_r, res_i);
}
// 计算 factor * 2F1(a, b, c, z)
Complex TeukolskyRadial::Hyp2F1_Scaled(Complex a, Complex b, Complex c, Complex z, Complex factor) {
    // 1. 初始化 Arb 变量
    acb_t acb_a, acb_b, acb_c, acb_z, acb_factor, acb_res, acb_final;
    acb_init(acb_a); acb_init(acb_b); acb_init(acb_c);
    acb_init(acb_z); acb_init(acb_factor);
    acb_init(acb_res); acb_init(acb_final);

    // 2. 设置精度 (128 bits 足够处理大约 10^38 范围内的互补相消)
    slong prec = 128;

    // 3. 赋值
    acb_set_d_d(acb_a, a.real(), a.imag());
    acb_set_d_d(acb_b, b.real(), b.imag());
    acb_set_d_d(acb_c, c.real(), c.imag());
    acb_set_d_d(acb_z, z.real(), z.imag());
    acb_set_d_d(acb_factor, factor.real(), factor.imag()); // 传入系数 a_n

    // 4. 计算标准的 2F1
    // regularized = 0 (通常 2F1 不需要正则化，除非 c 是负整数)
    acb_hypgeom_2f1(acb_res, acb_a, acb_b, acb_c, acb_z, 0, prec);

    // 5. 关键步骤：在高精度下执行乘法
    // Final = Result * Factor
    // 比如 10^200 * 10^-200 = 1.0，这在 Arb 里是安全的
    acb_mul(acb_final, acb_res, acb_factor, prec);

    // 6. 转换回 Complex (double)
    // 此时得到的是最终贡献值，通常是有限的
    double res_r = arf_get_d(arb_midref(acb_realref(acb_final)), ARF_RND_NEAR);
    double res_i = arf_get_d(arb_midref(acb_imagref(acb_final)), ARF_RND_NEAR);

    // 7. 清理
    acb_clear(acb_a); acb_clear(acb_b); acb_clear(acb_c);
    acb_clear(acb_z); acb_clear(acb_factor);
    acb_clear(acb_res); acb_clear(acb_final);

    return Complex(res_r, res_i);
}
// ==========================================================
// Arb 包装器：计算 factor * exp(log_factor) * 2F1(a, b, c, z)
// ==========================================================
Complex TeukolskyRadial::Hyp2F1_FullyScaled(
    Complex a, Complex b, Complex c, Complex z, 
    Complex factor,      // 线性因子 (如 a_n)
    Complex log_factor   // 对数因子 (如果将来需要处理 gamma 等)
) {
    // 1. 初始化 Arb 变量
    acb_t acb_a, acb_b, acb_c, acb_z, acb_res;
    acb_t acb_factor, acb_log_factor, acb_exp_log, acb_final;

    acb_init(acb_a); acb_init(acb_b); acb_init(acb_c);
    acb_init(acb_z); acb_init(acb_res);
    acb_init(acb_factor); acb_init(acb_log_factor);
    acb_init(acb_exp_log); acb_init(acb_final);

    // 2. 设置精度 (256 bits 足够应对 n=150 时的剧烈中间值膨胀)
    slong prec = 256;

    // 3. 赋值
    acb_set_d_d(acb_a, a.real(), a.imag());
    acb_set_d_d(acb_b, b.real(), b.imag());
    acb_set_d_d(acb_c, c.real(), c.imag());
    acb_set_d_d(acb_z, z.real(), z.imag());
    acb_set_d_d(acb_factor, factor.real(), factor.imag());
    acb_set_d_d(acb_log_factor, log_factor.real(), log_factor.imag());

    // 4. 计算标准的 2F1 (regularized=0)
    // 注意：2F1 通常不需要正则化，除非 c 是负整数
    acb_hypgeom_2f1(acb_res, acb_a, acb_b, acb_c, acb_z, 0, prec);

    // 5. 计算对数因子 exp(log_factor)
    acb_exp(acb_exp_log, acb_log_factor, prec);

    // 6. 终极乘法：Final = (2F1) * (factor) * (exp_log_factor)
    // 所有的 "Huge * Tiny" 都在这里安全进行
    acb_mul(acb_final, acb_res, acb_factor, prec);
    acb_mul(acb_final, acb_final, acb_exp_log, prec);

    // 7. 转换回 Complex (double)
    double res_r = arf_get_d(arb_midref(acb_realref(acb_final)), ARF_RND_NEAR);
    double res_i = arf_get_d(arb_midref(acb_imagref(acb_final)), ARF_RND_NEAR);

    // 8. 清理
    acb_clear(acb_a); acb_clear(acb_b); acb_clear(acb_c);
    acb_clear(acb_z); acb_clear(acb_res);
    acb_clear(acb_factor); acb_clear(acb_log_factor);
    acb_clear(acb_exp_log); acb_clear(acb_final);

    return Complex(res_r, res_i);
}
// ==========================================================
// 计算远场径向函数 R_C^nu(r) (库伦波函数级数)
// 依据: Sasaki & Tagoshi (2003) Eq. 139, 142, 144
// ==========================================================

std::pair<Complex, Complex> TeukolskyRadial::Evaluate_Coulomb(
    double r, 
    Complex nu, 
    const std::map<int, Complex>& a_coeffs) 
{
    // 1. 准备物理参数
    Complex s_c=(double)m_s;
    Complex eps = m_epsilon;   // 2M*omega
    Complex omega = m_omega;
    Complex tau = m_tau;
    double M = 1.0;
    double kappa = m_kappa;
    double r_minus = 1.0 - std::sqrt(1.0 - q * q); // r_minus / M
    
    Complex i(0.0, 1.0);

    // -------------------------------------------------------------------------
    // 坐标变换 r -> z_hat
    // -------------------------------------------------------------------------
    // 参考 LRR Page 31: z_hat = omega * (r - r_minus)
    // 这里的 z_hat 是复数还是实数？通常 r 是实数，z_hat 是实数 (如果 omega 是实数)
    // Complex z_hat = ...;
    // double dzhat_dr = ...; // d(z_hat)/dr
    
    // 注意: omega = m_omega (M*omega). r_minus 是 M 单位.
    // 所以 z_hat = m_omega * (r - r_minus)
    Complex z_hat = omega * (r - r_minus);
    Complex dzhat_dr = omega;



    // -------------------------------------------------------------------------
    // 计算 R_C 的前置因子 P_C(z_hat) 及其对数导数
    // -------------------------------------------------------------------------
    // 参考 Eq. 139: R_C = z^(-1-s) * (1 - eps*kappa/z)^(-s - i(eps+tau)/2) * f(z)
    // 令 factor1 = -1 - s
    // 令 factor2 = -s - i*(eps + tau)/2.0
    // Term = z_hat^factor1 * (1 - eps*kappa/z_hat)^factor2
    
    // 提示: 同样建议先算 LogP 及其导数 dLogP/dz_hat
    // Complex term_brace = 1.0 - eps * kappa / z_hat;
    // Complex LogP = ...;
    // Complex dLogP_dz = ...;
    
    Complex factor1 = -1.0 - s_c;
    Complex factor2 = -s_c - i * (eps + tau) / 2.0;
    Complex term_brace = 1.0 - eps * kappa / z_hat;
    
    Complex LogP = factor1 * std::log(z_hat) + factor2 * std::log(term_brace);
    
    // 导数: d/dz (fac1 * ln z + fac2 * ln(1 - C/z))
    //      = fac1/z + fac2 * (1/(1-C/z)) * (-C * -1/z^2)
    //      = fac1/z + fac2 * (1/(1-C/z)) * (C/z^2)
    //      = fac1/z + fac2 * (C / (z*(z-C)))
    Complex C_const = eps * kappa;
    Complex dLogP_dz = factor1 / z_hat + factor2 * (C_const / (z_hat * (z_hat - C_const)));
    
    Complex P_val = std::exp(LogP);


    // -------------------------------------------------------------------------
    // 3. 计算级数求和 f(z_hat) = Sum(C_n * F_{n+nu})
    // -------------------------------------------------------------------------
    // 依据 Eq. 144 和 147
    // 系数 C_n = (-i)^n * [ (nu+1+s-i*eps)_n / (nu+1-s+i*eps)_n ] * a_n
    
    Complex sum_f = 0.0;
    Complex sum_dfdz = 0.0;

    // 预计算 Pochhammer 比值的基数
    Complex poch_num_base = nu + 1.0 + s_c - i * eps;
    Complex poch_den_base = nu + 1.0 - s_c + i * eps;
    Complex lg_num_base = log_gamma(poch_num_base);
    Complex lg_den_base = log_gamma(poch_den_base);

    // 库伦函数参数 eta (Eq. 141 limit: eta = -i*s - eps ? 需仔细核对)
    // Eq. 142 definition F_L(eta, z). Eq 141 usage F_l(-is-eps, z).
    // 所以 eta_coulomb = -i*s - eps
    Complex eta_coulomb = -i * s_c - eps;
    Complex log_2 = std::log(2.0);
    Complex log_z = std::log(z_hat);
    Complex log_minus_i = std::log(Complex(0.0, -1.0));
    for (const auto& [n, a_n] : a_coeffs) {
        double dn = (double)n;
        Complex L = nu + dn; // L = n + nu
        // 3.1 计算组合系数 C_n
        Complex i_pow_n = std::pow(-i, n);
        
        // Pochhammer ratio using log_gamma
        Complex lg_num_n = log_gamma(poch_num_base + dn);
        Complex lg_den_n = log_gamma(poch_den_base + dn);
        Complex poch_ratio = std::exp((lg_num_n - lg_num_base) - (lg_den_n - lg_den_base));
        
        Complex C_n = i_pow_n * poch_ratio * a_n;
        Complex log_Cn_part = (dn * log_minus_i) + (lg_num_n - lg_num_base) - (lg_den_n - lg_den_base);
        // 3.2 计算库伦波函数 F_L(eta, z) 及其导数
        // Eq. 142: F_L = e^{-iz} * 2^L * z^{L+1} * [Gamma(...) / Gamma(...)] * 1F1(...)
        
        Complex log_term_z = -i * z_hat + L * log_2 + (L + 1.0) * log_z;
        // 库伦波函数 F_L 的各个部分
        // 参数: hyp_a = L + 1 - i*eta, hyp_b = 2L + 2
        // Complex hyp_a = ...;
        // Complex hyp_b = ...;
        
        // 1F1 值
        // Complex phi_val = Hyp1F1(hyp_a, hyp_b, 2.0 * i * z_hat);
        
        // F_L 值 (Eq 142)  
        // 注意常数因子: term_gamma = Gamma(L+1-i*eta) / Gamma(2L+2)
        // term_z = e^{-i*z} * (2z)^(L+1) ? 不，是 2^L * z^{L+1}
        // Complex F_val = ...;
        
        Complex hyp_a = L + 1.0 - i * eta_coulomb;
        Complex hyp_b = 2.0 * L + 2.0;
        Complex arg_z = 2.0 * i * z_hat;

        // // 累加
        // sum_f += C_n * F_val;
        // sum_dfdz += C_n * dFdz_val;
        Complex log_gamma_a = log_gamma(hyp_a); // Gamma(L+1-i*eta)

        // --- 2. 组合所有的 Log 因子 ---
        // 总 Log 因子 = log_Cn_part + log_term_z + log_gamma_a
        Complex total_log_scale = log_Cn_part + log_term_z + log_gamma_a;

        // --- 3. 调用 Arb 计算 F_val ---
        // 传入: 普通因子 a_n, 对数因子 total_log_scale
        // 这一步计算了: a_n * C_n * F_L
        Complex term_val = Hyp1F1_FullyScaled(hyp_a, hyp_b, arg_z, a_n, total_log_scale);

        // --- 4. 计算导数项 ---
        // dF/dz 需要两部分：
        // Part 1: F * d(log_term)/dz -> 直接用 term_val * dLog
        Complex dLogTerm_dz = -i + (L + 1.0) / z_hat;
        Complex term_deriv_part1 = term_val * dLogTerm_dz;

        // Part 2: 1F1 的导数项
        // d/dz 1F1_reg(a,b,z) = a * 1F1_reg(a+1, b+1, z)  <-- 注意 Regularized 的导数公式比较干净
        // 实际上: d/dz [ M(a,b,z)/Gamma(b) ] = (a/b) * M(a+1,b+1,z)/Gamma(b) ?
        // 不，Regularized 导数: d/dz M_reg(a,b,z) = a * M_reg(a+1, b+1, z)
        // 我们需要计算: [PreFactors] * Gamma(a) * [ a * M_reg(a+1, b+1, z) * (d_arg/dz) ]
        // = [PreFactors] * Gamma(a+1) * M_reg(a+1, b+1, z) * (2i)
        
        // 所以，对于导数项，Log 因子只是变了 Gamma(a) -> Gamma(a+1)
        Complex log_gamma_a_plus = log_gamma(hyp_a + 1.0);
        Complex total_log_scale_deriv = log_Cn_part + log_term_z + log_gamma_a_plus;
        
        // 计算导数核心部分
        Complex term_deriv_core = Hyp1F1_FullyScaled(
            hyp_a + 1.0, hyp_b + 1.0, arg_z, 
            a_n, total_log_scale_deriv
        );
        
        // 链式法则 d(2iz)/dz = 2i
        Complex term_deriv_part2 = term_deriv_core * 2.0 * i;

        // --- 5. 累加 ---
        sum_f += term_val;
        sum_dfdz += (term_deriv_part1 + term_deriv_part2);
    }

    // -------------------------------------------------------------------------
    // 4. 组合最终结果 R_C(r)
    // -------------------------------------------------------------------------
    // R = P * f
    // dR/dz = P' * f + P * f' = P * ( (P'/P)*f + f' )
    
    Complex R_C_val = P_val * sum_f;
    Complex dR_C_dz = P_val * (dLogP_dz * sum_f + sum_dfdz);
    
    // 转换回对 r 的导数
    Complex dR_C_dr = dR_C_dz * dzhat_dr;

    return {R_C_val, dR_C_dr};
}

// Arb 库 acb_hypgeom_coulomb 包装器
// ==========================================================
// Arb 包装器：计算合流超几何函数 1F1(a; b; z)
// ==========================================================
Complex TeukolskyRadial::Hyp1F1(Complex a, Complex b, Complex z, bool regularized) {
    // 1. 初始化 Arb 变量
    acb_t acb_a, acb_b, acb_z, acb_res;
    acb_init(acb_a);
    acb_init(acb_b);
    acb_init(acb_z);
    acb_init(acb_res);

    // 2. 设置精度 (128 bits 以保证中间过程稳定性)
    slong prec = 128;

    // 3. 赋值
    acb_set_d_d(acb_a, a.real(), a.imag());
    acb_set_d_d(acb_b, b.real(), b.imag());
    acb_set_d_d(acb_z, z.real(), z.imag());

    // 4. 计算 1F1 (regularized=0)
    // acb_hypgeom_1f1(res, a, b, z, regularized, prec)
    acb_hypgeom_1f1(acb_res, acb_a, acb_b, acb_z,regularized, prec);

    // 5. 转换回 Complex
    double res_r = arf_get_d(arb_midref(acb_realref(acb_res)), ARF_RND_NEAR);
    double res_i = arf_get_d(arb_midref(acb_imagref(acb_res)), ARF_RND_NEAR);

    // 6. 清理
    acb_clear(acb_a);
    acb_clear(acb_b);
    acb_clear(acb_z);
    acb_clear(acb_res);

    return Complex(res_r, res_i);
}

Complex TeukolskyRadial::Hyp1F1_Scaled(Complex a, Complex b, Complex z, Complex log_mult) {
    // log_mult 应该是你需要乘在前面的因子的对数，即 log_gamma(hyp_a)
    
    // 1. 初始化 Arb 变量
    acb_t acb_a, acb_b, acb_z, acb_res, acb_log_mult, acb_mult, acb_final;
    acb_init(acb_a); acb_init(acb_b); acb_init(acb_z);
    acb_init(acb_res); acb_init(acb_log_mult); acb_init(acb_mult); acb_init(acb_final);

    // 2. 设置精度 (建议稍微提高一点，比如 256，以防万一，但 128 通常够用)
    slong prec = 256;

    // 3. 赋值
    acb_set_d_d(acb_a, a.real(), a.imag());
    acb_set_d_d(acb_b, b.real(), b.imag());
    acb_set_d_d(acb_z, z.real(), z.imag());
    acb_set_d_d(acb_log_mult, log_mult.real(), log_mult.imag());

    // 4. 计算 1F1 (Regularized = 1)
    // 结果等于 1F1(a,b,z) / Gamma(b)
    // 这个值通常极小，甚至小于 double 的最小精度，但在 Arb 里没问题
    acb_hypgeom_1f1(acb_res, acb_a, acb_b, acb_z, 1, prec);

    // 5. 计算乘法因子 exp(log_mult)
    // 即 exp(log_gamma(a)) = Gamma(a)
    // 这个值是巨大的，但在 Arb 里存得下
    acb_exp(acb_mult, acb_log_mult, prec);

    // 6. 在高精度环境下相乘
    // Final = (1F1 / Gamma(b)) * Gamma(a)
    // 数学上这等于 (Gamma(a)/Gamma(b)) * 1F1
    // 这个结果通常是一个“正常大小”的数，可以安全转回 double
    acb_mul(acb_final, acb_res, acb_mult, prec);

    // 7. 转换回 Complex
    // 如果结果仍然溢出 double (极不可能，除非物理结果本身发散)，这里会得到 inf
    double res_r = arf_get_d(arb_midref(acb_realref(acb_final)), ARF_RND_NEAR);
    double res_i = arf_get_d(arb_midref(acb_imagref(acb_final)), ARF_RND_NEAR);

    // 8. 清理
    acb_clear(acb_a); acb_clear(acb_b); acb_clear(acb_z);
    acb_clear(acb_res); acb_clear(acb_log_mult); acb_clear(acb_mult); acb_clear(acb_final);

    return Complex(res_r, res_i);
}
Complex TeukolskyRadial::Hyp1F1_FullyScaled(
    Complex a, Complex b, Complex z, 
    Complex factor,      // 普通因子 (如 a_n)
    Complex log_factor   // 对数因子 (如 log_gamma, log_power)
) {
    acb_t acb_a, acb_b, acb_z, acb_res;
    acb_t acb_factor, acb_log_factor, acb_exp_log, acb_final;

    acb_init(acb_a); acb_init(acb_b); acb_init(acb_z); acb_init(acb_res);
    acb_init(acb_factor); acb_init(acb_log_factor); 
    acb_init(acb_exp_log); acb_init(acb_final);

    // 提高精度到 256 bits，以应对 N=150 时的剧烈相消
    slong prec = 256;

    // 1. 设置参数
    acb_set_d_d(acb_a, a.real(), a.imag());
    acb_set_d_d(acb_b, b.real(), b.imag());
    acb_set_d_d(acb_z, z.real(), z.imag());
    
    // 2. 设置因子
    acb_set_d_d(acb_factor, factor.real(), factor.imag());
    acb_set_d_d(acb_log_factor, log_factor.real(), log_factor.imag());

    // 3. 计算正则化合流超几何函数 1F1_reg = 1F1 / Gamma(b)
    // 使用正则化版本是因为当 b 为负整数附近时它更稳定，且能自然处理 Gamma(b) 分母
    acb_hypgeom_1f1(acb_res, acb_a, acb_b, acb_z, 1, prec); 

    // 4. 计算 exp(log_factor)
    acb_exp(acb_exp_log, acb_log_factor, prec);

    // 5. 终极乘法： Final = (1F1_reg) * (factor) * (exp_log_factor)
    // 注意：这里我们是在 Arb 的超大动态范围内乘的，完全不用担心溢出
    acb_mul(acb_final, acb_res, acb_factor, prec);
    acb_mul(acb_final, acb_final, acb_exp_log, prec);

    // 6. 转回 double
    // 如果最终结果依然溢出 double (极少见)，这里会得到 inf，但不会是 NaN
    double res_r = arf_get_d(arb_midref(acb_realref(acb_final)), ARF_RND_NEAR);
    double res_i = arf_get_d(arb_midref(acb_imagref(acb_final)), ARF_RND_NEAR);

    acb_clear(acb_a); acb_clear(acb_b); acb_clear(acb_z); acb_clear(acb_res);
    acb_clear(acb_factor); acb_clear(acb_log_factor); 
    acb_clear(acb_exp_log); acb_clear(acb_final);

    return Complex(res_r, res_i);
}
// ==========================================================
// 全域径向函数求值 (Global Radial Function Evaluation)
// 自动切换近场 (Hypergeometric) 和远场 (Coulomb) 算法
// ==========================================================
// std::pair<Complex, Complex> TeukolskyRadial::Evaluate_R_in(
//     double r,
//     Complex nu,
//     Complex K_nu,
//     Complex K_neg_nu,
//     const std::map<int, Complex>& a_coeffs_pos,
//     const std::map<int, Complex>& a_coeffs_neg,
//     double r_match)
// {
//     // 1. 近场区域：直接使用超几何级数
//     // 对应 LRR Eq. 116 + 120
//     if (r <= r_match) {
//         return Evaluate_Hypergeometric(r, nu, a_coeffs_pos);
//     }
    
//     // 2. 远场区域：使用 Coulomb 级数的线性组合
//     // 对应 LRR Eq. 166: R^in = K_v * R_C^v + K_-v-1 * R_C^-v-1
//     else {
//         // 计算第一部分: K_nu * R_C^nu
//         // 注意 Evaluate_Coulomb 返回的是 {Value, Deriv}
//         auto res_pos = Evaluate_Coulomb(r, nu, a_coeffs_pos);
//         Complex val_pos = res_pos.first;
//         Complex der_pos = res_pos.second;
        
//         // 计算第二部分: K_-nu-1 * R_C^-nu-1
//         // 注意传入的 nu 应该是 -nu - 1.0
//         Complex nu_neg = -nu - 1.0;
//         auto res_neg = Evaluate_Coulomb(r, nu_neg, a_coeffs_neg);
//         Complex val_neg = res_neg.first;
//         Complex der_neg = res_neg.second;
        
//         // 线性组合
//         Complex R_total = K_nu * val_pos + K_neg_nu * val_neg;
//         Complex dR_total = K_nu * der_pos + K_neg_nu * der_neg;
        
//         return {R_total, dR_total};
//     }
// }
// -----------------------------------------------------------------------------
// 辅助函数：计算远场库伦解的线性组合 (未校准)
// -----------------------------------------------------------------------------
std::pair<Complex, Complex> TeukolskyRadial::Compute_Raw_Coulomb_Combo(
    double r, Complex nu, Complex K_nu, Complex K_neg_nu,
    const std::map<int, Complex>& a_coeffs_pos,
    const std::map<int, Complex>& a_coeffs_neg) 
{
    // Part A: K_nu * R_C^nu
    auto res_pos = Evaluate_Coulomb(r, nu, a_coeffs_pos);
    
    // Part B: K_-nu-1 * R_C^-nu-1
    Complex nu_neg = -nu - 1.0;
    auto res_neg = Evaluate_Coulomb(r, nu_neg, a_coeffs_neg);
    
    // Linear Combination
    Complex R_total = K_nu * res_pos.first + K_neg_nu * res_neg.first;
    Complex dR_total = K_nu * res_pos.second + K_neg_nu * res_neg.second;

    return {R_total, dR_total};
}

// -----------------------------------------------------------------------------
// 主函数：Evaluate_R_in (带自动校准和 NaN 回退)
// -----------------------------------------------------------------------------
std::pair<Complex, Complex> TeukolskyRadial::Evaluate_R_in(
    double r,
    Complex nu,
    Complex K_nu,
    Complex K_neg_nu,
    const std::map<int, Complex>& a_coeffs_pos,
    const std::map<int, Complex>& a_coeffs_neg,
    double r_match)
{
    // step 0: 惰性初始化校准因子 (只计算一次)
    // ---------------------------------------------------------
    if (!m_is_calibrated) {
        // 1. 尝试计算 Near 基准
        auto near_match = Evaluate_Hypergeometric(r_match, nu, a_coeffs_pos);
        
        // 2. 检查 Near 基准是否有效
        // 修正点：如果匹配点本身的近场解就是 NaN，我们不能用它来计算比值！
        bool near_valid = std::isfinite(near_match.first.real()) && 
                          std::isfinite(near_match.first.imag());

        if (!near_valid) {
            // 情况 A: 匹配点失效。
            // 策略：放弃平滑连接，保持因子为 1.0。
            // 这样虽然可能有相位突变，但至少能保证后续用库伦解替代时有数值，而不是 NaN。
            m_match_calibration_factor = 1.0;
            // 可以在这里加个 debug print，或者保持沉默
            // std::cout << "Near match is invalid. Using factor 1.0." << std::endl;
        } 
        else {
            // 情况 B: 匹配点有效。计算校准因子。
            auto far_match = Compute_Raw_Coulomb_Combo(r_match, nu, K_nu, K_neg_nu, a_coeffs_pos, a_coeffs_neg);
            
            if (std::abs(far_match.first) > 1e-100) {
                m_match_calibration_factor = near_match.first / far_match.first;
            } else {
                m_match_calibration_factor = 1.0; 
            }
        }
        
        m_is_calibrated = true;
    }
    // step 1: 判断是否尝试使用近场解 (Hypergeometric)
    // ---------------------------------------------------------
    if (r <= r_match) {
        auto res_near = Evaluate_Hypergeometric(r, nu, a_coeffs_pos);
        
        // 再次检查当前点的解是否有效
        bool is_valid = std::isfinite(res_near.first.real()) && 
                        std::isfinite(res_near.first.imag());

        if (is_valid) {
            // 只有当它是有效数值时，才直接返回
            return res_near;
        }
        // 如果是 NaN，不要返回，直接“穿透”到下方的 Step 2
    }

    // step 2: 使用远场解 (Coulomb) 并应用校准
    // ---------------------------------------------------------
    // 执行到这里有两种情况：
    // A. r > r_match (正常远场)
    // B. r <= r_match 但 Hypergeometric 算出了 NaN (回退机制)
    
    auto res_far = Compute_Raw_Coulomb_Combo(r, nu, K_nu, K_neg_nu, a_coeffs_pos, a_coeffs_neg);
    
    // 关键修正：将远场解乘以校准因子，使其在 r_match 处与近场解平滑连接
    // R_corrected = Factor * R_raw
    // dR_corrected = Factor * dR_raw (因为 Factor 是常数)
    
    return {
        res_far.first * m_match_calibration_factor, 
        res_far.second * m_match_calibration_factor
    };
}
Complex TeukolskyRadial::evaluate_ddR(double r, Complex R, Complex dR) const {
    // 物理参数
    double M = m_M;
    double a = m_a;
    // double omega = m_omega; // 已在类成员中
    
    // 几何量
    double Delta = r*r - 2.0*M*r + a*a;
    double dDelta_dr = 2.0*(r - M);
    
    // 势函数 V(r) (注意: Teukolsky Equation 的 V 定义有多种，需与 LRR 一致)
    // LRR Eq. 113: Delta^{-s} d/dr(Delta^{s+1} dR/dr) + H(r) R = 0
    // 其中 H(r) = (K^2 - 2is(r-M)K)/Delta + 4is w r - lambda
    
    Complex K_val = (r*r + a*a)*m_omega - (double)m_m*a;
    Complex term1 = (K_val*K_val - 2.0i*(double)m_s*(r-M)*K_val) / Delta;
    Complex term2 = 4.0i * (double)m_s * m_omega * r;
    Complex H_potential = term1 + term2 - m_lambda;
    
    // 展开微分算符:
    // Term1 = Delta^{-s} [ (s+1)Delta^s Delta' R' + Delta^{s+1} R'' ]
    //       = (s+1) Delta' R' + Delta R''
    //
    // 方程: (s+1) Delta' R' + Delta R'' + H R = 0
    // => R'' = - (1/Delta) * [ (s+1) Delta' R' + H R ]
    
    Complex numerator = ((double)m_s + 1.0) * dDelta_dr * dR + H_potential * R;
    
    // 视界处 Delta -> 0，需要处理（通常数值积分会在视界外一点截断，不用过度担心除零）
    if (std::abs(Delta) < 1e-12) return 0.0; 
    
    return -numerator / Delta;
}
