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
    
    const int max_iter = 50;
    const double tol = 1e-12; // 目标精度
    
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

//注意到这里代入了s=-2
Complex TeukolskyRadial::k_factor(Complex nu) const {
    Complex eps = m_epsilon;
    Complex tau = m_tau;
    Complex kappa = m_kappa;
    Complex i = 1.0i;
    
    // ------------------------------------------------------
    // 1. Prefactor (预因子)
    // 对应 GremlinEq 中的 prefact
    // formula: 4 * (2eps*kappa)^(-2-nu) * exp(...)
    // ------------------------------------------------------
    Complex nu2_1 = 2.0 * nu + 1.0;
    Complex nu_3_ie = nu + 3.0 + i * eps;
    Complex nu_3__ie = nu + 3.0 - i * eps; // nu + 3 - i*eps
    Complex nu_1_it = nu + 1.0 + i * tau;
    Complex nu_1__it = nu + 1.0 - i * tau;

    // Log Gamma 组合
    Complex ln_pre = log_gamma(3.0 - i * (eps + tau)) 
                   + log_gamma(2.0 * nu + 2.0)
                   - log_gamma(nu_3_ie) 
                   + i * eps * kappa 
                   + log_gamma(nu2_1)
                   - log_gamma(nu_3__ie) 
                   - log_gamma(nu_1__it);
                   
    Complex prefact = 4.0 * std::pow(2.0 * eps * kappa, -2.0 - nu) * std::exp(ln_pre);

    // ------------------------------------------------------
    // 2. 级数求和 (Summation of Expansion Terms)
    // MST 方法中 K_nu 包含两个收敛级数乘积的比值
    // 正向级数 (n >= 0) 和 负向级数 (n < 0)
    // ------------------------------------------------------
    
    // --- 正向部分 (n 从 0 到 +inf) ---
    Complex num_sum = 1.0;
    Complex term = 1.0;
    const int max_iter = 2000;
    const double tol = 1e-15;

    for (int n = 0; n < max_iter; ++n) {
        double dn = (double)n;
        // f_n / f_{n-1} 的比值 (参考 Sasaki-Tagoshi Eq 4.14 或 fsum.cc)
        // 注意：GremlinEq 的 fsum.cc 中 term_num/term_den 就是这个比值
        
        Complex numer = (dn + nu2_1) * (dn + nu - 1.0 + i * eps) * (dn + nu_1_it);
        Complex denom = (dn + 1.0) * (dn + nu_3__ie) * (dn + nu_1__it);
        
        // 注意符号：MST 级数通常有 (-1)^n 或类似的交错符号
        // fsum.cc 中使用了 `lastcoeff *= - ...`
        term *= - numer / denom;
        
        num_sum += term;
        if (std::abs(term) < tol * std::abs(num_sum)) break;
    }

    // --- 负向部分 (n 从 0 到 -inf，也就是代码里的 increasing n map to negative index) ---
    // 对应 GremlinEq 中的 LOOP_NEGATIVE
    // 这里计算的是分母部分的级数修正
    Complex den_sum = 1.0;
    term = 1.0;

    for (int n = 0; n < max_iter; ++n) {
        double dn = (double)n;
        // 这里的 n 对应原文公式中的 -n (方向相反)
        // 对应的比值公式 (f_{-n} / f_{-n+1} ?)
        
        double lastn = -dn; // 模拟递减
        
        // 公式适配 fsum.cc LOOP_NEGATIVE
        Complex numer = (lastn + nu2_1) * (lastn + nu + 2.0 + i * eps);
        Complex denom = (lastn - 1.0) * (lastn + nu - 2.0 - i * eps);
        
        // 同样有负号
        term *= numer / denom; // 注意 fsum.cc 这里没有显式负号? 
        // 仔细检查 fsum.cc: lastcoeff *= ((lastn+nu2+1.0)...) / ((lastn-1.0)...)
        // 这里的符号隐含在 lastn 的负值中吗？
        // GremlinEq 代码： lastcoeff *= ...
        // 在 LOOP_NEGATIVE 中，分母有 (lastn - 1.0)。因为 lastn <= 0，所以分母是负的。
        // 分子项主要由实部主导。
        // 所以这一项本身就是负的。
        
        den_sum += term;
        if (std::abs(term) < tol * std::abs(den_sum)) break;
    }

    // K_nu = Prefactor * (Sum_Pos / Sum_Neg)
    return prefact * num_sum / den_sum;
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

// [填空 1] B_trans 公式
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

// [填空 2] B_inc 公式
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

// [填空 3] B_ref 公式
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

// [填空 4] C_trans 公式
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
    // [填空 1] 坐标变换 r -> x
    // -------------------------------------------------------------------------
    // 参考 Eq. 114: x = (omega*r_plus - omega*r) / (epsilon * kappa)
    // 注意: epsilon = 2*M*omega. 
    // 这里 x 通常是一个负实数 (当 r > r_+)
    double x_val=(r_plus - r) / (2.0 * kappa);
    // Complex x_val = ...;
    // 提示: x_val = (r_plus - r) / (2.0 * M * kappa); // 化简后的形式，请验证
    // 对应的 dx/dr = ... (链式法则需要)
    // Complex dxdR = ...;
    double dx_dr=-1.0 / (2.0 * kappa);

    // -------------------------------------------------------------------------
    // [填空 2] 计算前置因子 P(x) 及其关于 x 的对数导数
    // -------------------------------------------------------------------------
    // 参考 Eq. 116: R = P(x) * p_in(x)

    Complex alpha = -s_c - i * (eps + tau) / 2.0; 
    Complex beta  = i * (eps - tau) / 2.0;
    Complex P_val=exp(i*eps*kappa*x_val) * pow((-x_val),(alpha)) * pow((1-x_val),(beta));

    Complex LogP=i*eps*kappa*x_val+(alpha)*log(-x_val)+(beta)*log(1-x_val);
    Complex dLogPdx = i*eps*kappa - (alpha)/(-x_val) - (beta)/(1-x_val);

    



    // -------------------------------------------------------------------------
    // 3. 计算级数求和 S(x) = Sum(a_n * F_n(x)) 及其导数
    // -------------------------------------------------------------------------
    Complex sum_S = 0.0;      // S(x)
    Complex sum_dSdx = 0.0;   // dS/dx
    
    // 超几何函数的参数 c (Eq. 120: c = 1 - s - i*eps - i*tau)
    

    for (const auto& [n, a_n] : a_coeffs) {
        double dn = (double)n;
        Complex hyp_a = dn + nu + 1.0 - i * tau;
        Complex hyp_b = -dn-nu - i * tau;
        Complex hyp_c = 1.0 - s_c - i * eps - i * tau;
        // [填空 3] 超几何函数参数 a, b
        // 参考 Eq. 120: F(a, b; c; x)
        // a = n + nu + 1 - i*tau
        // b = -n - nu - i*tau
        // Complex hyp_a = ...;
        // Complex hyp_b = ...;
        
        // 计算 F(a,b;c;x)
        // 注意: 这里假设你有一个 Hyp2F1 实现或暂时用占位符
        // 如果没有库，可以用简单的泰勒级数展开(仅当 |x| 很小时有效)，或者调用外部库
        Complex F_val = Hyp2F1(hyp_a, hyp_b, hyp_c, x_val);
        
        // 计算 dF/dx
        // 公式: d/dx 2F1(a,b;c;x) = (a*b/c) * 2F1(a+1, b+1; c+1; x) 
        Complex dFdx_val = (hyp_a * hyp_b / hyp_c) * Hyp2F1(hyp_a + 1.0, hyp_b + 1.0, hyp_c + 1.0, x_val);
        
        sum_S += a_n * F_val;
        sum_dSdx += a_n * dFdx_val;
    }

    // -------------------------------------------------------------------------
    // 4. 组合最终结果
    // -------------------------------------------------------------------------
    // R(r) = P(x) * S(x)
    Complex R_val = P_val * sum_S;
    
    // dR/dx = P'(x)S(x) + P(x)S'(x) = P(x) * [ (P'/P)*S + S' ]
    Complex dRdx_val = P_val * (dLogPdx* sum_S + sum_dSdx);
    
    // 转换回对 r 的导数: dR/dr = dR/dx * dx/dr
    Complex dRdr_val = dRdx_val * dx_dr;

    return {R_val, dRdr_val};
}

// 简单的超几何函数占位符 (如果还没有 GSL 或其他库绑定)
// 注意：实际运行时需要替换为真实的数值实现 (如 GSL 的 gsl_sf_hyperg_2F1_complex)
Complex TeukolskyRadial::Hyp2F1(Complex a, Complex b, Complex c, Complex z) {
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
    acb_hypgeom_2f1(acb_res, acb_a, acb_b, acb_c, acb_z, 0, prec);

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
    // [填空 1] 坐标变换 r -> z_hat
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
    // [填空 2] 计算 R_C 的前置因子 P_C(z_hat) 及其对数导数
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

    for (const auto& [n, a_n] : a_coeffs) {
        double dn = (double)n;
        
        // 3.1 计算组合系数 C_n
        double sign = (std::abs(n) % 2 == 0) ? 1.0 : -1.0;
        
        // Pochhammer ratio using log_gamma
        Complex lg_num_n = log_gamma(poch_num_base + dn);
        Complex lg_den_n = log_gamma(poch_den_base + dn);
        Complex poch_ratio = std::exp((lg_num_n - lg_num_base) - (lg_den_n - lg_den_base));
        
        Complex C_n = sign * poch_ratio * a_n;

        // 3.2 计算库伦波函数 F_L(eta, z) 及其导数
        // Eq. 142: F_L = e^{-iz} * 2^L * z^{L+1} * [Gamma(...) / Gamma(...)] * 1F1(...)
        Complex L = nu + dn; // L = n + nu
        
        // [填空 3] 库伦波函数 F_L 的各个部分
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
        
        Complex phi_val = Hyp1F1(hyp_a, hyp_b, arg_z);
        Complex dphi_dz_arg = (hyp_a / hyp_b) * Hyp1F1(hyp_a + 1.0, hyp_b + 1.0, arg_z); // dPhi / d(2iz)
        
        // Gamma 因子
        Complex lg_g1 = log_gamma(L + 1.0 - i * eta_coulomb);
        Complex lg_g2 = log_gamma(2.0 * L + 2.0);
        Complex gamma_factor = std::exp(lg_g1 - lg_g2);
        
        // 组合 F_L
        // F_L = e^{-iz} * 2^L * z^{L+1} * gamma_factor * phi
        // 建议在 log 域组合幂次项
        Complex log_term_z = -i * z_hat + L * std::log(2.0) + (L + 1.0) * std::log(z_hat);
        Complex term_z_combined = std::exp(log_term_z);
        
        Complex F_val = term_z_combined * gamma_factor * phi_val;

        // 3.3 计算 F_L 对 z_hat 的导数
        // dF/dz = F * (d(log_term)/dz) + term_z * gamma * (dPhi/d_arg * d_arg/dz)
        // d(log_term)/dz = -i + (L+1)/z
        // d_arg/dz = 2i
        
        Complex dLogTerm_dz = -i + (L + 1.0) / z_hat;
        Complex dPhi_dz = dphi_dz_arg * 2.0 * i;
        
        Complex dFdz_val = F_val * dLogTerm_dz + (term_z_combined * gamma_factor) * dPhi_dz;

        // 累加
        sum_f += C_n * F_val;
        sum_dfdz += C_n * dFdz_val;
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
Complex TeukolskyRadial::Hyp1F1(Complex a, Complex b, Complex z) {
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
    acb_hypgeom_1f1(acb_res, acb_a, acb_b, acb_z, 0, prec);

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
// ==========================================================
// 全域径向函数求值 (Global Radial Function Evaluation)
// 自动切换近场 (Hypergeometric) 和远场 (Coulomb) 算法
// ==========================================================
std::pair<Complex, Complex> TeukolskyRadial::Evaluate_R_in(
    double r,
    Complex nu,
    Complex K_nu,
    Complex K_neg_nu,
    const std::map<int, Complex>& a_coeffs_pos,
    const std::map<int, Complex>& a_coeffs_neg,
    double r_match)
{
    // 1. 近场区域：直接使用超几何级数
    // 对应 LRR Eq. 116 + 120
    if (r <= r_match) {
        return Evaluate_Hypergeometric(r, nu, a_coeffs_pos);
    }
    
    // 2. 远场区域：使用 Coulomb 级数的线性组合
    // 对应 LRR Eq. 166: R^in = K_v * R_C^v + K_-v-1 * R_C^-v-1
    else {
        // 计算第一部分: K_nu * R_C^nu
        // 注意 Evaluate_Coulomb 返回的是 {Value, Deriv}
        auto res_pos = Evaluate_Coulomb(r, nu, a_coeffs_pos);
        Complex val_pos = res_pos.first;
        Complex der_pos = res_pos.second;
        
        // 计算第二部分: K_-nu-1 * R_C^-nu-1
        // 注意传入的 nu 应该是 -nu - 1.0
        Complex nu_neg = -nu - 1.0;
        auto res_neg = Evaluate_Coulomb(r, nu_neg, a_coeffs_neg);
        Complex val_neg = res_neg.first;
        Complex der_neg = res_neg.second;
        
        // 线性组合
        Complex R_total = K_nu * val_pos + K_neg_nu * val_neg;
        Complex dR_total = K_nu * der_pos + K_neg_nu * der_neg;
        
        return {R_total, dR_total};
    }
}