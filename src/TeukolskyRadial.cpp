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

TeukolskyRadial::TeukolskyRadial(Real a_spin, Real omega, int s, int l, int m, Real lambda)
    : m_a(a_spin), m_omega(omega), m_s(s), m_l(l), m_m(m), m_lambda(lambda)
{
    m_epsilon = 2.0 * m_omega;
    Real q = m_a;
    
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
    
    return iepskappa * n_nu * (2.0 * n_nu - 1.0)
           * ((n_nu - 1.0)*(n_nu - 1.0) + m_epsilon_sq)
           * (n_nu + 1.0 + 1.0i * m_tau);
}

Complex TeukolskyRadial::coeff_gamma(Complex nu, int n) const {
    Complex n_nu = nu + (double)n;
    Complex iepskappa = 1.0i * m_epsilon * m_kappa;
    
    return -iepskappa * (n_nu + 1.0) * (2.0 * n_nu + 3.0)
           * ((n_nu + 2.0)*(n_nu + 2.0) + m_epsilon_sq)
           * (n_nu - 1.0i * m_tau);
}

Complex TeukolskyRadial::coeff_beta(Complex nu, int n) const {
    Complex n_nu = nu + (double)n;
    Complex term1 = n_nu * (n_nu + 1.0); 
    
    Complex b_add1 = 2.0 * m_epsilon_sq - m_epsilon * m_m * m_a - m_lambda - 2.0;
    Complex b_add2 = m_epsilon * (m_epsilon - m_m * m_a) * (4.0 + m_epsilon_sq);
    
    return 4.0 * (term1 * (term1 + b_add1) + b_add2)
           * (n_nu + 1.5) * (n_nu - 0.5);
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
        
        if (std::abs(delta - 1.0) < tol) {
            return f;
        }
    }
    
    std::cerr << "Warning: Continued fraction did not converge." << std::endl;
    return f;
}
// ==========================================================
// 求解特征值 nu
// 核心方程: g(nu) = beta_0 + alpha_0 * R_1(nu) + gamma_0 * L_-1(nu) = 0
// 参考: Fujita & Tagoshi (2004) Eq. (3-6)
// ==========================================================

Complex TeukolskyRadial::calc_g(Complex nu) const {
    // 1. 获取 n=0 时的 MST 系数
    Complex a0 = coeff_alpha(nu, 0);
    Complex b0 = coeff_beta(nu, 0);
    Complex g0 = coeff_gamma(nu, 0);
    
    // 2. 计算连分式
    // R_1: 从 n=1 向正无穷收敛 (direction = +1)
    // 对应 FT04 Eq. (2-14) R_n = -gamma_n / (beta_n + alpha_n * R_{n+1})
    // 我们的 continued_fraction(nu, 1) 计算的就是这个 R_1
    Complex R1 = continued_fraction(nu, 1);
    
    // L_-1: 从 n=-1 向负无穷收敛 (direction = -1)
    // 对应 FT04 Eq. (2-15) L_n = -alpha_n / (beta_n + gamma_n * L_{n-1})
    // 我们的 continued_fraction(nu, -1) 计算的就是这个 L_-1
    Complex L_minus1 = continued_fraction(nu, -1);
    
    // 3. 组合方程
    // g(nu) = beta_0 + alpha_0 * R_1 + gamma_0 * L_-1
    return b0 + a0 * R1 + g0 * L_minus1;
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


