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
// ==========================================================
// 1. 计算级数系数 a_n (Minimal Solution)
// ==========================================================
void TeukolskyRadial::compute_coefficients(Complex nu) {
    m_nu = nu; // <--- 关键：保存 nu 供后续 evaluate_R_in 使用
    m_coefficients.clear();
    
    // 设定归一化条件 a_0 = 1
    m_coefficients[0] = 1.0;
    
    // 截断范围：通常取到 |n|=20~40 即可达到机器精度
    // GremlinEq 是动态判断收敛，我们这里先给个固定大范围，或者直到系数极小
    const int n_max = 40;
    const double tiny = 1e-30;

    // 向正向递推: a_n = R_n * a_{n-1}
    // R_n = continued_fraction(nu, 1) 计算的是 R_1, R_2... 吗？
    // 注意：我们的 continued_fraction(nu, 1) 内部从 k=1 开始迭代，实际上计算的是 R_1
    // 但我们需要 R_n = -gamma_n / (beta_n + alpha_n * R_{n+1})
    // 我们的 continued_fraction 函数是通用的 Lentz 方法，它计算的是整个连分式的值。
    // 在 MST 中，R_n 本身就是一个连分式。
    // R_n(\nu) 实际上就等于 continued_fraction 逻辑中把 n_start 设为 n 的结果。
    
    // 为了效率，我们不能对每个 n 都重新跑一遍几万次的 Lentz 迭代。
    // 幸运的是，MST 系数递推有一个特性：
    // a_{n+1} = - (beta_n * a_n + gamma_n * a_{n-1}) / alpha_n
    // 但这是“非最小解”方向，会发散！
    // 所以必须用连分式比值：a_n / a_{n-1} = R_n
    
    // 正向 (n > 0): a_n = a_{n-1} * R_n
    // 我们需要修改一下 continued_fraction 或者新增一个函数来计算任意 n 的 R_n？
    // 实际上，GremlinEq 在 fsum.cc 中并没有显式存储所有 a_n，
    // 而是边算边求和 (fsum 函数)。
    // 但为了清晰，我们先存储。
    
    // 计算 R_n 序列 (从高阶往回推，或者直接对每个n调连分式)
    // 这种做法效率稍低但逻辑最简单安全。
    
    // n = 1, 2, ...
    for (int n = 1; n <= n_max; ++n) {
        // R_n 是从 n 开始向 +inf 的连分式
        // 我们需要稍微 hack 一下 continued_fraction 函数让它支持起始 n
        // 或者我们简单地重载一个私有函数
        // 暂时用一个临时方案：复制 continued_fraction 的逻辑但修改 n_start
        // (见下方私有辅助函数实现)
        
        // 实际上，GremlinEq 的 radialfrac.cc 是计算整个 nu 对应的连分式。
        // 我们之前实现的 continued_fraction(nu, 1) 是 R_1。
        // R_n(nu) 其实等价于 R_1(nu + (n-1)) ? 
        // 不完全是，因为 alpha/beta/gamma 依赖于 n 和 nu 的组合 N = n+nu。
        // 是的！ MST 系数只依赖于 N = n + nu。
        // 所以 R_n(\nu) == R_1(\nu + (n-1))。
        // 利用这个性质，我们可以直接调用 continued_fraction。
        
        Complex R_n = continued_fraction(nu + (double)(n - 1), 1);
        Complex a_prev = m_coefficients[n - 1];
        m_coefficients[n] = a_prev * R_n;
        
        if (std::abs(m_coefficients[n]) < tiny) break;
    }

    // 反向 (n < 0): a_n = a_{n+1} * L_n
    // L_n(\nu) 对应 continued_fraction(nu, -1) 的逻辑
    // L_n(\nu) == L_{-1}(\nu + (n+1)) ?
    // 检查：L_{-1} 是从 -1 往负无穷。L_n 是从 n 往负无穷。
    // 系数依赖 N = n + nu。
    // L_{-1}(\nu) 此时 N 从 -1+nu 开始。
    // L_n(\nu) 此时 N 从 n+nu 开始。
    // 令 \nu' = \nu + (n+1)，则 -1+\nu' = -1+nu+n+1 = n+nu。
    // 所以 L_n(\nu) == L_{-1}(\nu + n + 1)。
    
    for (int n = -1; n >= -n_max; --n) {
        Complex L_n = continued_fraction(nu + (double)(n + 1), -1);
        Complex a_next = m_coefficients[n + 1];
        m_coefficients[n] = a_next * L_n;
        
        if (std::abs(m_coefficients[n]) < tiny) break;
    }
}

Complex TeukolskyRadial::get_coef(int n) const {
    if (m_coefficients.find(n) != m_coefficients.end()) {
        return m_coefficients.at(n);
    }
    return 0.0;
}

// ==========================================================
// 2. K 因子 (kfactor)
// GremlinEq/src/fujtag/fsum.cc
// ==========================================================
// [补全] K 因子 (k_factor)
// 对应 GremlinEq/src/fujtag/fsum.cc :: kfactor
// 这是一个非常复杂的公式，包含 Gamma 函数预因子和两个级数求和
// ==========================================================
Complex TeukolskyRadial::k_factor(Complex nu) const {
    // 1. 预因子 (Prefactor)
    Complex eps = m_epsilon;
    Complex tau = m_tau;
    Complex kappa = m_kappa;
    Complex i = 1.0i;
    
    // 常用变量
    Complex nu2_1 = 2.0 * nu + 1.0;
    Complex nu_1_it = nu + 1.0 + i * tau;
    Complex nu_1__it = nu + 1.0 - i * tau; // nu + 1 - i*tau
    Complex nu_3_ie = nu + 3.0 + i * eps;
    Complex nu_3__ie = nu + 3.0 - i * eps;
    
    // 对应 fsum.cc line 192: prefact calculation
    // log_gamma 项的组合
    Complex ln_pre = log_gamma(3.0 - i * (eps + tau)) 
                   + log_gamma(2.0 * nu + 2.0)
                   - log_gamma(nu_3_ie) 
                   + i * eps * kappa 
                   + log_gamma(nu2_1)
                   - log_gamma(nu_3__ie) 
                   - log_gamma(nu_1__it);
                   
    Complex prefact = 4.0 * std::pow(2.0 * eps * kappa, -2.0 - nu) * std::exp(ln_pre);

    // 2. 正向级数 (LOOP_POSITIVE) -> num
    // GremlinEq 这里的级数是超几何函数的展开项
    Complex num = 1.0;
    Complex lastcoeff = 1.0;
    const int max_iter = 1000;
    const double tol = 1e-14;

    for (int n = 0; n < max_iter; ++n) {
        double dn = (double)n;
        // 对应 fsum.cc line 198: lastcoeff *= - (...)
        Complex term_num = (dn + nu2_1) * (dn + nu - 1.0 + i*eps) * (dn + nu_1_it);
        Complex term_den = (dn + 1.0) * (dn + nu_3__ie) * (dn + nu_1__it);
        
        lastcoeff *= - term_num / term_den;
        num += lastcoeff;
        
        if (std::abs(lastcoeff) < tol * std::abs(num)) break;
    }

    // 3. 负向级数 (LOOP_NEGATIVE) -> denom
    // 注意: GremlinEq 这里是用 loop negative 来计算分母部分的级数
    Complex denom = 1.0;
    lastcoeff = 1.0;
    
    // fsum.cc line 208: LOOP_NEGATIVE
    // 这里的 n 实际上是递增的索引，对应原文 loop 中的 lastn 递减逻辑
    // 原文逻辑较晦涩，公式化简后如下：
    // lastcoeff *= ((lastn + nu2_1) * (lastn + nu_2_ie)) / ((lastn - 1) * (lastn + nu__2__ie));
    // 注意原文 LOOP_NEGATIVE 宏里 lastn 是递减的。
    // 我们这里用 n 表示迭代步数，对应原文的 -n (或者说反向递归的深度)
    
    // 让我们直接复刻宏展开后的数学逻辑。
    // 这一项对应超几何级数 2F1 的系数
    for (int n = 0; n < max_iter; ++n) {
        double dn = (double)n; 
        // 实际上这里的递推是针对 specific hypergeometric series
        // 参照 fsum.cc line 209:
        // lastcoeff *= ( (lastn + nu2_1) * (lastn + nu + 2.0 + i*eps) ) 
        //            / ( (lastn - 1.0) * (lastn + nu - 2.0 - i*eps) );
        // 这里的 lastn 在 LOOP_NEGATIVE 里是从 0 开始递减：0, -1, -2...
        // 所以我们在代码里用 dn = -n
        
        double lastn = -dn;
        
        // 避开 lastn = 1 的奇点（虽然 LOOP_NEGATIVE 从 0 开始减，不会碰到 1）
        // 但注意 n=0 时 lastn=0。公式里分母有 (lastn-1)，即 -1，安全。
        
        Complex term_num = (lastn + nu2_1) * (lastn + nu + 2.0 + i*eps);
        Complex term_den = (lastn - 1.0) * (lastn + nu - 2.0 - i*eps);
        
        lastcoeff *= term_num / term_den;
        denom += lastcoeff;
        
        if (std::abs(lastcoeff) < tol * std::abs(denom)) break;
    }

    return prefact * num / denom;
}


// ==========================================================
// [补全] A_minus 因子
// 对应 GremlinEq/src/fujtag/fsum.cc :: aminus
// 用于计算 C_trans
// ==========================================================
Complex TeukolskyRadial::calc_aminus(Complex nu) const {
    Complex i = 1.0i;
    Complex eps = m_epsilon;
    
    // 1. 正向级数 (sum_pos)
    // 对应 fsum.cc line 242
    Complex sum_pos = 1.0;
    Complex lastcoeff = 1.0;
    const int max_iter = 1000;
    const double tol = 1e-14;

    for (int n = 0; n < max_iter; ++n) {
        double lastn = (double)n;
        // lastcoeff *= - (lastn + nu - 1.0 - i*eps) / (lastn + nu + 3.0 + i*eps);
        Complex num = lastn + nu - 1.0 - i*eps;
        Complex den = lastn + nu + 3.0 + i*eps;
        
        lastcoeff *= - num / den;
        sum_pos += lastcoeff;
        if (std::abs(lastcoeff) < tol * std::abs(sum_pos)) break;
    }
    
    // 注意：GremlinEq 的 aminus 函数只计算了 LOOP_POSITIVE 部分？
    // 仔细看 fsum.cc line 242 (LOOP_POSITIVE) 和 line 250 (LOOP_NEGATIVE).
    // 原文好像是把它们串起来了？不，那是两个独立的代码块。
    // 但是 aminus 函数只返回了一个 sum。
    // 原来：它根据某种条件选择一个方向？或者两部分相乘？
    // 让我们看 fsum.cc 的结构：
    // sum = 1; LOOP_POSITIVE(...);
    // lastcoeff = 1; LOOP_NEGATIVE(...); 
    // 上面的 sum 变量在 LOOP_NEGATIVE 之前没有重置！
    // 这意味着它是接着加的？不，LOOP_NEGATIVE 里是 `sum += lastcoeff`。
    // 所以它是把两个方向的级数都加到了同一个 sum 变量里！
    // 也就是说 sum = 1 + (正向项) + (负向项)。
    
    // 2. 负向级数 (sum_neg)
    lastcoeff = 1.0; // 重置系数
    for (int n = 0; n < max_iter; ++n) {
        double lastn = -(double)n; // 从 0 开始递减
        
        // lastcoeff *= - (lastn + nu + 2.0 + i*eps) / (lastn + nu - 2.0 - i*eps);
        Complex num = lastn + nu + 2.0 + i*eps;
        Complex den = lastn + nu - 2.0 - i*eps;
        
        lastcoeff *= - num / den;
        
        // 把结果加到同一个 sum_pos 里 (对应 fsum.cc 的逻辑)
        sum_pos += lastcoeff;
        if (std::abs(lastcoeff) < tol * std::abs(sum_pos)) break;
    }
    
    // 3. 预因子
    // return pow(2.0, 1.0 + i*eps) * exp(-0.5 * PI * (eps + i*(nu - 1.0))) * sum;
    Complex prefact = std::pow(2.0, 1.0 + i*eps) * std::exp(-0.5 * M_PI * (eps + i*(nu - 1.0)));
    
    return prefact * sum_pos;
}

// ==========================================================
// [完善] 渐进振幅计算 (compute_amplitudes)
// 对应 fsum.cc :: asympt_amps
// ==========================================================
void TeukolskyRadial::compute_amplitudes(Complex nu) {
    // 1. 确保系数已计算
    compute_coefficients(nu);
    
    // 2. 计算系数总和 (Straight Sum of a_n)
    // 对应 fsum.cc :: fsum
    // 同样，GremlinEq 的 fsum 也是把正向和负向的级数加到一起
    // 我们已经在 compute_coefficients 里把 a_n 都存到 map 里了，直接求和即可
    Complex straight_sum = 0.0;
    for (auto const& [n, val] : m_coefficients) {
        straight_sum += val;
    }
    
    // 3. 计算关键因子
    Complex k_nu = k_factor(nu);
    Complex k_minus_nu_1 = k_factor(-nu - 1.0);
    Complex aminus_val = calc_aminus(nu); // 新增
    
    Complex eps = m_epsilon;
    Complex kappa = m_kappa;
    Complex tau = m_tau;
    Complex i = 1.0i;
    
    // 4. B_trans (Transmission)
    // fsum.cc line 72
    Complex phase_trans = i * (eps + tau) * kappa * (0.5 + std::log(kappa)/(1.0+kappa));
    m_B_trans = (1.0 / std::pow(2.0*kappa, 4)) * std::exp(phase_trans) * straight_sum;
    
    // 5. C_trans (Transmission, Upgoing)
    // fsum.cc line 85: *c_trans = epsilon^3 / 8 * aminussum * exp(...)
    // ieloge = i * epsilon * log(epsilon)
    Complex ieloge = i * eps * std::log(eps);
    m_C_trans = (eps * eps * eps / 8.0) * aminus_val * std::exp(ieloge - 0.5*i*(1.0-kappa)*eps);
    
    // 6. B_inc (Incident)
    // fsum.cc line 77
    // term_sin = sin(pi*(nu + 2 + i*eps)) / sin(pi*(nu - 2 - i*eps))
    // 利用 sin(z+2pi) = sin(z)，其实就是 sin(pi*(nu+i*eps)) / sin(pi*(nu-i*eps))
    Complex arg1 = M_PI * (nu + 2.0 + i*eps);
    Complex arg2 = M_PI * (nu - 2.0 - i*eps);
    Complex term_sin = std::sin(arg1) / std::sin(arg2);
    
    Complex term_brackets = k_nu - i * std::exp(-i * M_PI * nu) * term_sin * k_minus_nu_1;
    
    // 指数项
    // exp( pi/2 * (-eps + i(nu+3)) - ieloge + ... )
    Complex exp_arg = M_PI * 0.5 * (-eps + i*(nu + 3.0)) 
                    - ieloge 
                    + 0.5*i*(1.0-kappa)*eps
                    + log_gamma(nu + 3.0 + i*eps) 
                    - log_gamma(nu - 1.0 - i*eps);
                    
    m_B_inc = (2.0 * term_brackets / eps)
            * std::pow(2.0, -3.0 - i*eps)
            * std::exp(exp_arg)
            * straight_sum;
}
// ==========================================================
// [内部辅助] 基础超几何级数 (仅在 |z| < 1 时有效)
// ==========================================================
std::pair<Complex, Complex> TeukolskyRadial::hypergeom_series(Complex a, Complex b, Complex c, Complex z) {
    Complex sum = 1.0;
    Complex term = 1.0;
    Complex deriv = 0.0;
    
    // k=0 term: 1
    // derivative of 1 is 0
    
    for (int k = 0; k < 2000; ++k) { // 增加迭代上限以防万一
        double dk = (double)k;
        
        // term_{k+1}
        Complex num = (a + dk) * (b + dk);
        Complex den = (c + dk) * (dk + 1.0);
        
        term *= (num / den) * z;
        sum += term;
        
        // d/dz (z^{k+1}) = (k+1) z^k
        // term has z^{k+1}, so deriv_term = term * (k+1) / z
        if (std::abs(z) > 1e-15) {
            deriv += term * (dk + 1.0) / z;
        } else {
            // z->0 limit, derivative is coefficient of linear term (k=0)
            if (k == 0) deriv += a * b / c;
        }
        
        if (std::abs(term) < 1e-15 * std::abs(sum)) break;
    }
    return {sum, deriv};
}

// ==========================================================
// 超几何函数求解器
// 自动选择直接求和或线性变换 (FT04 Eq 3-8)
// ==========================================================
std::pair<Complex, Complex> TeukolskyRadial::hypergeom_2F1_with_deriv(Complex a, Complex b, Complex c, Complex z) {
    // 策略：如果 |z| <= 0.5，直接求和。
    // 如果 |z| > 0.5，使用 z -> w = 1/(1-z) 变换。
    // 注意：MST 的 x 在物理区域 r > r+ 总是负的，所以 1/(1-z) 总是落在 (0, 1) 内。
    
    if (std::abs(z) <= 0.6) {
        return hypergeom_series(a, b, c, z);
    } else {
        // 使用变换公式 FT04 Eq (3-8)
        // F(a,b;c;z) = (1-z)^-a * G1 * F(a, c-b; a-b+1; w) 
        //            + (1-z)^-b * G2 * F(b, c-a; b-a+1; w)
        // 其中 w = 1/(1-z)
        
        Complex w = 1.0 / (1.0 - z);
        Complex one_minus_z = 1.0 - z;
        
        // 检查奇点: a-b 必须不是整数
        // 在 MST 中，a-b = 2n + 2nu + 1。只要 nu 不是半整数，这就没问题。
        // 我们假设 solve_nu 返回的 nu 是复数，满足条件。
        
        // 系数 G1, G2 (利用 log_gamma 计算以防溢出)
        // G1 = Gamma(c)Gamma(b-a) / (Gamma(b)Gamma(c-a))
        // G2 = Gamma(c)Gamma(a-b) / (Gamma(a)Gamma(c-b))
        
        Complex ln_Gc = log_gamma(c);
        Complex ln_Ga = log_gamma(a);
        Complex ln_Gb = log_gamma(b);
        Complex ln_G_ba = log_gamma(b - a);
        Complex ln_G_ab = log_gamma(a - b);
        Complex ln_G_ca = log_gamma(c - a);
        Complex ln_G_cb = log_gamma(c - b);
        
        Complex G1 = std::exp(ln_Gc + ln_G_ba - ln_Gb - ln_G_ca);
        Complex G2 = std::exp(ln_Gc + ln_G_ab - ln_Ga - ln_G_cb);
        
        // 计算两个新的超几何级数
        // F1: F(a, c-b; a-b+1; w)
        auto [F1, dF1_dw] = hypergeom_series(a, c - b, a - b + 1.0, w);
        
        // F2: F(b, c-a; b-a+1; w)
        auto [F2, dF2_dw] = hypergeom_series(b, c - a, b - a + 1.0, w);
        
        // 组合项 T1 = (1-z)^-a * G1 * F1
        // (1-z)^-a = w^a
        Complex wa = std::pow(w, a);
        Complex wb = std::pow(w, b);
        
        Complex T1 = wa * G1 * F1;
        Complex T2 = wb * G2 * F2;
        
        Complex F_val = T1 + T2;
        
        // 计算导数 dF/dz
        // 链式法则: d/dz = (dw/dz) * d/dw = w^2 * d/dw
        // d(w^a F1)/dw = a w^{a-1} F1 + w^a F1' = w^a (a/w F1 + F1')
        // 所以 dT1/dz = w^2 * G1 * w^a * (a/w F1 + dF1_dw)
        //             = G1 * w^{a+1} * (a F1 + w dF1_dw)
        
        // 注意：这里 w*dF1_dw 可以直接用级数导数计算结果
        Complex dT1_dz = G1 * std::pow(w, a + 1.0) * (a * F1 + w * dF1_dw);
        Complex dT2_dz = G2 * std::pow(w, b + 1.0) * (b * F2 + w * dF2_dw);
        
        return {F_val, dT1_dz + dT2_dz};
    }
}
// ==========================================================
// [修复] 评估 R_in(r)
// 参考 FT04 Eq. (2-3) 和 (2-4)
// ==========================================================
std::pair<Complex, Complex> TeukolskyRadial::evaluate_R_in(double r) {
    // 1. 几何变量定义 (FT04 Eq 2.4 下方)
    // x = omega * (r_plus - r) / (epsilon * kappa)
    // 简化后: x = (r_plus - r) / (r_plus - r_minus)
    // r_plus = 1 + kappa, r_minus = 1 - kappa (M=1)
     
    double r_plus = 1.0 + m_kappa;
    double r_minus = 1.0 - m_kappa;
    double r_diff = r_plus - r_minus; // 2 * kappa
    
    // 自变量 x
    // 注意：当 r > r_plus 时，x < 0。超几何级数在 x <= 0 时收敛良好。
    double x = (r_plus - r) / r_diff;
    
    // dx/dr = -1 / (r_plus - r_minus) = -1 / (2*kappa)
    double dx_dr = -1.0 / r_diff;

    // 2. 预因子 (Prefactors)
    // FT04 Eq (2-3): R_in = e^{i eps kappa x} * (-x)^{-s - i(eps+tau)/2} * (1-x)^{i(eps-tau)/2} * p_in(x)
    
    Complex i = 1.0i;
    
    // Term 1: Exponential
    Complex term_exp = std::exp(i * m_epsilon * m_kappa * x);
    Complex d_term_exp = term_exp * (i * m_epsilon * m_kappa); // d/dx
    
    // Term 2: (-x)^Power1
    // Power1 = -s - i(eps+tau)/2
    Complex p1 = -(double)m_s - 0.5i * (m_epsilon + m_tau);
    // 注意: r > r_plus => x < 0 => -x > 0. 这是一个正实数，不会有多值问题。
    Complex term_x = std::pow(Complex(-x), p1);
    Complex d_term_x = (x != 0.0) ? (p1 * term_x / x) : 0.0; // d/dx (x^p) = p x^{p-1}
    // 修正导数符号: d/dx [(-x)^p] = p(-x)^{p-1} * (-1) = -p (-x)^p / (-x) = p (-x)^p / x ?
    // 链式法则: u = -x, du/dx = -1. d/du (u^p) = p u^{p-1}. 
    // d/dx = p (-x)^{p-1} * (-1). 
    // term_x / x = (-x)^p / x = - (-x)^{p-1}. 
    // 所以 d_term_x = p * (-1) * (-x)^{p-1} = p * (term_x / x) * (-1) * (-1)? 
    // 简单推导: d/dx (-x)^p = - p (-x)^{p-1}. 
    // 代码中 term_x / (-x) = (-x)^{p-1}.
    // 所以 d/dx = - p * (term_x / -x) = p * term_x / x. 正确。

    // Term 3: (1-x)^Power2
    // Power2 = i(eps-tau)/2
    Complex p2 = 0.5i * (m_epsilon - m_tau);
    Complex term_1x = std::pow(Complex(1.0 - x), p2);
    // d/dx (1-x)^p = p (1-x)^{p-1} * (-1) = -p * term_1x / (1-x)
    Complex d_term_1x = -p2 * term_1x / (1.0 - x);

    // 3. 级数求和 p_in(x) (FT04 Eq 2-4)
    Complex P = 0.0;
    Complex dP_dx = 0.0;
    
    // 超几何参数 gamma = 1 - s - i(eps + tau)
    Complex param_c = 1.0 - (double)m_s - i * (m_epsilon + m_tau);

    for (auto const& [n, an] : m_coefficients) {
        // alpha = n + nu + 1 - i*tau
        // beta  = -n - nu - i*tau
        Complex param_a = (double)n + m_nu + 1.0 - i * m_tau;
        Complex param_b = -(double)n - m_nu - i * m_tau;
        
        auto [F, dF_dx] = hypergeom_2F1_with_deriv(param_a, param_b, param_c, Complex(x));
        
        P += an * F;
        dP_dx += an * dF_dx;
    }
    
    // 4. 组合 R = T1 * T2 * T3 * P
    Complex R = term_exp * term_x * term_1x * P;
    
    // 5. 组合导数 dR/dx
    // dR/dx = R * (d_T1/T1 + d_T2/T2 + d_T3/T3 + dP/P)
    // 这种对数微分法数值上更稳
    
    Complex dR_dx = d_term_exp * term_x * term_1x * P
                  + term_exp * d_term_x * term_1x * P
                  + term_exp * term_x * d_term_1x * P
                  + term_exp * term_x * term_1x * dP_dx;
                  
    // 转换为对 r 的导数
    return {R, dR_dx * dx_dr};
}
// ==========================================================
// 评估 d^2R/dr^2 (利用 Teukolsky 方程)
// ==========================================================
Complex TeukolskyRadial::evaluate_ddR(double r, Complex R, Complex dR) const {
    // Teukolsky Radial Equation (Sasaki-Tagoshi 2003 Eq. 2.2 or similar)
    // Delta^{-s} d/dr (Delta^{s+1} dR/dr) + ( ... ) R = 0
    // 展开:
    // Delta d2R + (s+1)(dDelta/dr) dR + Delta^{-s} (...) R = 0 ?
    // 让我们用最标准的展开形式 (Teukolsky 1973 Eq 2.2):
    // Delta d^2R/dr^2 + 2(s+1)(r-M) dR/dr + V(r) R = 0
    
    // 势函数 V(r) (注意：不同文献 V 定义不同，要小心符号)
    // 标准 Teukolsky Eq:
    // V = [ ( (r^2+a^2)omega - m a )^2 - 2is(r-M)( (r^2+a^2)omega - ma ) ] / Delta
    //     + 4is omega r - lambda
    // (注意 lambda 定义差异)
    
    double M = 1.0; // Geometric units
    double a = m_a;
    double delta = r*r - 2.0*r + a*a;
    double d_delta = 2.0*r - 2.0; // 2(r-M)
    
    double omega = m_omega;
    Complex i = 1.0i;
    
    // K = (r^2+a^2)w - am
    double K = (r*r + a*a) * omega - a * (double)m_m;
    
    // Term 1: 2(s+1)(r-M) dR
    Complex term1 = (double)(2 * (m_s + 1)) * (r - M) * dR;
    
    // Term 2: Potential * R
    // V_teuk = (K^2 - 2is(r-M)K)/Delta + 4iswr - lambda
    Complex term_K = K*K - 2.0*i*(double)m_s*(r-M)*K;
    Complex V = term_K / delta + 4.0*i*(double)m_s*omega*r - m_lambda;
    
    Complex term2 = V * R;
    
    // Delta * d2R + term1 + term2 = 0
    // => d2R = - (term1 + term2) / Delta
    
    return - (term1 + term2) / delta;
}