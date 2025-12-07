/**
 * @file TeukolskyRadial.h
 * @brief Teukolsky 径向方程求解器 (MST 方法)
 * 对应 GremlinEq 的 FT 类及 fujtag 文件夹下的相关实现
 */

 #ifndef TEUKOLSKY_RADIAL_H
 #define TEUKOLSKY_RADIAL_H
 
 #include <complex>
 #include <vector>
 #include <cmath>

 
 using Real = double;
 using Complex = std::complex<double>;
 
 class TeukolskyRadial {
 public:
    /**
     * @brief 构造函数
     * @param a_spin 黑洞自旋 a (a/M)
     * @param omega 频率 omega (M*omega)
     * @param s 自旋权重 (对于引力波通常为 -2)
     * @param l 谐波指数 l
     * @param m 谐波指数 m
     * @param lambda 角向特征值 (从 SWSH 模块获得)
     * /**
     * @brief 求解重整化角动量 nu
     * 使用割线法寻找 nu，使得 g(nu) = 0
     * @param nu_guess 初始猜测值
     * @return 收敛后的 nu
     */
    Complex solve_nu(Complex nu_guess) const;
    /**
     * @brief 计算方程残差 g(nu)
     * g(nu) = beta_0 + alpha_0 * R_1 + gamma_0 * L_-1
     * 暴露此接口主要用于调试和验证
     */
    Complex calc_g(Complex nu) const;
    
     TeukolskyRadial(Real a_spin, Real omega, int s, int l, int m, Real lambda);
 
     // ==========================================================
     // 基础工具函数
     // ==========================================================
     
     // 复数版 Log Gamma 函数 (辅助函数)
     static Complex log_gamma(Complex z);
 
     // ==========================================================
     // MST 系数与连分式
     // ==========================================================
 
     // 计算 MST 级数系数 alpha_n, beta_n, gamma_n
     // n 是级数索引
     Complex coeff_alpha(Complex nu, int n) const;
     Complex coeff_beta(Complex nu, int n) const;
     Complex coeff_gamma(Complex nu, int n) const;
 
     /**
      * @brief 计算连分式的值
      * 用于求解重整化角动量 nu 的超越方程
      * @param nu 当前猜测的 nu 值
      * @param direction +1 表示向 n 正无穷求和 (收敛部分)，-1 表示向负无穷
      * @return 连分式的值
      */
     Complex continued_fraction(Complex nu, int direction) const;
 
    
 private:
     Real m_a;
     Real m_omega;
     int m_s;
     int m_l;
     int m_m;
     Real m_lambda;
 
     // 常用中间变量，预计算以提高效率
     Real m_epsilon;   // 2 * omega
     Real m_kappa;     // sqrt(1 - q^2)
     Real m_tau;       // (epsilon - m*q) / kappa
     Complex m_epsilon_sq;
     Complex m_tau_sq;
 };
 
 #endif // TEUKOLSKY_RADIAL_H

