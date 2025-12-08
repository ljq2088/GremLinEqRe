/**
 * @file TeukolskyRadial.h
 * @brief Teukolsky 径向方程求解器 (MST 方法)
 * 对应 GremlinEq 的 FT 类及 fujtag 文件夹下的相关实现
 */

 #ifndef TEUKOLSKY_RADIAL_H
 #define TEUKOLSKY_RADIAL_H
 #include <map>
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
    /**
     * @brief 计算 MST 级数展开系数 a_n
     * 对应 GremlinEq 中的 minimal solution 生成逻辑
     * @param nu 特征值
     */
    void compute_coefficients(Complex nu);

    /**
     * @brief 获取计算好的系数 a_n (用于调试或外部计算)
     */
    Complex get_coef(int n) const;

    /**
     * @brief 计算渐进振幅
     * 对应 GremlinEq/src/fujtag/fsum.cc 中的 asympt_amps
     * 计算结果存储在成员变量中
     */
    void compute_amplitudes(Complex nu);

    // 获取振幅 (计算通量所需)
    Complex get_B_inc() const { return m_B_inc; }
    Complex get_B_trans() const { return m_B_trans; }
    Complex get_C_trans() const { return m_C_trans; }
    
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
 
    /**
     * @brief 计算径向函数 R_in(r) 及其对 r 的导数
     * @param r BL 坐标半径
     * @return pair<R, dR/dr>
     */
    std::pair<Complex, Complex> evaluate_R_in(double r);

    // [新增] 强制使用库伦波级数求值 (用于远场)
    std::pair<Complex, Complex> evaluate_R_in_Coulomb(double r);

    // [新增] 强制使用超几何级数求值 (用于近场，即原来的 evaluate_R_in)
    std::pair<Complex, Complex> evaluate_R_in_Hypergeom(double r);

    /**
     * @brief 计算径向函数 R_up(r) 及其对 r 的导数
     * (利用 R_in 和 R_up 的关系或独立级数)
     */
    std::pair<Complex, Complex> evaluate_R_up(double r);
    
    // 获取二阶导数 (利用 Teukolsky 方程 V(r) 反推，不需要数值差分)
    Complex evaluate_ddR(double r, Complex R, Complex dR) const;

    
    // Getters for parameters (needed by Source)
    Real get_omega() const { return m_omega; }
    int get_m() const { return m_m; }
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
    // ===存储特征值 nu ===
    Complex m_nu;
    // 使用 map 存储系数 a_n，支持负索引
    std::map<int, Complex> m_coefficients;
    // 渐进振幅
    Complex m_B_inc;   // 入射波振幅 (Infinity -> Horizon)
    Complex m_B_trans; // 透射波振幅 (Horizon)
    Complex m_C_trans; // 透射波振幅 (Infinity)
    
    // 计算 K_nu 因子 (包含无穷级数求和)
    Complex k_factor(Complex nu) const;
    // 计算 A_minus 因子 (用于 C_trans)
    // 对应 GremlinEq/src/fujtag/fsum.cc 中的 aminus 函数
    Complex calc_aminus(Complex nu) const;
    // 内部辅助：超几何函数 2F1(a,b;c;z)
    static std::pair<Complex, Complex> hypergeom_series(Complex a, Complex b, Complex c, Complex z);
    static std::pair<Complex, Complex> hypergeom_2F1_with_deriv(Complex a, Complex b, Complex c, Complex z); 
    void rk4_step(double r, Complex& R, Complex& dR, double h) const;
    std::pair<Complex, Complex> evaluate_R_in_series(double r) const;
    static std::pair<Complex, Complex> hypergeom_1F1_with_deriv(Complex a, Complex b, Complex z);
    // L = nu + n (复数), rho = omega * r
    static std::pair<Complex, Complex> coulomb_F_with_deriv(Complex L, Complex eta, Complex rho);

 };
 
 #endif // TEUKOLSKY_RADIAL_H

