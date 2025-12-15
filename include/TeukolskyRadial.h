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
 struct AsymptoticAmplitudes {
   std::complex<double> R_in_coef_inf_inc;   // R^in 在无穷远的入射系数 (对应 r^{-1} e^{-i*omega*r*})
   std::complex<double> R_in_coef_inf_trans; // R^in 在无穷远的透射系数 (对应 r^{-(2s+1)} e^{+i*omega*r*})
   // 如果需要 R^up 的系数也可以加在这里
};
// 存放物理散射系数和 Wronskian
struct PhysicalAmplitudes {
   std::complex<double> B_trans; // Horizon transmission coeff of R^in
   std::complex<double> B_inc;   // Infinity incidence coeff of R^in
   std::complex<double> B_ref;   // Infinity reflection coeff of R^in
   std::complex<double> C_trans; // Infinity transmission coeff of R^up (outgoing)
};

 class TeukolskyRadial {
 public:
    /**
     * @brief 构造函数
     * @param M 黑洞质量
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

 


   TeukolskyRadial(Real M, Real a_spin, Real omega, int s, int l, int m, Real lambda);
 
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
   // 计算 K_nu 因子 (包含无穷级数求和)
   Complex k_factor(Complex nu) const;
   // 输入: nu, 以及之前算好的级数系数 map a_coeffs
   AsymptoticAmplitudes ComputeAmplitudes(std::complex<double> nu, 
      const std::map<int, std::complex<double>>& a_coeffs);
   PhysicalAmplitudes ComputePhysicalAmplitudes(
      std::complex<double> nu,
      const AsymptoticAmplitudes& amp_nu,        // nu 分支的 A+, A-
      std::complex<double> K_nu,                 // nu 分支的 K因子
      const AsymptoticAmplitudes& amp_minus_nu,  // -nu-1 分支的 A+, A-
      std::complex<double> K_minus_nu            // -nu-1 分支的 K因子
   );

     /**
      * @brief 计算连分式的值
      * 用于求解重整化角动量 nu 的超越方程
      * @param nu 当前猜测的 nu 值
      * @param direction +1 表示向 n 正无穷求和 (收敛部分)，-1 表示向负无穷
      * @return 连分式的值
      */
   Complex continued_fraction(Complex nu, int direction) const;
   std::map<int, std::complex<double>> ComputeSeriesCoefficients(std::complex<double> nu, int n_max);
   PhysicalAmplitudes ComputePhysicalAmplitudes(std::complex<double> nu, 
      const std::map<int, std::complex<double>>& a_coeffs,
      const AsymptoticAmplitudes& amps_nu);

   // 计算径向函数 R_in(r) 及其导数 dR_in/dr
   // 适用范围: 近视界区域 (r 接近 r_+)
   // 返回值: pair.first = R(r), pair.second = dR/dr
   std::pair<std::complex<double>, std::complex<double>> Evaluate_Hypergeometric(
   double r, 
   std::complex<double> nu, 
   const std::map<int, std::complex<double>>& a_coeffs
);
   // 辅助函数：计算高斯超几何函数 2F1(a,b;c;z) 及其导数
   std::complex<double> Hyp2F1(std::complex<double> a, std::complex<double> b, 
      std::complex<double> c, std::complex<double> z,bool regularized=false);
   // 计算远场径向函数 R_C^nu(r) 及其导数
   // 适用范围: 远场区域 (r 较大)
   // 返回值: pair.first = R_C(r), pair.second = dR_C/dr
   std::pair<std::complex<double>, std::complex<double>> Evaluate_Coulomb(
   double r, 
   std::complex<double> nu, 
   const std::map<int, std::complex<double>>& a_coeffs
);

   // 辅助函数：计算合流超几何函数 1F1(a;b;z)
   // 对应 LRR Eq. 142 中的 Phi 或 M 函数
   std::complex<double> Hyp1F1(std::complex<double> a, std::complex<double> b, std::complex<double> z,bool regularized=false);
   // 全域径向函数求值器
   // 自动在近场和远场算法间切换
   std::pair<Complex, Complex> Evaluate_R_in(
      double r,
      Complex nu,
      Complex K_nu,       // 预先算好的 K_nu
      Complex K_neg_nu,   // 预先算好的 K_{-nu-1}
      const std::map<int, Complex>& a_coeffs_pos, // nu 的系数
      const std::map<int, Complex>& a_coeffs_neg, // -nu-1 的系数
      double r_match = 5.0 // 拼接半径，默认 5M (可视情况调整)
  );
 private:
   Real m_M;
   Real m_a;
   Real m_omega;
   int m_s;
   int m_l;
   int m_m;
   Real m_lambda;

   // 常用中间变量，预计算以提高效率
   Real m_epsilon;   // 2 * omega
   Real m_kappa;     // sqrt(1 - q^2)
   Real q;
   Real m_tau;       // (epsilon - m*q) / kappa
   Complex m_epsilon_sq;
   Complex m_tau_sq;
   // ===存储特征值 nu ===
   Complex m_nu;

    
    
   
 };
 
 #endif // TEUKOLSKY_RADIAL_H

