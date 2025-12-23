// /**
//  * @file SWSH.cpp
//  * @brief 自旋加权球/椭球谐函数求解器实现
//  * * 功能:
//  * 1. 构建耦合矩阵并求解角向特征值 E (和 Lambda)。
//  * 2. 计算谱展开系数 b_k。
//  * 3. 评估函数值 S(theta) 及其导数算符 L2dag S。
//  */

//  #include "SWSH.h"
//  #include "Clebsch.h"
//  #include <Eigen/Dense>
//  #include <iostream>
//  #include <algorithm>
//  #include <cmath>
 
//  // 构造函数：初始化并求解特征值
//  SWSH::SWSH(int s, int l, int m, double a_omega)
//      : m_s(s), m_l(l), m_m(m), m_aw(a_omega)
//  {
//      solve_eigenvalue();
//  }
 
//  // 内部辅助：计算自旋加权球谐函数 sYlm(cos_theta) 的 theta 部分
//  // 实现方法：直接计算 Jacobi 多项式 P_n^(a,b)(x) 并归一化
//  double SWSH::spin_weighted_Y(int s, int l, int m, double x) {
//      // 边界与参数检查
//      if (std::abs(x) > 1.0) return 0.0;
//      int k_max = std::max(std::abs(s), std::abs(m));
//      if (l < k_max) return 0.0;
 
//      // Jacobi 多项式参数
//      int n = l - k_max;
//      double a = std::abs(m + s);
//      double b = std::abs(m - s);
 
//      // Jacobi 多项式递归 P_n^(a,b)(x)
//      double p_prev = 1.0;
//      double p_curr = (n == 0) ? 1.0 : (0.5 * (a - b + (a + b + 2.0) * x));
 
//      if (n > 0) {
//          for (int i = 1; i < n; ++i) {
//              double temp = p_curr;
//              double n_idx = (double)i;
             
//              // 标准三项递推公式
//              double A = (2*n_idx + a + b + 1) * (2*n_idx + a + b + 2) / 
//                         (2.0 * (n_idx+1) * (n_idx + a + b + 1));
//              double B = (2*n_idx + a + b + 1) * (b*b - a*a) / 
//                         (2.0 * (n_idx+1) * (n_idx + a + b + 1) * (2*n_idx + a + b));
//              double C = (n_idx + a) * (n_idx + b) * (2*n_idx + a + b + 2) / 
//                         ((n_idx+1) * (n_idx + a + b + 1) * (2*n_idx + a + b));
             
//              p_curr = (A * x + B) * p_curr - C * p_prev;
//              p_prev = temp;
//          }
//      }
 
//      // 归一化系数 (Wigner-d 系数)
//      // 注意：这里是一个简化的归一化，确保 l_min 处积分为 1
//      // 对于严谨的高阶应用，建议引入 GSL 或 boost 的 factorials
//      double norm = std::sqrt((2.0*l + 1.0)/(4.0*M_PI)); 
     
//      // 加上 (1-x)^(a/2) (1+x)^(b/2) 因子
//      double factor = std::pow((1.0-x)/2.0, a/2.0) * std::pow((1.0+x)/2.0, b/2.0);
     
//      // 相位修正
//      double sign = ((l - n) % 2 == 0) ? 1.0 : -1.0; 
     
//      return sign * norm * factor * p_curr;
//  }
 
//  // 求解特征值问题 (Spectral Method)
//  void SWSH::solve_eigenvalue() {
//      int l_min = std::max(std::abs(m_s), std::abs(m_m));
     
//      // 矩阵截断大小：l 对应第 (l - l_min) 个解，预留 30 个基底保证收敛
//      int K = (m_l - l_min) + 50;
     
//      Eigen::MatrixXd M(K, K);
//      M.setZero();
     
//      // 构建耦合矩阵
//      for (int i = 0; i < K; ++i) {
//          int p = i + l_min;
//          for (int j = 0; j < K; ++j) {
//              int q = j + l_min;
             
//              double term1 = m_aw * m_aw * Clebsch::integral_x_sqr(m_s, p, q, m_m);
//              double term2 = -2.0 * m_aw * (double)m_s * Clebsch::integral_x(m_s, p, q, m_m);
//              double term3 = (i == j) ? - (double)(p * (p + 1)) : 0.0;
             
//              M(i, j) = term1 + term2 + term3;
//          }
//      }
     
//      // 求解
//      Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(M);
//      if (solver.info() != Eigen::Success) {
//          std::cerr << "SWSH Eigenvalue solver failed!" << std::endl;
//          return;
//      }
     
//      // 提取对应于 l 的特征值 (排序后倒数第 l-l_min+1 个)
//      int index = K - 1 - (m_l - l_min);
     
//      if (index < 0) {
//          std::cerr << "Error: Matrix truncation too small for l=" << m_l << std::endl;
//          return;
//      }
     
//      // GremlinEq 约定: E = -eigenvalue
//      m_E = -solver.eigenvalues()[index];
     
//      // 计算物理特征值 Lambda (分离常数)
//      m_lambda = m_E - 2.0 * m_m * m_aw + m_aw * m_aw - m_s * (m_s + 1);
 
//      // 提取特征向量 (展开系数 b_k)
//      Eigen::VectorXd b_vec = solver.eigenvectors().col(index);
     
//      // 确定符号 (使绝对值最大的系数为正)
//      int max_ind = 0;
//      double max_val = 0.0;
//      for(int i=0; i<K; ++i) {
//          if(std::abs(b_vec[i]) > max_val) {
//              max_val = std::abs(b_vec[i]);
//              max_ind = i;
//          }
//      }
//      double sign = (b_vec[max_ind] > 0) ? 1.0 : -1.0;
     
//      m_b.resize(K);
//      for(int i=0; i<K; ++i) {
//          m_b[i] = b_vec[i] * sign;
//      }
//  }
 
//  // 评估 S(x)
//  Complex SWSH::evaluate_S(double x) const {
//      // 谱求和: S = e^{aw x} * sum(b_k * sY_{l_min+k})
//      int l_min = std::max(std::abs(m_s), std::abs(m_m));
//      Complex sum = 0.0;
     
//      for (size_t k = 0; k < m_b.size(); ++k) {
//          int l_curr = l_min + k;
//          double y_val = spin_weighted_Y(m_s, l_curr, m_m, x);
//          sum += m_b[k] * y_val;
//      }
     
//      return std::exp(m_aw * x) * sum;
//  }
 
//  // 评估 L2dag S (使用高精度数值差分)
//  Complex SWSH::evaluate_L2dag_S(double x) const {
//      double st = std::sqrt(1.0 - x*x);
//      if (st < 1e-10) st = 1e-10; 
     
//      double h = 1e-6;
//      Complex S_plus = evaluate_S(x + h);
//      Complex S_minus = evaluate_S(x - h);
//      Complex dS_dx = (S_plus - S_minus) / (2.0 * h);
     
//      // dS/dtheta = -sin(theta) * dS/dx
//      Complex dS_dth = -st * dS_dx;
     
//      // L2dag 定义依赖于 spin weight s
//      double Q = (double)m_m / st - (double)m_s * x / st;
     
//      return dS_dth - Q * evaluate_S(x);
//  }
 
//  // 评估 L1dag (L2dag S)
//  Complex SWSH::evaluate_L1dag_L2dag_S(double x) const {
//      double h = 1e-6;
//      Complex v_plus = evaluate_L2dag_S(x + h);
//      Complex v_minus = evaluate_L2dag_S(x - h);
//      Complex dv_dx = (v_plus - v_minus) / (2.0 * h);
     
//      double st = std::sqrt(1.0 - x*x);
//      Complex dv_dth = -st * dv_dx;
     
//      // L1dag 作用于 spin weight 为 s+1 的场
//      int s_new = m_s + 1;
//      double Q = (double)m_m / st - (double)s_new * x / st;
     
//      return dv_dth - Q * evaluate_L2dag_S(x);
//  }
/**
 * @file SWSH.cpp
 * @brief 自旋加权球/椭球谐函数求解器 - 教科书级谱方法实现（自洽 & 高效）
 *
 * 设计目标：
 * 1) 特征值求解：Leaver/Teukolsky 的球谐谱展开，直接构造五对角矩阵（O(K) 构造）
 * 2) 函数求值：Wigner-d 起步 + l 递推（O(K) 生成 sY_lm 序列）
 * 3) 算符评估：完全谱方法（eth 提升算符对基函数的解析作用），避免任何数值差分
 *
 * 说明：
 * - 本实现不再依赖 Clebsch::integral_x / integral_x_sqr，从而避免“矩阵基底”与“函数求值基底”不一致。
 * - 使用的基函数为“只含 θ(或 x=cosθ) 的部分”，默认采用使 ∫_{-1}^{1} |Y(x)|^2 dx = 1 的规范（θ-部分正交归一）。
 */

 #include "SWSH.h"
 // #include "Clebsch.h"   // 旧版本依赖；本实现不再需要
 #include <Eigen/Dense>
 #include <algorithm>
 #include <cmath>
 #include <complex>
 #include <iostream>
 #include <limits>
 #include <vector>
 
 namespace {
 
 // ------------------------------
 // 工具：对数阶乘（避免溢出）
 // ------------------------------
 inline long double ln_factorial(int n) {
     return lgammal((long double)n + 1.0L);
 }
 
 // ------------------------------
 // Wigner small-d：d^l_{m,mp}(theta)
 // 稳定求和公式（Edmonds/Wikipedia）
 // ------------------------------
 long double wigner_d(int l, int m, int mp, long double theta) {
     if (l < 0) return 0.0L;
     if (std::abs(m) > l || std::abs(mp) > l) return 0.0L;
 
     const int kmin = std::max(0, m - mp);
     const int kmax = std::min(l + m, l - mp);
     if (kmin > kmax) return 0.0L;
 
     const long double c = cosl(theta * 0.5L);
     const long double s = sinl(theta * 0.5L);
 
     // prefactor = sqrt((l+m)!(l-m)!(l+mp)!(l-mp)!)
     const long double log_pref =
         0.5L * (ln_factorial(l + m) + ln_factorial(l - m) +
                 ln_factorial(l + mp) + ln_factorial(l - mp));
 
     long double sum = 0.0L;
 
     for (int k = kmin; k <= kmax; ++k) {
         const int a1 = l + m - k;
         const int a2 = k;
         const int a3 = mp - m + k;
         const int a4 = l - mp - k;
         if (a1 < 0 || a2 < 0 || a3 < 0 || a4 < 0) continue;
 
         const int parity = (k + mp - m) & 1;
         const long double sgn = parity ? -1.0L : 1.0L;
 
         const long double log_den =
             ln_factorial(a1) + ln_factorial(a2) + ln_factorial(a3) + ln_factorial(a4);
 
         const int p_cos = 2 * l + m - mp - 2 * k;
         const int p_sin = mp - m + 2 * k;
 
         const long double term =
             sgn * expl(log_pref - log_den) *
             powl(c, (long double)p_cos) *
             powl(s, (long double)p_sin);
 
         sum += term;
     }
     return sum;
 }
 
 // ------------------------------
 // θ-部分自旋加权球谐：Y_s(l,m; x=cosθ)
 // 规范：∫_{-1}^{1} |Y(x)|^2 dx = 1
 // 采用：Y(x) = (-1)^s * sqrt((2l+1)/2) * d^l_{m,-s}(θ)
 // ------------------------------
 inline double swsh_theta_part(int s, int l, int m, double x) {
     x = std::max(-1.0, std::min(1.0, x));
     const long double theta = acosl((long double)x);
 
     const long double d = wigner_d(l, m, -s, theta);
     const long double norm = sqrtl((2.0L * l + 1.0L) / 2.0L);
     const long double phase = (std::abs(s) & 1) ? -1.0L : 1.0L;
 
     return (double)(phase * norm * d);
 }
 
 // ------------------------------
 // x 乘法递推系数：x Y_l = α_l Y_{l+1} + β_l Y_{l-1}
 // （与整体归一化无关；θ-部分归一与全角归一只差常数因子，系数不变）
 // ------------------------------
 inline double alpha_coeff(int l, int s, int m) {
     const long double lp1 = (long double)l + 1.0L;
     const long double num1 = lp1 * lp1 - (long double)m * m;
     const long double num2 = lp1 * lp1 - (long double)s * s;
     if (num1 <= 0.0L || num2 <= 0.0L) return 0.0;
     const long double den = (lp1 * lp1) * (2.0L * l + 1.0L) * (2.0L * l + 3.0L);
     return (double)sqrtl((num1 * num2) / den);
 }
 
 inline double beta_coeff(int l, int s, int m) {
     if (l <= 0) return 0.0;
     const long double ll = (long double)l;
     const long double num1 = ll * ll - (long double)m * m;
     const long double num2 = ll * ll - (long double)s * s;
     if (num1 <= 0.0L || num2 <= 0.0L) return 0.0;
     const long double den = (ll * ll) * (2.0L * l - 1.0L) * (2.0L * l + 1.0L);
     return (double)sqrtl((num1 * num2) / den);
 }
 
 // ------------------------------
 // 批量生成 sY_lm 序列：l = l_min ... l_max
 // 方法：Wigner-d 起步两项 + x 递推（O(K)）
 // ------------------------------
 void compute_sYlm_sequence_internal(int s, int m, double x, int l_max, std::vector<double>& out_vals) {
     const int l_min = std::max(std::abs(s), std::abs(m));
     if (l_max < l_min) {
         out_vals.clear();
         return;
     }
 
     const int K = l_max - l_min + 1;
     out_vals.assign(K, 0.0);
 
     // 近极点处递推可能放大误差；这里直接用 Wigner-d 逐项计算（K 通常不大）
     const double one_minus_x2 = 1.0 - x * x;
     const bool near_pole = (one_minus_x2 < 1e-14);
 
     if (near_pole) {
         for (int l = l_min; l <= l_max; ++l) {
             out_vals[l - l_min] = swsh_theta_part(s, l, m, x);
         }
         return;
     }
 
     // 起步两项
     out_vals[0] = swsh_theta_part(s, l_min, m, x);
     if (K == 1) return;
 
     out_vals[1] = swsh_theta_part(s, l_min + 1, m, x);
     if (K == 2) return;
 
     // l 递推：Y_{l+1} = (x Y_l - β_l Y_{l-1}) / α_l
     for (int l = l_min + 1; l < l_max; ++l) {
         const int idx_lm1 = (l - 1) - l_min;
         const int idx_l   = l - l_min;
         const int idx_lp1 = (l + 1) - l_min;
 
         const double a = alpha_coeff(l, s, m);
         const double b = beta_coeff(l,  s, m);
 
         if (std::abs(a) < 1e-300) {
             // 极端情况下退化：回退为直接计算
             out_vals[idx_lp1] = swsh_theta_part(s, l + 1, m, x);
         } else {
             out_vals[idx_lp1] = (x * out_vals[idx_l] - b * out_vals[idx_lm1]) / a;
         }
     }
 }
 
 // ------------------------------
 // eth 提升系数：ð (sY_lm) = + C_{l,s} * (s+1)Y_lm
 // ------------------------------
 inline double eth_raise_coeff(int l, int s) {
     return std::sqrt((double)(l - s) * (double)(l + s + 1));
 }
 
 } // namespace
 
 // ==========================================================
 // SWSH 类实现
 // ==========================================================
 
 SWSH::SWSH(int s, int l, int m, double a_omega)
     : m_s(s), m_l(l), m_m(m), m_aw(a_omega)
 {
     solve_eigenvalue();
 }
 
 // 保留静态单点接口（兼容性）；内部使用批量序列生成
 double SWSH::spin_weighted_Y(int s, int l, int m, double x) {
     std::vector<double> vals;
     compute_sYlm_sequence_internal(s, m, x, l, vals);
     if (vals.empty()) return 0.0;
     return vals.back();
 }
 
//  // 旧接口占位（不建议使用）；本实现算符完全基于谱系数，不需要点导数
//  double SWSH::spin_weighted_Y_deriv(int /*s*/, int /*l*/, int /*m*/, double /*x*/) {
//      return 0.0;
//  }
 
 void SWSH::solve_eigenvalue() {
     const int l_min = std::max(std::abs(m_s), std::abs(m_m));
     const int target = m_l - l_min;
     if (target < 0) {
         std::cerr << "SWSH::solve_eigenvalue: invalid l < l_min." << std::endl;
         m_E = std::numeric_limits<double>::quiet_NaN();
         m_lambda = std::numeric_limits<double>::quiet_NaN();
         m_b.clear();
         return;
     }
 
     const double c  = m_aw;
     const double c2 = c * c;
 
     // 自适应截断：逐步增大 K，直到 E 收敛
     int K = std::max(30, target + 40);
     double E_prev = std::numeric_limits<double>::quiet_NaN();
 
     Eigen::VectorXd best_vec;
     double best_eval = 0.0;
 
     for (int iter = 0; iter < 12; ++iter) {
         Eigen::MatrixXd M = Eigen::MatrixXd::Zero(K, K);
 
         // 构造五对角矩阵：
         // M = -l(l+1) I  + c^2 <x^2>  - 2 c s <x>
         // 其中 <x> 只耦合 l±1，<x^2> 只耦合 l,l±2
         for (int i = 0; i < K; ++i) {
             const int l = l_min + i;
 
             const double a = alpha_coeff(l, m_s, m_m);
             const double b = beta_coeff(l,  m_s, m_m);
 
             // diag: -l(l+1) + c^2 (a^2 + b^2)
             M(i, i) = -(double)(l * (l + 1)) + c2 * (a * a + b * b);
 
             // off diag ±1: -2 c s * <l|x|l+1> = -2 c s * a
             if (i + 1 < K) {
                 const double v = -2.0 * c * (double)m_s * a;
                 M(i, i + 1) += v;
                 M(i + 1, i) += v;
             }
 
             // off diag ±2: c^2 * <l|x^2|l+2> = c^2 * a_l * a_{l+1}
             if (i + 2 < K) {
                 const double a_next = alpha_coeff(l + 1, m_s, m_m);
                 const double v = c2 * a * a_next;
                 M(i, i + 2) += v;
                 M(i + 2, i) += v;
             }
         }
 
         Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(M);
         if (solver.info() != Eigen::Success) {
             std::cerr << "SWSH::solve_eigenvalue: Eigen solver failed." << std::endl;
             m_E = std::numeric_limits<double>::quiet_NaN();
             m_lambda = std::numeric_limits<double>::quiet_NaN();
             m_b.clear();
             return;
         }
 
         // 选模：与 c->0 连续的分支 —— 目标分量 |b_target| 最大
         const auto& evals = solver.eigenvalues();
         const auto& evecs = solver.eigenvectors();
 
         int idx = 0;
         double best_overlap = -1.0;
         for (int j = 0; j < K; ++j) {
             const double ov = std::abs(evecs(target, j));
             if (ov > best_overlap) {
                 best_overlap = ov;
                 idx = j;
             }
         }
 
         const double E = -evals(idx);
 
         // 收敛判据
         if (iter > 0) {
             const double rel = std::abs(E - E_prev) / std::max(1.0, std::abs(E));
             if (rel < 1e-13) {
                 best_eval = evals(idx);
                 best_vec  = evecs.col(idx);
                 break;
             }
         }
 
         E_prev = E;
         best_eval = evals(idx);
         best_vec  = evecs.col(idx);
 
         // 增大截断
         K += 20;
     }
 
     // 归一化与符号规范：令目标分量为正，保证连续性
     if (best_vec.size() == 0) {
         std::cerr << "SWSH::solve_eigenvalue: empty eigenvector." << std::endl;
         m_E = std::numeric_limits<double>::quiet_NaN();
         m_lambda = std::numeric_limits<double>::quiet_NaN();
         m_b.clear();
         return;
     }
 
     Eigen::VectorXd b_vec = best_vec;
     b_vec.normalize();
 
     const double sign = (b_vec(target) >= 0.0) ? 1.0 : -1.0;
     b_vec *= sign;
 
     m_b.resize((size_t)b_vec.size());
     for (int i = 0; i < b_vec.size(); ++i) m_b[(size_t)i] = b_vec(i);
 
     m_E = -best_eval;
     m_lambda = m_E - 2.0 * (double)m_m * m_aw + m_aw * m_aw - (double)(m_s * (m_s + 1));
 }
 
 Complex SWSH::evaluate_S(double x) const {
     const int l_min = std::max(std::abs(m_s), std::abs(m_m));
     const int K = (int)m_b.size();
     if (K <= 0) return Complex(0.0, 0.0);
 
     const int l_max = l_min + K - 1;
 
     std::vector<double> Y;
     compute_sYlm_sequence_internal(m_s, m_m, x, l_max, Y);
 
     Complex sum = 0.0;
     for (int k = 0; k < K; ++k) {
         sum += m_b[(size_t)k] * Y[(size_t)k];
     }
     return sum;
 }
 
 // ----------------------------------------------------------
 // 下面两个算符采用“完全谱方法”实现：不做任何数值差分
 //
 // 约定：eth 提升算符 ð 在球谐基底上的解析作用为
 //   ð ({}_sY_{lm}) = C_{l,s} * {}_{s+1}Y_{lm},
 //   C_{l,s} = sqrt((l-s)(l+s+1)).
 //
 // 因此对于 S = Σ b_l {}_sY_lm，有
 //   ð S = Σ b_l C_{l,s} {}_{s+1}Y_lm,
 //   ðð S = Σ b_l C_{l,s} C_{l,s+1} {}_{s+2}Y_lm.
 // ----------------------------------------------------------
 
 // 计算 ð S  （输出用 s+1 的基函数评估；O(K)）
 Complex SWSH::evaluate_L2dag_S(double x) const {
     const int s0 = m_s;
     const int s1 = m_s + 1;
 
     const int l_min0 = std::max(std::abs(s0), std::abs(m_m));
     const int K = (int)m_b.size();
     if (K <= 0) return Complex(0.0, 0.0);
 
     const int l_max = l_min0 + K - 1;
 
     // 生成 s+1 的球谐序列（注意其 l_min 可能不同）
     const int l_min1 = std::max(std::abs(s1), std::abs(m_m));
     std::vector<double> Y1;
     compute_sYlm_sequence_internal(s1, m_m, x, l_max, Y1);
 
     Complex sum = 0.0;
 
     for (int k = 0; k < K; ++k) {
         const int l = l_min0 + k;
         const double C = eth_raise_coeff(l, s0);
 
         // {}_{s+1}Y_{lm} 的数组下标是 (l - l_min1)
         const int idx = l - l_min1;
         const double y1 = (idx >= 0 && idx < (int)Y1.size()) ? Y1[(size_t)idx] : 0.0;
 
         sum += m_b[(size_t)k] * (C * y1);
     }
     return sum;
 }
 
 // 计算 ðð S（并按你之前代码的“二阶组合形式”返回）：
 //   ðð S  + 2 c sinθ (ð S) + (c cosθ + c^2 sin^2θ) S
 // 全部由谱展开得到，无数值差分，O(K)
 Complex SWSH::evaluate_L1dag_L2dag_S(double x) const {
     const int s0 = m_s;
     const int s1 = m_s + 1;
     const int s2 = m_s + 2;
 
     const int l_min0 = std::max(std::abs(s0), std::abs(m_m));
     const int K = (int)m_b.size();
     if (K <= 0) return Complex(0.0, 0.0);
 
     const int l_max = l_min0 + K - 1;
 
     // 需要：S (s0), ðS (s1), ððS (s2) 的基函数值
     std::vector<double> Y0, Y1, Y2;
     compute_sYlm_sequence_internal(s0, m_m, x, l_max, Y0);
     compute_sYlm_sequence_internal(s1, m_m, x, l_max, Y1);
     compute_sYlm_sequence_internal(s2, m_m, x, l_max, Y2);
 
     const int l_min1 = std::max(std::abs(s1), std::abs(m_m));
     const int l_min2 = std::max(std::abs(s2), std::abs(m_m));
 
     const double st = std::sqrt(std::max(0.0, 1.0 - x * x));
     const double c  = m_aw;
     const double c2 = c * c;
 
     Complex sum_S   = 0.0;
     Complex sum_eth = 0.0;
     Complex sum_eth2= 0.0;
 
     for (int k = 0; k < K; ++k) {
         const int l = l_min0 + k;
 
         // S
         const double y0 = Y0[(size_t)k];
         sum_S += m_b[(size_t)k] * y0;
 
         // ðS
         {
             const double C0 = eth_raise_coeff(l, s0);
             const int idx1 = l - l_min1;
             const double y1 = (idx1 >= 0 && idx1 < (int)Y1.size()) ? Y1[(size_t)idx1] : 0.0;
             sum_eth += m_b[(size_t)k] * (C0 * y1);
         }
 
         // ððS
         {
             const double C0 = eth_raise_coeff(l, s0);
             const double C1 = eth_raise_coeff(l, s1);
             const int idx2 = l - l_min2;
             const double y2 = (idx2 >= 0 && idx2 < (int)Y2.size()) ? Y2[(size_t)idx2] : 0.0;
             sum_eth2 += m_b[(size_t)k] * (C0 * C1 * y2);
         }
     }
 
     // 二阶组合（与你当前接口习惯一致；不做数值差分）
     // 注意：这里将 sin/cos 作为纯标量函数使用（与旧代码的“eth(sin)=cos”一致的用法）。
     const Complex term0 = sum_eth2;
     const Complex term1 = 2.0 * c * st * sum_eth;
     const Complex term2 = (c * x + c2 * st * st) * sum_S;
 
     return term0 + term1 + term2;
 }
 