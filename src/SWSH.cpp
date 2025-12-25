/**
 * @file SWSH.cpp
 * @brief 自旋加权球/椭球谐函数求解器 - LRR 兼容教科书级谱方法实现
 */

 #include "SWSH.h"
 #include <Eigen/Dense>
 #include <algorithm>
 #include <cmath>
 #include <complex>
 #include <iostream>
 #include <limits>
 #include <vector>
 
 namespace {
 
 // ==========================================================
 // 数学工具函数
 // ==========================================================
 
 inline long double ln_factorial(int n) {
     return std::lgamma((long double)n + 1.0L);
 }
 
 long double wigner_d(int l, int m, int mp, long double theta) {
     if (l < 0) return 0.0L;
     if (std::abs(m) > l || std::abs(mp) > l) return 0.0L;
 
     const int kmin = std::max(0, m - mp);
     const int kmax = std::min(l + m, l - mp);
     if (kmin > kmax) return 0.0L;
 
     const long double c = std::cos(theta * 0.5L);
     const long double s = std::sin(theta * 0.5L);
     
     const long double log_pref = 0.5L * (ln_factorial(l + m) + ln_factorial(l - m) +
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
         
         const long double log_den = ln_factorial(a1) + ln_factorial(a2) + 
                                     ln_factorial(a3) + ln_factorial(a4);
 
         const int p_cos = 2 * l + m - mp - 2 * k;
         const int p_sin = mp - m + 2 * k;
 
         sum += sgn * std::exp(log_pref - log_den) * std::pow(c, (long double)p_cos) * std::pow(s, (long double)p_sin);
     }
     return sum;
 }
 
 inline double swsh_theta_part(int s, int l, int m, double x) {
     x = std::max(-1.0, std::min(1.0, x));
     const long double theta = std::acos((long double)x);
     const long double d = wigner_d(l, m, -s, theta);
     const long double norm = std::sqrt((2.0L * l + 1.0L) / 2.0L);
     const long double phase = (std::abs(s) & 1) ? -1.0L : 1.0L;
     return (double)(phase * norm * d);
 }
 
 inline double alpha_coeff(int l, int s, int m) {
     const long double lp1 = (long double)l + 1.0L;
     const long double num1 = lp1 * lp1 - (long double)m * m;
     const long double num2 = lp1 * lp1 - (long double)s * s;
     if (num1 <= 0.0L || num2 <= 0.0L) return 0.0;
     const long double den = (lp1 * lp1) * (2.0L * l + 1.0L) * (2.0L * l + 3.0L);
     return (double)std::sqrt((num1 * num2) / den);
 }
 
 inline double beta_coeff(int l, int s, int m) {
     if (l <= 0) return 0.0;
     const long double ll = (long double)l;
     const long double num1 = ll * ll - (long double)m * m;
     const long double num2 = ll * ll - (long double)s * s;
     if (num1 <= 0.0L || num2 <= 0.0L) return 0.0;
     const long double den = (ll * ll) * (2.0L * l - 1.0L) * (2.0L * l + 1.0L);
     return (double)std::sqrt((num1 * num2) / den);
 }
 
 inline double eth_raise_coeff(int l, int s) {
     return std::sqrt((double)(l - s) * (double)(l + s + 1));
 }
 
 void compute_sYlm_sequence_internal(int s, int m, double x, int l_max, std::vector<double>& out_vals) {
     const int l_min = std::max(std::abs(s), std::abs(m));
     if (l_max < l_min) { out_vals.clear(); return; }
 
     const int K = l_max - l_min + 1;
     out_vals.assign(K, 0.0);
 
     const double one_minus_x2 = 1.0 - x * x;
     if (one_minus_x2 < 1e-10) { 
         for (int l = l_min; l <= l_max; ++l) {
             out_vals[l - l_min] = swsh_theta_part(s, l, m, x);
         }
         return;
     }
 
     out_vals[0] = swsh_theta_part(s, l_min, m, x);
     if (K == 1) return;
     out_vals[1] = swsh_theta_part(s, l_min + 1, m, x);
     if (K == 2) return;
 
     for (int l = l_min + 1; l < l_max; ++l) {
         const int idx_lm1 = (l - 1) - l_min;
         const int idx_l   = l - l_min;
         const int idx_lp1 = (l + 1) - l_min;
         const double a = alpha_coeff(l, s, m);
         const double b = beta_coeff(l,  s, m);
         if (std::abs(a) < 1e-300) {
             out_vals[idx_lp1] = swsh_theta_part(s, l + 1, m, x);
         } else {
             out_vals[idx_lp1] = (x * out_vals[idx_l] - b * out_vals[idx_lm1]) / a;
         }
     }
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
 
 double SWSH::spin_weighted_Y(int s, int l, int m, double x) {
     // 静态调用时，调用者需自行决定是否传入 -m
     std::vector<double> vals;
     compute_sYlm_sequence_internal(s, m, x, l, vals);
     if (vals.empty()) return 0.0;
     return vals.back();
 }
 
 void SWSH::solve_eigenvalue() {
     
     const int m_calc = m_m; 
     
     const int l_min = std::max(std::abs(m_s), std::abs(m_calc));
     const int target = m_l - l_min;
     
     if (target < 0) {
         m_E = m_lambda = std::numeric_limits<double>::quiet_NaN();
         return;
     }
 
     const double c  = m_aw;
     const double c2 = c * c;
     int K = std::max(30, target + 40);
     double E_prev = std::numeric_limits<double>::quiet_NaN();
     Eigen::VectorXd best_vec;
     double best_eval = 0.0;
 
     for (int iter = 0; iter < 12; ++iter) {
         Eigen::MatrixXd M = Eigen::MatrixXd::Zero(K, K);
         for (int i = 0; i < K; ++i) {
             const int l = l_min + i;
             const double a = alpha_coeff(l, m_s, m_calc);
             const double b = beta_coeff(l,  m_s, m_calc);
             
             M(i, i) = -(double)(l * (l + 1)) + c2 * (a * a + b * b);
             if (i + 1 < K) {
                 const double v = -2.0 * c * (double)m_s * a;
                 M(i, i + 1) += v; M(i + 1, i) += v;
             }
             if (i + 2 < K) {
                 const double a_next = alpha_coeff(l + 1, m_s, m_calc);
                 const double v = c2 * a * a_next;
                 M(i, i + 2) += v; M(i + 2, i) += v;
             }
         }
 
         Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(M);
         if (solver.info() != Eigen::Success) return;
 
         int idx = 0;
         double best_overlap = -1.0;
         const auto& evecs = solver.eigenvectors();
         for (int j = 0; j < K; ++j) {
             if (std::abs(evecs(target, j)) > best_overlap) {
                 best_overlap = std::abs(evecs(target, j));
                 idx = j;
             }
         }
 
         const double E = -solver.eigenvalues()(idx);
         if (iter > 0 && std::abs(E - E_prev) / std::max(1.0, std::abs(E)) < 1e-14) {
             best_eval = solver.eigenvalues()(idx);
             best_vec = evecs.col(idx);
             break;
         }
         E_prev = E;
         best_eval = solver.eigenvalues()(idx);
         best_vec = evecs.col(idx);
         K += 20;
     }
 
     if (best_vec.size() == 0) return;
     
     Eigen::VectorXd b_vec = best_vec;
     b_vec.normalize();
     if (b_vec(target) < 0) b_vec = -b_vec;
 
     m_b.resize(b_vec.size());
     for (int i = 0; i < b_vec.size(); ++i) m_b[i] = b_vec(i);
 
     m_E = -best_eval;
     m_lambda = m_E - 2.0 * m_m * m_aw + c2 - m_s * (m_s + 1);
 }
 
 Complex SWSH::evaluate_S(double x) const {
     
     const int m_calc = m_m;
     const int l_min = std::max(std::abs(m_s), std::abs(m_calc));
     const int K = (int)m_b.size();
     if (K <= 0) return 0.0;
     
     std::vector<double> Y;
     compute_sYlm_sequence_internal(m_s, m_calc, x, l_min + K - 1, Y);
     
     Complex sum = 0.0;
     for (int k = 0; k < K; ++k) sum += m_b[k] * Y[k];
     return sum;
 }
 
 // L_{-2}^dag S = -eth S + aw sin S
 Complex SWSH::evaluate_L2dag_S(double x) const {
     const int m_calc = m_m; 
     const int s0 = m_s;
     const int s1 = m_s + 1;
     const int l_min0 = std::max(std::abs(s0), std::abs(m_calc));
     const int K = (int)m_b.size();
     if (K <= 0) return 0.0;
 
     const int l_min1 = std::max(std::abs(s1), std::abs(m_calc));
     std::vector<double> Y1;
     compute_sYlm_sequence_internal(s1, m_calc, x, l_min0 + K - 1, Y1);
     
     std::vector<double> Y0;
     compute_sYlm_sequence_internal(s0, m_calc, x, l_min0 + K - 1, Y0);
 
     const double st = std::sqrt(std::max(0.0, 1.0 - x * x));
     const double c  = m_aw;
     Complex sum = 0.0;
 
     for (int k = 0; k < K; ++k) {
         const int l = l_min0 + k;
         const double C = eth_raise_coeff(l, s0);
         const int idx1 = l - l_min1;
         const double y1 = (idx1 >= 0 && idx1 < (int)Y1.size()) ? Y1[idx1] : 0.0;
         const double y0 = Y0[k];
 
         // L^dag = -eth + aw*sin
         // eth -> C*Y1
         sum += m_b[k] * (-C * y1 + c * st * y0);
     }
     return sum;
 }
 
 // L_{-1}^dag L_{-2}^dag S
 // Operator Algebra: eth^2 - 2 aw sin eth + (aw cos + (aw sin)^2)
 Complex SWSH::evaluate_L1dag_L2dag_S(double x) const {
     const int m_calc = m_m; 
     const int s0 = m_s;
     const int s1 = m_s + 1;
     const int s2 = m_s + 2;
     const int l_min0 = std::max(std::abs(s0), std::abs(m_calc));
     const int K = (int)m_b.size();
     if (K <= 0) return 0.0;
 
     std::vector<double> Y0, Y1, Y2;
     int l_max = l_min0 + K - 1;
     compute_sYlm_sequence_internal(s0, m_calc, x, l_max, Y0);
     compute_sYlm_sequence_internal(s1, m_calc, x, l_max, Y1);
     compute_sYlm_sequence_internal(s2, m_calc, x, l_max, Y2);
 
     const int l_min1 = std::max(std::abs(s1), std::abs(m_calc));
     const int l_min2 = std::max(std::abs(s2), std::abs(m_calc));
     const double st = std::sqrt(std::max(0.0, 1.0 - x * x));
     const double c  = m_aw;
     const double c2 = c * c;
 
     Complex sum_S = 0.0, sum_eth = 0.0, sum_eth2 = 0.0;
 
     for (int k = 0; k < K; ++k) {
         const int l = l_min0 + k;
         sum_S += m_b[k] * Y0[k];
 
         const double C0 = eth_raise_coeff(l, s0);
         const int idx1 = l - l_min1;
         const double y1 = (idx1 >= 0 && idx1 < (int)Y1.size()) ? Y1[idx1] : 0.0;
         sum_eth += m_b[k] * (C0 * y1);
 
         const double C1 = eth_raise_coeff(l, s1);
         const int idx2 = l - l_min2;
         const double y2 = (idx2 >= 0 && idx2 < (int)Y2.size()) ? Y2[idx2] : 0.0;
         sum_eth2 += m_b[k] * (C0 * C1 * y2);
     }
 
     // 组合算符
     const Complex term0 = sum_eth2;                 // + eth^2
     const Complex term1 = -2.0 * c * st * sum_eth;  // - 2 aw sin eth
     const Complex term2 = (c * x + c2 * st * st) * sum_S; // + aw cos + ...
 
     return term0 + term1 + term2;
 }