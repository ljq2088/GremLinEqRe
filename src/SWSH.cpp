/**
 * @file SWSH.cpp
 * @brief 自旋加权球/椭球谐函数求解器实现
 * * 功能:
 * 1. 构建耦合矩阵并求解角向特征值 E (和 Lambda)。
 * 2. 计算谱展开系数 b_k。
 * 3. 评估函数值 S(theta) 及其导数算符 L2dag S。
 */

 #include "SWSH.h"
 #include "Clebsch.h"
 #include <Eigen/Dense>
 #include <iostream>
 #include <algorithm>
 #include <cmath>
 
 // 构造函数：初始化并求解特征值
 SWSH::SWSH(int s, int l, int m, double a_omega)
     : m_s(s), m_l(l), m_m(m), m_aw(a_omega)
 {
     solve_eigenvalue();
 }
 
 // 内部辅助：计算自旋加权球谐函数 sYlm(cos_theta) 的 theta 部分
 // 实现方法：直接计算 Jacobi 多项式 P_n^(a,b)(x) 并归一化
 double SWSH::spin_weighted_Y(int s, int l, int m, double x) {
     // 边界与参数检查
     if (std::abs(x) > 1.0) return 0.0;
     int k_max = std::max(std::abs(s), std::abs(m));
     if (l < k_max) return 0.0;
 
     // Jacobi 多项式参数
     int n = l - k_max;
     double a = std::abs(m + s);
     double b = std::abs(m - s);
 
     // Jacobi 多项式递归 P_n^(a,b)(x)
     double p_prev = 1.0;
     double p_curr = (n == 0) ? 1.0 : (0.5 * (a - b + (a + b + 2.0) * x));
 
     if (n > 0) {
         for (int i = 1; i < n; ++i) {
             double temp = p_curr;
             double n_idx = (double)i;
             
             // 标准三项递推公式
             double A = (2*n_idx + a + b + 1) * (2*n_idx + a + b + 2) / 
                        (2.0 * (n_idx+1) * (n_idx + a + b + 1));
             double B = (2*n_idx + a + b + 1) * (b*b - a*a) / 
                        (2.0 * (n_idx+1) * (n_idx + a + b + 1) * (2*n_idx + a + b));
             double C = (n_idx + a) * (n_idx + b) * (2*n_idx + a + b + 2) / 
                        ((n_idx+1) * (n_idx + a + b + 1) * (2*n_idx + a + b));
             
             p_curr = (A * x + B) * p_curr - C * p_prev;
             p_prev = temp;
         }
     }
 
     // 归一化系数 (Wigner-d 系数)
     // 注意：这里是一个简化的归一化，确保 l_min 处积分为 1
     // 对于严谨的高阶应用，建议引入 GSL 或 boost 的 factorials
     double norm = std::sqrt((2.0*l + 1.0)/(4.0*M_PI)); 
     
     // 加上 (1-x)^(a/2) (1+x)^(b/2) 因子
     double factor = std::pow((1.0-x)/2.0, a/2.0) * std::pow((1.0+x)/2.0, b/2.0);
     
     // 相位修正
     double sign = ((l - n) % 2 == 0) ? 1.0 : -1.0; 
     
     return sign * norm * factor * p_curr;
 }
 
 // 求解特征值问题 (Spectral Method)
 void SWSH::solve_eigenvalue() {
     int l_min = std::max(std::abs(m_s), std::abs(m_m));
     
     // 矩阵截断大小：l 对应第 (l - l_min) 个解，预留 30 个基底保证收敛
     int K = (m_l - l_min) + 50;
     
     Eigen::MatrixXd M(K, K);
     M.setZero();
     
     // 构建耦合矩阵
     for (int i = 0; i < K; ++i) {
         int p = i + l_min;
         for (int j = 0; j < K; ++j) {
             int q = j + l_min;
             
             double term1 = m_aw * m_aw * Clebsch::integral_x_sqr(m_s, p, q, m_m);
             double term2 = -2.0 * m_aw * (double)m_s * Clebsch::integral_x(m_s, p, q, m_m);
             double term3 = (i == j) ? - (double)(p * (p + 1)) : 0.0;
             
             M(i, j) = term1 + term2 + term3;
         }
     }
     
     // 求解
     Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(M);
     if (solver.info() != Eigen::Success) {
         std::cerr << "SWSH Eigenvalue solver failed!" << std::endl;
         return;
     }
     
     // 提取对应于 l 的特征值 (排序后倒数第 l-l_min+1 个)
     int index = K - 1 - (m_l - l_min);
     
     if (index < 0) {
         std::cerr << "Error: Matrix truncation too small for l=" << m_l << std::endl;
         return;
     }
     
     // GremlinEq 约定: E = -eigenvalue
     m_E = -solver.eigenvalues()[index];
     
     // 计算物理特征值 Lambda (分离常数)
     m_lambda = m_E - 2.0 * m_m * m_aw + m_aw * m_aw - m_s * (m_s + 1);
 
     // 提取特征向量 (展开系数 b_k)
     Eigen::VectorXd b_vec = solver.eigenvectors().col(index);
     
     // 确定符号 (使绝对值最大的系数为正)
     int max_ind = 0;
     double max_val = 0.0;
     for(int i=0; i<K; ++i) {
         if(std::abs(b_vec[i]) > max_val) {
             max_val = std::abs(b_vec[i]);
             max_ind = i;
         }
     }
     double sign = (b_vec[max_ind] > 0) ? 1.0 : -1.0;
     
     m_b.resize(K);
     for(int i=0; i<K; ++i) {
         m_b[i] = b_vec[i] * sign;
     }
 }
 
 // 评估 S(x)
 Complex SWSH::evaluate_S(double x) const {
     // 谱求和: S = e^{aw x} * sum(b_k * sY_{l_min+k})
     int l_min = std::max(std::abs(m_s), std::abs(m_m));
     Complex sum = 0.0;
     
     for (size_t k = 0; k < m_b.size(); ++k) {
         int l_curr = l_min + k;
         double y_val = spin_weighted_Y(m_s, l_curr, m_m, x);
         sum += m_b[k] * y_val;
     }
     
     return std::exp(m_aw * x) * sum;
 }
 
 // 评估 L2dag S (使用高精度数值差分)
 Complex SWSH::evaluate_L2dag_S(double x) const {
     double st = std::sqrt(1.0 - x*x);
     if (st < 1e-10) st = 1e-10; 
     
     double h = 1e-6;
     Complex S_plus = evaluate_S(x + h);
     Complex S_minus = evaluate_S(x - h);
     Complex dS_dx = (S_plus - S_minus) / (2.0 * h);
     
     // dS/dtheta = -sin(theta) * dS/dx
     Complex dS_dth = -st * dS_dx;
     
     // L2dag 定义依赖于 spin weight s
     double Q = (double)m_m / st - (double)m_s * x / st;
     
     return dS_dth - Q * evaluate_S(x);
 }
 
 // 评估 L1dag (L2dag S)
 Complex SWSH::evaluate_L1dag_L2dag_S(double x) const {
     double h = 1e-6;
     Complex v_plus = evaluate_L2dag_S(x + h);
     Complex v_minus = evaluate_L2dag_S(x - h);
     Complex dv_dx = (v_plus - v_minus) / (2.0 * h);
     
     double st = std::sqrt(1.0 - x*x);
     Complex dv_dth = -st * dv_dx;
     
     // L1dag 作用于 spin weight 为 s+1 的场
     int s_new = m_s + 1;
     double Q = (double)m_m / st - (double)s_new * x / st;
     
     return dv_dth - Q * evaluate_L2dag_S(x);
 }