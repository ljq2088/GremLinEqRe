#include "SWSH.h"
#include "Clebsch.h"
#include <Eigen/Dense>
#include <iostream>
#include <algorithm>

SWSH::SWSH(int s, int l, int m, double a_omega)
    : m_s(s), m_l(l), m_m(m), m_aw(a_omega)
{
    solve_eigenvalue();
}

void SWSH::solve_eigenvalue() {
    // 确定基底范围
    int l_min = std::max(std::abs(m_s), std::abs(m_m));
    
    // 矩阵截断大小：足够大以保证收敛
    // l 模态通常位于第 (l - l_min) 个特征值，往后多取 30 个基底通常足够
    int K = (m_l - l_min) + 30;
    
    Eigen::MatrixXd M(K, K);
    M.setZero();
    
    // 构建矩阵 (参考 GremlinEq/src/swsh/SWSHSpheroid.cc load_M)
    for (int i = 0; i < K; ++i) {
        int p = i + l_min;
        for (int j = 0; j < K; ++j) {
            int q = j + l_min;
            
            // GremlinEq 矩阵元素公式:
            // M_pq = (aw)^2 <p|x^2|q> - 2aw s <p|x|q> - p(p+1) delta_pq
            double term1 = m_aw * m_aw * Clebsch::integral_x_sqr(m_s, p, q, m_m);
            double term2 = -2.0 * m_aw * (double)m_s * Clebsch::integral_x(m_s, p, q, m_m);
            double term3 = (i == j) ? - (double)(p * (p + 1)) : 0.0;
            
            M(i, j) = term1 + term2 + term3;
        }
    }
    
    // 求解特征值
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(M);
    if (solver.info() != Eigen::Success) {
        std::cerr << "SWSH Eigenvalue solver failed!" << std::endl;
        return;
    }
    
    // 特征值排序与提取
    // GremlinEq 矩阵对角线包含 -p(p+1)，因此特征值是负数，且绝对值随 p 增大。
    // Eigen 从小到大排序 (val[0] 最负 -> val[K-1] 最接近0)。
    // 基态 (l = l_min) 对应 p(p+1) 最小，即 -p(p+1) 最大，对应 Eigen 的最后一个值。
    // 激发态 l 对应 Eigen 倒数第 (l - l_min + 1) 个值。
    
    int index = K - 1 - (m_l - l_min);
    
    if (index < 0) {
        std::cerr << "Error: Matrix truncation too small for l=" << m_l << std::endl;
        return;
    }
    
    // GremlinEq 约定: E = - eigenvalue
    m_E = -solver.eigenvalues()[index];
    
    // 物理特征值 Lambda (Teukolsky 方程中的分离常数)
    // Formula: lambda = E - 2*m*aw + aw*aw - s(s+1)
    m_lambda = m_E - 2.0 * m_m * m_aw + m_aw * m_aw - m_s * (m_s + 1);
}