#ifndef SWSH_H
#define SWSH_H

#include <vector>
#include <complex>

class SWSH {
public:
    /**
     * @brief 构造函数
     * @param s 自旋权重 (通常 -2)
     * @param l 谐波指数 l
     * @param m 谐波指数 m
     * @param a_omega 无量纲频率参数 (a * omega)
     */
    SWSH(int s, int l, int m, double a_omega);

    // 获取物理特征值 Lambda (用于径向方程)
    double get_lambda() const { return m_lambda; }
    
    // 获取本征值 E (用于与文献数据表比对)
    // E = lambda + s(s+1) - (aw)^2 + 2*m*aw
    double get_E() const { return m_E; }

private:
    int m_s, m_l, m_m;
    double m_aw;
    double m_E;
    double m_lambda;
    
    void solve_eigenvalue();
};

#endif // SWSH_H