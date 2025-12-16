#ifndef SWSH_H
#define SWSH_H

#include <vector>
#include <complex>


using Complex = std::complex<double>;
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


    
    // 获取本征值 E (用于与文献数据表比对)
    // E = lambda + s(s+1) - (aw)^2 + 2*m*aw
    double get_E() const { return m_E; }
    double get_lambda() const { return m_lambda; }
    /**
     * @brief 计算自旋加权球/椭球谐函数 S_lm(theta) 的值
     * @param x = cos(theta)
     */
    Complex evaluate_S(double x) const;

    /**
     * @brief 计算 Teukolsky 算符 L2dag S
     * L2dag = partial_theta - m/sin(theta) - s*cot(theta) (针对 s=-2 等定义)
     * 具体定义需参考 Teukolsky 方程
     * @param x = cos(theta)
     */
    Complex evaluate_L2dag_S(double x) const;
    
    // 计算 L1dag (L2dag S) -> 二阶算符，用于源项计算
    Complex evaluate_L1dag_L2dag_S(double x) const;

private:
    int m_s, m_l, m_m;
    double m_aw;
    double m_E;
    double m_lambda;
    // 谱展开系数 (已在构造函数中计算)
    std::vector<double> m_b;
    void solve_eigenvalue();
    // === 内部辅助 ===
    // 计算自旋加权球谐函数 sYlm(x)
    // 使用 Wigner-d 递归
    static double spin_weighted_Y(int s, int l, int m, double x);
    
    // 计算 sYlm 对 theta 的导数
    static double spin_weighted_Y_deriv(int s, int l, int m, double x);
};

#endif // SWSH_H