// #ifndef SWSH_H
// #define SWSH_H

// #include <vector>
// #include <complex>


// using Complex = std::complex<double>;
// class SWSH {
// public:
//     /**
//      * @brief 构造函数
//      * @param s 自旋权重 (通常 -2)
//      * @param l 谐波指数 l
//      * @param m 谐波指数 m
//      * @param a_omega 无量纲频率参数 (a * omega)
//      */
//     SWSH(int s, int l, int m, double a_omega);

//     // 获取物理特征值 Lambda (用于径向方程)
//     double get_lambda() const { return m_lambda; }
    
//     // 获取本征值 E (用于与文献数据表比对)
//     // E = lambda + s(s+1) - (aw)^2 + 2*m*aw
//     double get_E() const { return m_E; }
//     /**
//      * @brief 计算自旋加权球/椭球谐函数 S_lm(theta) 的值
//      * @param x = cos(theta)
//      */
//     Complex evaluate_S(double x) const;

//     /**
//      * @brief 计算 Teukolsky 算符 L2dag S
//      * L2dag = partial_theta - m/sin(theta) - s*cot(theta) (针对 s=-2 等定义)
//      * 具体定义需参考 Teukolsky 方程
//      * @param x = cos(theta)
//      */
//     Complex evaluate_L2dag_S(double x) const;
    
//     // 计算 L1dag (L2dag S) -> 二阶算符，用于源项计算
//     Complex evaluate_L1dag_L2dag_S(double x) const;

// private:
//     int m_s, m_l, m_m;
//     double m_aw;
//     double m_E;
//     double m_lambda;
//     // 谱展开系数 (已在构造函数中计算)
//     std::vector<double> m_b;
//     void solve_eigenvalue();
//     // === 内部辅助 ===
//     // 计算自旋加权球谐函数 sYlm(x)
//     // 使用 Wigner-d 递归
//     static double spin_weighted_Y(int s, int l, int m, double x);
    
//     // 计算 sYlm 对 theta 的导数
//     static double spin_weighted_Y_deriv(int s, int l, int m, double x);
// };

// #endif // SWSH_H
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

    // 获取本征值 E (Teukolsky 角向本征值)
    // 对应方程: [ (d/dx)(1-x^2)d/dx + (aw x)^2 - 2aw s x + s + A - (m+sx)^2/(1-x^2) ] S = 0
    // E = -A
    double get_E() const { return m_E; }
    
    // 获取物理分离常数 Lambda (用于 Radial Teukolsky 方程)
    // Lambda = E - 2*m*aw + (aw)^2 - s(s+1)
    double get_lambda() const { return m_lambda; }
    double get_aw() const { return m_aw; }
    double get_m() const { return m_m; }
    double get_s() const { return m_s; }
    double get_l() const { return m_l; }
    /**
     * @brief 计算自旋加权球谐函数 S_lm(theta)
     * 方法：Jacobi 多项式递归 + 谱求和
     * 复杂度：O(K)
     */
    Complex evaluate_S(double x) const;

    /**
     * @brief 计算一阶 Teukolsky 算符 L_{-2}^dag S
     * L^dag = eth + a*omega*sin(theta)
     * 方法：解析提升算符代数 (Exact Spectral Operator)
     * 结果自旋权重：s + 1
     */
    Complex evaluate_L2dag_S(double x) const;
    
    /**
     * @brief 计算二阶算符 L_{-1}^dag L_{-2}^dag S
     * 方法：解析二阶算符展开
     * 结果自旋权重：s + 2 (通常用于 s=-2 -> s=0 的源项)
     */
    Complex evaluate_L1dag_L2dag_S(double x) const;

    // 为了兼容旧测试代码保留的静态接口 (内部调用高效实现)
    static double spin_weighted_Y(int s, int l, int m, double x);
    // 废弃接口，仅占位
    // static double spin_weighted_Y_deriv(int s, int l, int m, double x);

private:
    int m_s, m_l, m_m;
    double m_aw;
    double m_E;
    double m_lambda;
    
    // 谱展开系数 (在构造函数中计算并归一化)
    std::vector<double> m_b;
    // Evaluate S(θ), ∂θ S, ∂θ^2 S at x = cosθ, consistent with LRR conventions
    void evaluate_S_theta_derivs(double x, Complex& S, Complex& dS_dth, Complex& d2S_dth2) const;
    void solve_eigenvalue();
};

#endif // SWSH_H