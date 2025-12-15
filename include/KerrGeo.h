/**
 * @file KerrGeo.h
 * @brief 通用克尔黑洞测地线几何核心
 * * 对应 GremlinEq 的:
 * - src/circeq/CEKG.cc (圆形轨道参数)
 * - src/utility/GKG.cc (通用势函数及其解析导数)
 */

 #ifndef KERRGEO_H
 #define KERRGEO_H
 
 #include <cmath>
 #include <stdexcept>
 #include <vector>
 
 using Real = double;
 
 class KerrGeo {
 public:
    // ==========================================================
    // 构造函数
    // ==========================================================
    // ==========================================================
    // 粒子状态结构体 (State)
    // ==========================================================
    /**
     * @brief 存储粒子在测地线演化过程中的瞬时状态
     * 包含 4-位置 和 4-速度
     */
    struct State {
        std::vector<double> x; // 4-position: t, r, theta, phi
        std::vector<double> u; // 4-velocity: ut, ur, uth, uphi
        
        // 构造函数初始化，避免未定义行为
        State() : x(4, 0.0), u(4, 0.0) {}
    };

    // [修改] 构造函数需要存储守恒量
    KerrGeo(Real a, double E, double Lz, double Q);


   
    
 
     /**
      * @brief 辅助构造函数：圆形赤道轨道 (Circular Equatorial)
      * 内部会自动计算对应的 E, Lz, 并设置 Q=0
      */
     static KerrGeo from_circular_equatorial(Real a, Real r, bool is_prograde);
 
     // TODO: 未来可以添加 from_eccentric_inclined(a, p, e, x)
 
     // ==========================================================
     // 获取守恒量
     // ==========================================================
     Real spin() const { return m_a; }
     double energy() const { return m_E; }
     double angular_momentum() const { return m_Lz; }
     double carter_constant() const { return m_Q; }
 
     // ==========================================================
     // 势函数及其解析导数 (Analytical Potentials & Jacobians)
     // 对应 GremlinEq/src/utility/GKG.cc
     // 这些是计算 Teukolsky 源项的核心输入
     // ==========================================================
 
     /**
      * @brief 径向势函数 R(r)
      * R(r) = [E(r^2+a^2) - a Lz]^2 - Delta * [ (Lz - aE)^2 + Q + r^2 ]
      * (注意：原文 GKG 中实际上计算的是 Sigma^2 * r_dot^2)
      */
     Real potential_r(Real r) const;
 
     /**
      * @brief 径向势的一阶导数 dR/dr (解析形式)
      */
     Real diff_potential_r(Real r) const;
 
     /**
      * @brief 径向势的二阶导数 d^2R/dr^2 (解析形式)
      */
     Real diff2_potential_r(Real r) const;
 
     /**
      * @brief 角向势函数 Theta(z) where z = cos(theta)
      * Theta(z) = Q - cot^2(th) * Lz^2 - a^2 * z^2 * (1-E^2) 
      * 注意：GKG 实现中对 Theta 的定义略有不同，我们需要严格对齐原文
      */
     Real potential_theta(Real z) const;
 
     // 辅助几何函数
     Real delta(Real r) const;      // Delta = r^2 - 2r + a^2
     Real sigma(Real r, Real z) const; // Sigma = r^2 + a^2 z^2
 
 private:
     Real m_a;  // Spin
     Real m_E;  // Energy
     Real m_Lz; // Axial Angular Momentum
     Real m_Q;  // Carter Constant
 
     // 内部使用的圆形轨道计算函数 (复用之前的逻辑)
     static void calc_circular_params(Real a, Real r, bool is_prograde, 
                                      Real& E, Real& Lz, Real& Q);
 };
 
 #endif // KERRGEO_H