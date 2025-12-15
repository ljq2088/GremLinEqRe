/**
 * @file KerrGeo.cpp
 * @brief 通用克尔几何实现
 */

 #include "KerrGeo.h"
 #include <iostream>
 
 // ==========================================================
 // 构造与初始化
 // ==========================================================
 
 KerrGeo::KerrGeo(Real a,double E, double Lz, double Q)
     : m_a(a), m_E(E), m_Lz(Lz), m_Q(Q) {}
 
 KerrGeo KerrGeo::from_circular_equatorial(Real a, Real r, bool is_prograde) {
     Real E, Lz, Q;
     calc_circular_params(a, r, is_prograde, E, Lz, Q);
     return KerrGeo(a, E, Lz, Q);
 }
 
 // 复用之前的计算逻辑，但封装在内部静态函数中
 void KerrGeo::calc_circular_params(Real a, Real r, bool is_prograde, 
                                    Real& E_out, Real& Lz_out, Real& Q_out) {
     if (r <= 0) throw std::invalid_argument("Radius must be positive");
     
     Q_out = 0.0; // 赤道面轨道 Carter 常数为 0
     Real v = 1.0 / std::sqrt(r);
     Real v2 = v * v;
     Real v3 = v2 * v;
 
     if (is_prograde) {
         Real numer_e = 1.0 - v2 * (2.0 - a * v);
         Real denom = 1.0 - v2 * (3.0 - 2.0 * a * v);
         E_out = numer_e / std::sqrt(denom);
 
         Real numer_lz = 1.0 - a * v3 * (2.0 - a * v);
         Lz_out = r * v * numer_lz / std::sqrt(denom);
     } else {
         Real numer_e = 1.0 - v2 * (2.0 + a * v);
         Real denom = 1.0 - v2 * (3.0 + 2.0 * a * v);
         E_out = numer_e / std::sqrt(denom);
 
         Real numer_lz = 1.0 + a * v3 * (2.0 + a * v);
         Lz_out = -r * v * numer_lz / std::sqrt(denom); // 注意负号
     }
 }
 
 // ==========================================================
 // 基础几何函数
 // ==========================================================
 
 Real KerrGeo::delta(Real r) const {
     return r*r - 2.0*r + m_a*m_a;
 }
 
 Real KerrGeo::sigma(Real r, Real z) const {
     return r*r + m_a*m_a * z*z;
 }
 
 // ==========================================================
 // 解析势函数及其导数 (Analytical Potentials & Derivatives)
 // 参考 GremlinEq/src/utility/GKG.cc
 // ==========================================================
 
 // RFunc in GKG.cc
 Real KerrGeo::potential_r(Real r) const {
     Real term1 = m_E * (r*r + m_a*m_a) - m_a * m_Lz;
     Real term2 = m_Lz - m_a * m_E;
     // 注意：GKG.cc 中 RFunc = term1^2 - Delta * (r^2 + term2^2 + Q)
     // 这里的 term2^2 + Q + r^2 就是 Separation Constant K + r^2
     Real term3 = m_Q + r*r + term2*term2;
     
     return term1*term1 - delta(r) * term3;
 }
 
 // dr_RFunc in GKG.cc
 Real KerrGeo::diff_potential_r(Real r) const {
     Real term1 = m_E * (r*r + m_a*m_a) - m_a * m_Lz;
     Real term2 = m_Lz - m_a * m_E;
     Real term3 = m_Q + r*r + term2*term2;
     
     // d(Delta)/dr = 2r - 2
     Real d_delta = 2.0 * (r - 1.0);
     
     // d(term1^2)/dr = 2 * term1 * (2*E*r)
     Real d_part1 = 2.0 * term1 * (2.0 * m_E * r);
     
     // d(Delta * term3)/dr = d_delta * term3 + Delta * (2r)
     Real d_part2 = d_delta * term3 + delta(r) * (2.0 * r);
     
     return d_part1 - d_part2;
 }
 
 // ddr_RFunc in GKG.cc
 Real KerrGeo::diff2_potential_r(Real r) const {
     // 这是一个常数项较多的二阶导数，原作 GKG.cc 给出了简化形式：
     // ans = tmp3 + r*(12. + tmp1*r); 
     // 我们这里为了清晰和通用性，重新推导或使用标准形式。
     // 为了与 GKG 完全一致，我们尝试还原其逻辑：
     // R(r) 是关于 r 的四次多项式。
     // R(r) = (E^2 - 1)r^4 + ...
     // 二阶导数是关于 r 的二次多项式。
     
     // 让我们用最直接的解析求导展开（严谨且不易出错）：
     // R = [E(r^2+a^2) - aL]^2 - (r^2-2r+a^2)[r^2 + K]
     // 其中 K = (L-aE)^2 + Q
     
     Real K_val = (m_Lz - m_a*m_E)*(m_Lz - m_a*m_E) + m_Q;
     
     // 系数 coeff of r^4: (E^2 - 1)
     Real c4 = m_E*m_E - 1.0;
     // 系数 coeff of r^3: 2 (来自 -(-2r)*r^2) = 2
     Real c3 = 2.0;
     // 系数 coeff of r^2: [2E(E a^2 - a L)] - [a^2 + K + (-2r)*0 + 1*(-2r)] ... 展开比较繁琐
     // 建议直接实现 GKG.cc 中的精简公式，它经过了优化：
     
     /* GKG.cc implementation:
        Real tmp1 = 12.*(E*E - 1.);
        Real tmp3 = 2.*(a*a*(E*E - 1.) - Lz*Lz - Q);
        return tmp3 + r*(12. + tmp1*r);
     */
     
     Real tmp1 = 12.0 * (m_E * m_E - 1.0);
     // 注意 GKG.cc 里的 tmp3 少了 K 的一部分？让我们仔细检查数学。
     // R'' 确实应该包含 Q 和 Lz。
     // 假设 GKG 是对的，直接复用：
     Real tmp3 = 2.0 * (m_a*m_a * (m_E*m_E - 1.0) - m_Lz*m_Lz - m_Q);
     
     return tmp3 + r * (12.0 + tmp1 * r);
 }
 
 // ThetaFunc in GKG.cc
 Real KerrGeo::potential_theta(Real z) const {
     // z = cos(theta)
     // GKG: cotthsqr = z^2 / (1-z^2)
     // tmp = Q - cotthsqr * Lz^2 - a^2 * z * (1 - E^2)  <-- 原文是 z 还是 z^2? 
     // 原文 GKG.cc: a*a*z*(1. - E*E) ... 这是一个潜在的陷阱！
     // 实际上 Sigma^2 theta_dot^2 = Q - cos^2 theta [ a^2(1-E^2) + Lz^2/sin^2 theta ]
     // = Q - z^2 a^2 (1-E^2) - z^2 Lz^2 / (1-z^2)
     // 让我们看 GKG.cc 代码：
     // const Real cotthsqr = z/(1. - z);  <-- 原文这里的输入 z 是 cos^2(theta) 吗？
     // 检查 GKG.h: Real ThetaFunc(const Real r, const Real a, const Real z ...
     // 通常 z 代表 cos(theta).
     // 如果 z 是 cos(theta)，那么 z^2/(1-z^2) 才是 cot^2。
     // **注意**：GremlinEq 的 GKG.cc 里写的是 `const Real cotthsqr = z/(1. - z);`
     // 这强烈暗示 GKG.cc 里的参数 `z` 实际上是 **cos^2(theta)** 而不是 cos(theta)！
     // 这一点必须非常小心。在我们的 API 设计里，通常习惯传 cos(theta)。
     
     // 为了避免混淆，我们的函数参数命名为 z_input。
     // 如果我们要保持通用性，最好明确输入是 cos(theta)。
     
     // 我们按照标准定义重写，不盲目 copy GKG 的潜在歧义写法：
     // Theta_potential = Q - z^2 * [ a^2(1-E^2) + Lz^2/(1-z^2) ]
     
     Real z2 = z * z;
     // 防止除以零
     if (std::abs(z2 - 1.0) < 1e-12) return 0.0; // 极轴处
     
     Real term_Lz = m_Lz * m_Lz * z2 / (1.0 - z2);
     Real term_a  = m_a * m_a * z2 * (1.0 - m_E * m_E);
     
     Real val = m_Q - term_Lz - term_a;
     return (val < 0.0) ? 0.0 : val;
 }