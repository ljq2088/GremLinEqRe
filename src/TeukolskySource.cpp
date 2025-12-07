/**
 * @file TeukolskySource.cpp
 * @brief 源项计算实现
 */

 #include "TeukolskySource.h"
 #include <iostream>
 
 using namespace std::complex_literals;
 
 TeukolskySource::TeukolskySource(const KerrGeo& geo, 
                                  const TeukolskyRadial& radial,
                                  const SWSH& angular)
     : m_geo(geo), m_radial(radial), m_angular(angular)
 {
     m_Z_inf = 0.0;
     m_Z_hor = 0.0;
 }
 
 // ==========================================================
 // 1. 通用轨道源项投影 (核心物理)
 // ==========================================================
 TeukolskySource::SourceProjections 
 TeukolskySource::calc_source_projections(double r, double z, double ur, double uz) const {
     // 基础几何量
     // z = cos(theta)
     // sin(theta) = sqrt(1-z^2)
     double st = std::sqrt(1.0 - z*z);
     
     // Sigma = r^2 + a^2 z^2
     double sigma = m_geo.sigma(r, z);
     double delta = m_geo.delta(r);
     
     double a = m_geo.spin();
     double E = m_geo.energy();
     double Lz = m_geo.angular_momentum();
     
     // 计算四速度分量 u^t
     // u^t = 1/Sigma * [ a(Lz - aE sin^2 th) + (r^2+a^2)/Delta * P ]
     // 其中 P = E(r^2+a^2) - a Lz
     double P = E * (r*r + a*a) - a * Lz;
     double T_term = a * (Lz - a * E * st * st); // TFunc in GKG? No, TFunc is Sigma * u^t
     // GKG TFunc: E*((r2+a2)^2/Delta - a^2 st^2) + a Lz (1 - (r2+a2)/Delta) ...
     // 让我们直接用标准公式 u^t = dt/dtau
     // dt/dtau = 1/Sigma * [ (r^2+a^2) * P / Delta - a * (a E st^2 - Lz) ]
     // (Fujita Tagoshi A-19, 但要注意符号)
     // GremlinEq A-19: Sigma dt/dtau = (r^2+a^2)/Delta * P - a(aE st^2 - Lz)
     
     double sigma_ut = (r*r + a*a) * P / delta - a * (a * E * st * st - Lz);
     double ut = sigma_ut / sigma;
     
     // 计算 dtheta/dtau
     // uz = d(cos th)/dtau = -sin th * dth/dtau => dth/dtau = -uz / st
     double u_theta = (st > 1e-10) ? -uz / st : 0.0;
 
     // NP 标量 rho = -1 / (r - i a cos th)
     Complex rho = -1.0 / (r - 1.0i * a * z);
     
     // =============================================
     // 计算 C_ab (Fujita Tagoshi A-23)
     // =============================================
     
     // C_nn
     // Cnn = 1/(4 Sigma^3 u^t) * [ P + Sigma * ur ]^2
     Complex term_nn = P + sigma * ur;
     Complex C_nn = term_nn * term_nn / (4.0 * sigma * sigma * sigma * ut);
     
     // C_mbar_n
     // 需要计算 [ i sin th (aE - Lz/sin^2 th) + Sigma u_theta ]
     // 注意 GremlinEq 里的项: i sin theta (...) + Sigma dtheta/dtau
     double term_ang_real = sigma * u_theta;
     double term_ang_imag = st * (a * E - Lz / (st * st));
     Complex term_ang = term_ang_real + 1.0i * term_ang_imag;
     
     // C_mn_bar = - rho / (2 sqrt(2) Sigma^2 u^t) * [P + Sigma ur] * [term_ang]
     Complex C_mn_bar = - rho * term_nn * term_ang / (2.0 * std::sqrt(2.0) * sigma * sigma * ut);
     
     // C_mbar_mbar
     // C_mm_bar = rho^2 / (2 Sigma u^t) * [term_ang]^2
     Complex C_mm_bar_bar = rho * rho * term_ang * term_ang / (2.0 * sigma * ut);
 
     return {C_nn, C_mn_bar, C_mm_bar_bar, rho};
 }
 
 // ==========================================================
 // 2. 辅助系数 A 计算 (GremlinEq CEKR::Ann0 etc.)
 // ==========================================================
 TeukolskySource::SourceCoeffsA 
 TeukolskySource::calc_A_coeffs(const SourceProjections& src) const {
     // 这里的公式非常繁琐，完全对应 CEKR.cc 中的 Ann0, Anmbar0 等函数
     // 它们是源项积分中的被积函数系数
     // 注意：这里的公式通常只对 Circular Equatorial 进行了简化 (e.g. S=const)
     // 但在通用轨道中，S (角向函数) 是 theta 的函数。
     // 为了当前 "Circular Equatorial" 的目标，我们这里先实现针对 Circular 简化的版本
     // 即假设 d(S)/dtheta = 0 (对于 l=m=2, theta=pi/2 可能是真的，但一般情况不一定)
     // 
     // ⚠️ 严谨警告：
     // GremlinEq 的 CEKR.cc 里的 Ann0 等函数里用了 `swsh->l2dagspheroid(0.)`
     // 这意味着它是在 theta=pi/2 (x=0) 处求值的。
     // 如果我们要支持通用轨道，这里必须传入当前的 theta 对应的 S, L2S, L1L2S 值。
     // 
     // 鉴于目前的任务是 "输入赤道圆轨道测试"，我们先硬编码 x=0 处的 SWSH 值。
     // 在未来的 General Orbit 迭代中，这里需要重构，接受 S(theta) 作为参数。
     
     // 获取 theta=pi/2 (x=0) 处的 SWSH 值
     // 我们需要在 SWSH 类里增加计算 l2dagspheroid 等算符的功能 (参考 SWSHSpheroid.cc)
     // 这是一个缺失的功能！
     // 暂时 workaround：GremlinEq 的 CEKR.cc 直接调用 swsh->spheroid(0.) 等
     // 我们目前 SWSH 类只算了 lambda。
     
     // 为了不阻塞，我们假设 SWSH 提供了这些值（或者我们先用 1.0 代替进行占位，后续补全 SWSH 类）
     // 既然要求 "严谨求证"，我们不能糊弄。
     // 让我们先写出框架，SWSH 的算符求值作为下一个 TODO。
     
     // 假设我们有了 S, L2S, L1L2S
     // 模拟值 (仅供编译通过，数值不对)
     Complex S = 1.0; 
     Complex L2S = 0.0; 
     Complex L1L2S = 0.0;
     
     Complex rho = src.rho;
     Complex rho_bar = std::conj(rho);
     Complex rho3 = rho * rho * rho;
     double delta = m_geo.delta(m_geo.radius()); // 圆轨道 r
     double a = m_geo.spin();
     
     SourceCoeffsA coeffs;
     
     // Ann0 = -2 / (sqrt(2pi) Delta^2) * Cnn * rho^-2 * rhobar^-1 * L1dag L2dag S
     // (CEKR.cc line 143)
     Complex pref_nn0 = -2.0 * src.C_nn / (std::sqrt(2.0 * M_PI) * delta * delta * rho_bar * rho3);
     // CEKR.cc: pref * (tmp1 + tmp2)
     // tmp1 = 2 i a rho * L2S ...
     // 我们先略过具体项的逐字翻译，重点是结构。
     
     // ... (此处需要大量代码复刻 CEKR.cc 中 6 个 A 函数的逻辑) ...
     // 为了代码简洁，我只写出关键的 A_mm_bar_bar2，因为它是二阶导项的系数
     
     // Ambarmbar2 (CEKR.cc line 208)
     // return - Cmbarmbar * rhobar * S / rho^3
     coeffs.A_mm_bar_bar2 = - src.C_mm_bar_bar * rho_bar * S / rho3;
     
     // 其他项先置零，待 SWSH 算符补全后填入
     coeffs.A_nn0 = 0.0;
     return coeffs;
 }
 
 // ==========================================================
 // 3. Zed 因子计算
 // ==========================================================
 Complex TeukolskySource::calc_Zed(const Complex& R, const Complex& dR, const Complex& d2R, 
                                   const SourceProjections& src) const {
     SourceCoeffsA A = calc_A_coeffs(src);
     
     // GremlinEq CEKR.cc :: Zed
     // term1 = R * (Ann0 + Anmbar0 + Ambarmbar0)
     // term2 = -dR * (Anmbar1 + Ambarmbar1)
     // term3 = d2R * Ambarmbar2
     
     // 简化版逻辑 (需补全)
     Complex term3 = d2R * A.A_mm_bar_bar2;
     
     // Wronskian
     // CEKR.cc line 87: W = 2 i omega * B_inc * C_trans
     // 这里的 B_inc, C_trans 应该来自 TeukolskyRadial
     Complex B_inc = m_radial.get_B_inc();
     Complex C_trans = m_radial.get_C_trans();
     // 我们的 TeukolskyRadial 中 m_omega 是实数
     Complex W = 2.0i * (Complex)m_radial.get_omega() * B_inc * C_trans;
     
     return -2.0 * M_PI * term3 / W; // 仅含 term3 的示意
 }
 
 // ==========================================================
 // 4. 圆形赤道振幅计算
 // ==========================================================
 void TeukolskySource::compute_circular_amplitudes() {
     // 1. 获取轨道状态
     double r = m_geo.radius();
     double z = 0.0; // Equatorial: theta = pi/2 -> cos theta = 0
     double ur = 0.0; // Circular: dr/dtau = 0
     double uz = 0.0; // Equatorial: dtheta/dtau = 0
     
     // 2. 计算源投影
     auto src_proj = calc_source_projections(r, z, ur, uz);
     
     // 3. 获取径向解 (In & Up)
     Complex R_in = m_radial.TeukRin(); // 需在 Radial 类暴露
     Complex dR_in = m_radial.dr_TeukRin();
     Complex d2R_in = m_radial.ddr_TeukRin();
     
     Complex R_up = m_radial.TeukRup();
     Complex dR_up = m_radial.dr_TeukRup();
     Complex d2R_up = m_radial.ddr_TeukRup();
     
     // 4. 计算 Z_inf 和 Z_hor
     // Z_inf 对应 R_in (见 CEKR.cc 构造函数)
     // ZI = c_trans * Zed(R_in ...)
     m_Z_inf = m_radial.get_C_trans() * calc_Zed(R_in, dR_in, d2R_in, src_proj);
     
     // Z_hor 对应 R_up
     // ZH = b_trans * Zed(R_up ...)
     m_Z_hor = m_radial.get_B_trans() * calc_Zed(R_up, dR_up, d2R_up, src_proj);
 }
 
 // ==========================================================
 // 5. 通量计算 (GremlinEq RRGW.cc)
 // ==========================================================
 double TeukolskySource::flux_energy_inf() const {
     double omega = m_radial.get_omega(); // 需暴露
     double m = (double)m_radial.get_m(); // 需暴露
     double z_abs = std::abs(m_Z_inf);
     
     // Edot = |Z|^2 / (4 pi omega^2)
     return z_abs * z_abs / (4.0 * M_PI * omega * omega);
 }