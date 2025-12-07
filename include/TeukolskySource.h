/**
 * @file TeukolskySource.h
 * @brief 计算 Teukolsky 方程的源项
 * 对应 GremlinEq/src/circeq/CEKR.h 和 src/utility/RRGW.h
 * * 设计理念：
 * 底层接口支持通用轨道 (r, theta, vr, vtheta)，
 * 上层接口针对 Circular Equatorial 进行封装。
 */

 #ifndef TEUKOLSKY_SOURCE_H
 #define TEUKOLSKY_SOURCE_H
 
 #include <complex>
 #include "KerrGeo.h"
 #include "TeukolskyRadial.h"
 #include "SWSH.h"
 
 using Complex = std::complex<double>;
 
 class TeukolskySource {
 public:
     // 构造函数：绑定物理背景
     TeukolskySource(const KerrGeo& geo, 
                     const TeukolskyRadial& radial,
                     const SWSH& angular);
 
     // ==========================================================
     // 1. 通用轨道接口 (General Orbit Interface)
     // ==========================================================
 
     // 存储应力能量张量在 NP 标架上的投影系数
     struct SourceProjections {
         Complex C_nn;
         Complex C_mn_bar;     // C_{m_bar n}
         Complex C_mm_bar_bar; // C_{m_bar m_bar}
         Complex rho;          // NP scalar rho = -1/(r - i a cos(theta))
     };
 
     /**
      * @brief 计算给定状态下的源项投影
      * 适用于任意轨道 (Circular, Eccentric, Inclined)
      * @param r 径向坐标
      * @param z cos(theta)
      * @param ur 四速度 u^r = dr/dtau
      * @param uz 四速度 u^z = d(cos(theta))/dtau = -sin(theta) * dtheta/dtau
      */
     SourceProjections calc_source_projections(double r, double z, double ur, double uz) const;
 
     // ==========================================================
     // 2. 圆形赤道轨道专用 (Circular Equatorial Specific)
     // ==========================================================
 
     /**
      * @brief 计算圆形赤道轨道的 Teukolsky 振幅 Z
      * 这一步将源项投影与径向/角向解结合，得到最终的 Z_infinity 和 Z_horizon
      * 结果存储在内部变量中
      */
     void compute_circular_amplitudes();
 
     Complex get_Z_inf() const { return m_Z_inf; }
     Complex get_Z_hor() const { return m_Z_hor; }
 
     // 获取计算出的通量 (基于 Z_inf 和 Z_hor)
     double flux_energy_inf() const;
     double flux_angular_momentum_inf() const;
     double flux_energy_hor() const;
     double flux_angular_momentum_hor() const;
 
 private:
     const KerrGeo& m_geo;
     const TeukolskyRadial& m_radial;
     const SWSH& m_angular;
 
     // 计算结果
     Complex m_Z_inf;
     Complex m_Z_hor;
 
     // 内部辅助：计算 Zed 因子 (GremlinEq/src/circeq/CEKR.cc :: Zed)
     // 将径向解 R 与源项系数 A 组合
     Complex calc_Zed(const Complex& R, const Complex& dR, const Complex& d2R, 
                      const SourceProjections& src) const;
 
     // 内部辅助：计算源项系数 A (GremlinEq/src/circeq/CEKR.cc :: Ann0 etc.)
     struct SourceCoeffsA {
         Complex A_nn0, A_nm_bar0, A_mm_bar_bar0;
         Complex A_nm_bar1, A_mm_bar_bar1;
         Complex A_mm_bar_bar2;
     };
     SourceCoeffsA calc_A_coeffs(const SourceProjections& src) const;
 };
 
 #endif // TEUKOLSKY_SOURCE_H