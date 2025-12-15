#pragma once

#include <complex>
#include <vector>
#include "KerrGeo.h"
#include "SWSH.h" 
using Complex = std::complex<double>;
class SWSH; 
// 源项投影系数结构体
// 对应径向方程源项 T_lmomega 的展开系数
// T = A_nn0 * delta 
//   + A_mbarn0 * delta + A_mbarn1 * delta' 
//   + A_mbarmbar0 * delta + A_mbarmbar1 * delta' + A_mbarmbar2 * delta''
struct SourceProjections {
    // T_nn 部分 (0阶导数)
    Complex A_nn0;

    // T_mbarn 部分 (0阶, 1阶导数)
    Complex A_mbarn0;
    Complex A_mbarn1;

    // T_mbarmbar 部分 (0阶, 1阶, 2阶导数)
    Complex A_mbarmbar0;
    Complex A_mbarmbar1;
    Complex A_mbarmbar2;
};

class TeukolskySource {
public:
    // 构造函数
    
    TeukolskySource(Real a_spin, Real omega, int s, int l, int m) ;
    /**
     * @brief 计算粒子在当前位置的源项投影系数
     * @param geo_state 粒子状态 (t, r, theta, phi, ut, ur, uth, uphi)
     * @param geo_obj 几何对象 (提供守恒量 E, Lz)
     * @return SourceProjections
     */
    SourceProjections ComputeProjections(const KerrGeo::State& st, const KerrGeo& geo_obj,SWSH& swsh);

private:
    Real m_a;
    double m_omega;
    int m_s;
    int m_l;
    int m_m;
};