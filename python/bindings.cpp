#include <pybind11/complex.h>
#include <pybind11/stl.h>
#include "KerrGeo.h"
#include "TeukolskyRadial.h"
#include "SWSH.h"
#include "TeukolskySource.h"
namespace py = pybind11;

PYBIND11_MODULE(_core, m) {
     m.doc() = "GremLinEqRe C++ Core Module (General Orbits)";
     py::class_<KerrGeo>geo(m,"KerrGeo","克尔几何");
     py::class_<KerrGeo::State>(geo, "State")
          .def(py::init<>())
          .def_readwrite("x", &KerrGeo::State::x)
          .def_readwrite("u", &KerrGeo::State::u)
          .def("__repr__", [](const KerrGeo::State &s) {
              return "<KerrGeo.State t=" + std::to_string(s.x[0]) + "...>";
          });
          geo.def(py::init<Real, double, double, double>(), 
            py::arg("a"), py::arg("E"), py::arg("Lz"), py::arg("Q"))
          
          .def("update_kinematics", &KerrGeo::update_kinematics,
               py::arg("state"), py::arg("r_direction")=0.0, py::arg("theta_direction")=0.0,
               "根据守恒量更新四速度")
          
          // 构造函数 : 圆形赤道
          .def_static("from_circular_equatorial", &KerrGeo::from_circular_equatorial,
               py::arg("a"), py::arg("r"), py::arg("is_prograde"),
               "工厂函数: 从轨道半径创建圆形赤道轨道")

          // 属性
          .def_property_readonly("energy", &KerrGeo::energy)
          .def_property_readonly("angular_momentum", &KerrGeo::angular_momentum)
          .def_property_readonly("carter_constant", &KerrGeo::carter_constant)
          .def_property_readonly("spin", &KerrGeo::spin)

          // 势函数及其解析导数 (Analytical Jacobians)
          .def("potential_r", &KerrGeo::potential_r, "径向势 R(r)")
          .def("diff_potential_r", &KerrGeo::diff_potential_r, "径向势一阶导 dR/dr")
          .def("diff2_potential_r", &KerrGeo::diff2_potential_r, "径向势二阶导 d2R/dr2")
          .def("potential_theta", &KerrGeo::potential_theta, "角向势 Theta(cos_theta)");
     py::class_<AsymptoticAmplitudes>(m, "AsymptoticAmplitudes")
          .def_readwrite("R_in_coef_inf_inc", &AsymptoticAmplitudes::R_in_coef_inf_inc)
          .def_readwrite("R_in_coef_inf_trans", &AsymptoticAmplitudes::R_in_coef_inf_trans);
     py::class_<PhysicalAmplitudes>(m, "PhysicalAmplitudes")
          .def_readwrite("B_trans", &PhysicalAmplitudes::B_trans)
          .def_readwrite("B_inc", &PhysicalAmplitudes::B_inc)
          .def_readwrite("B_ref", &PhysicalAmplitudes::B_ref)
          .def_readwrite("C_trans", &PhysicalAmplitudes::C_trans);
     py::class_<TeukolskyRadial>(m, "TeukolskyRadial")
          .def(py::init<Real, Real, Real, int, int, int, Real>(), 
               py::arg("M"), py::arg("a_spin"), py::arg("omega"), 
               py::arg("s"), py::arg("l"), py::arg("m"), py::arg("lambda"))
          
          .def("continued_fraction", &TeukolskyRadial::continued_fraction,
               py::arg("nu"), py::arg("direction"),
               "计算连分式: direction=+1 (正向), -1 (负向)")
          .def("coeff_alpha", &TeukolskyRadial::coeff_alpha, "MST 系数 alpha_n")
          .def("coeff_beta",  &TeukolskyRadial::coeff_beta,  "MST 系数 beta_n")
          .def("coeff_gamma", &TeukolskyRadial::coeff_gamma, "MST 系数 gamma_n")
          
          .def_static("log_gamma", &TeukolskyRadial::log_gamma)

          .def("solve_nu", &TeukolskyRadial::solve_nu, 
               py::arg("nu_guess"), 
               "求解特征值 nu，使 g(nu)=0")
          .def("k_factor", &TeukolskyRadial::k_factor, 
                    py::arg("nu"), "计算连接因子 K_nu")     
          .def("calc_g", &TeukolskyRadial::calc_g, 
               py::arg("nu"), 
               "计算超越方程残差 g(nu)")
          .def("ComputeSeriesCoefficients", &TeukolskyRadial::ComputeSeriesCoefficients)
          .def("hyp2f1", &TeukolskyRadial::Hyp2F1, 
               "Calculate Gaussian Hypergeometric function 2F1(a, b; c; z) using Arb library.",
               py::arg("a"), py::arg("b"), py::arg("c"), py::arg("z"),
               py::arg("regularized")=false)
          .def("hyp2f1_scaled", &TeukolskyRadial::Hyp2F1_Scaled,
               "Calculate Gaussian Hypergeometric function 2F1(a, b; c; z) using Arb library.",
               py::arg("a"), py::arg("b"), py::arg("c"), py::arg("z"), py::arg("factor"))
          .def("hyp2f1_fully_scaled", &TeukolskyRadial::Hyp2F1_FullyScaled,
               "Calculate Gaussian Hypergeometric function 2F1(a, b; c; z) using Arb library.",
               py::arg("a"), py::arg("b"), py::arg("c"), py::arg("z"), py::arg("factor"), py::arg("log_factor"))
          .def("Evaluate_Hypergeometric", &TeukolskyRadial::Evaluate_Hypergeometric,
                    py::arg("r"), py::arg("nu"), py::arg("a_coeffs"),
                    "计算近视界径向函数 R(r) 及其导数 dR/dr")
          .def("Evaluate_Coulomb", &TeukolskyRadial::Evaluate_Coulomb,
                         py::arg("r"), py::arg("nu"), py::arg("a_coeffs"),
                         "计算远场径向函数 R_C^nu(r) 及其导数")
                         
          .def("hyp1f1", &TeukolskyRadial::Hyp1F1,
                         py::arg("a"), py::arg("b"), py::arg("z"),py::arg("regularized")=false,
                         "Wrapper for 1F1 confluent hypergeometric function")
          .def("hyp1f1_scaled", &TeukolskyRadial::Hyp1F1_Scaled,
                         py::arg("a"), py::arg("b"), py::arg("z"),py::arg("log_mult"),
                         "Wrapper for 1F1 confluent hypergeometric function")
          .def("hyp1f1_fully_scaled", &TeukolskyRadial::Hyp1F1_FullyScaled,
                         py::arg("a"), py::arg("b"), py::arg("z"),py::arg("factor"),py::arg("log_factor"),
                         "Wrapper for 1F1 confluent hypergeometric function")
          .def("evaluate_ddR", &TeukolskyRadial::evaluate_ddR,
               py::arg("r"), py::arg("R"), py::arg("dR"),
               "利用径向方程精确计算 d2R/dr2")
          .def("Evaluate_R_in", &TeukolskyRadial::Evaluate_R_in,
               py::arg("r"), 
               py::arg("nu"), 
               py::arg("K_nu"), 
               py::arg("K_neg_nu"), 
               py::arg("a_coeffs_pos"), 
               py::arg("a_coeffs_neg"),
               py::arg("r_match") ,
               "计算全域径向函数 R^in(r) 及其导数 (自动拼接)")
          .def("ResetCalibration", &TeukolskyRadial::ResetCalibration,
               "重置连接处校准状态 (在更改物理参数后调用)")
          .def("ComputeAmplitudes", &TeukolskyRadial::ComputeAmplitudes,
               py::arg("nu"), py::arg("a_coeffs"),
               "计算R")
          .def("ComputePhysicalAmplitudes", &TeukolskyRadial::ComputePhysicalAmplitudes,
               py::arg("nu"), py::arg("a_coeffs"),py::arg("amps_nu"));

     py::class_<SWSH>(m, "SWSH")
          .def(py::init<int, int, int, double>(),
                    py::arg("s"), py::arg("l"), py::arg("m"), py::arg("a_omega"))
          .def_property_readonly("m_lambda", &SWSH::get_lambda)
          .def_property_readonly("E", &SWSH::get_E)
          .def_property_readonly("lambda", &SWSH::get_lambda, "物理分离常数 (Separation Constant)")
          .def("evaluate_S", &SWSH::evaluate_S)
          .def("evaluate_L2dag_S", &SWSH::evaluate_L2dag_S)
          .def("evaluate_L1dag_L2dag_S", &SWSH::evaluate_L1dag_L2dag_S)
          .def_static("spin_weighted_Y", &SWSH::spin_weighted_Y,
               py::arg("s"), py::arg("l"), py::arg("m"), py::arg("x"),
               "静态方法: 计算自旋加权球谐函数 Y_s_lm(x)");



          
     // 1. 绑定源项投影系数结构体
     py::class_<SourceProjections>(m, "SourceProjections")
          .def_readwrite("A_nn0", &SourceProjections::A_nn0)
          .def_readwrite("A_mbarn0", &SourceProjections::A_mbarn0)
          .def_readwrite("A_mbarn1", &SourceProjections::A_mbarn1)
          .def_readwrite("A_mbarmbar0", &SourceProjections::A_mbarmbar0)
          .def_readwrite("A_mbarmbar1", &SourceProjections::A_mbarmbar1)
          .def_readwrite("A_mbarmbar2", &SourceProjections::A_mbarmbar2)
          .def("__repr__", [](const SourceProjections &p) {
               return "<SourceProjections A_nn0=...>";
          });

     // 2. 绑定 TeukolskySource 类
     py::class_<TeukolskySource>(m, "TeukolskySource")
          .def(py::init<Real, Real, int, int, int>(),
               py::arg("a"), py::arg("omega"), py::arg("s"), py::arg("l"), py::arg("m"),
               "构造函数: TeukolskySource(a, omega, s,l,m)")
          .def("ComputeProjections", &TeukolskySource::ComputeProjections,
               py::arg("geo_state"), py::arg("geo_obj"), py::arg("swsh"),
               "计算源项投影系数 (输入: KerrGeo.State, KerrGeo, SWSH)");
}
