#include <pybind11/complex.h>
#include <pybind11/stl.h>
#include "KerrGeo.h"
#include "TeukolskyRadial.h"
#include "SWSH.h"
namespace py = pybind11;

PYBIND11_MODULE(_core, m) {
     m.doc() = "GremLinEqRe C++ Core Module (General Orbits)";

     py::class_<KerrGeo>(m, "KerrGeo")
          // 构造函数 1: 通用 (E, Lz, Q)
          .def(py::init<double, double, double, double>(),
               py::arg("a"), py::arg("E"), py::arg("Lz"), py::arg("Q"),
               "通用构造函数: 直接指定守恒量")
          
          // 构造函数 2: 圆形赤道 (工厂模式)
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
          .def("Evaluate_Hypergeometric", &TeukolskyRadial::Evaluate_Hypergeometric,
                    py::arg("r"), py::arg("nu"), py::arg("a_coeffs"),
                    "计算近视界径向函数 R(r) 及其导数 dR/dr")
          .def("Evaluate_Coulomb", &TeukolskyRadial::Evaluate_Coulomb,
                         py::arg("r"), py::arg("nu"), py::arg("a_coeffs"),
                         "计算远场径向函数 R_C^nu(r) 及其导数")
                         
          .def("hyp1f1", &TeukolskyRadial::Hyp1F1,
                         py::arg("a"), py::arg("b"), py::arg("z"),py::arg("regularized")=false,
                         "Wrapper for 1F1 confluent hypergeometric function")
          .def("Evaluate_R_in", &TeukolskyRadial::Evaluate_R_in,
               py::arg("r"), 
               py::arg("nu"), 
               py::arg("K_nu"), 
               py::arg("K_neg_nu"), 
               py::arg("a_coeffs_pos"), 
               py::arg("a_coeffs_neg"),
               py::arg("r_match") = 5.0,
               "计算全域径向函数 R^in(r) 及其导数 (自动拼接)");

     py::class_<SWSH>(m, "SWSH")
          .def(py::init<int, int, int, double>(),
                    py::arg("s"), py::arg("l"), py::arg("m"), py::arg("a_omega"))
          .def_property_readonly("lambda", &SWSH::get_lambda)
          .def_property_readonly("E", &SWSH::get_E)
          .def("evaluate_S", &SWSH::evaluate_S)
          .def("evaluate_L2dag_S", &SWSH::evaluate_L2dag_S)
          .def("evaluate_L1dag_L2dag_S", &SWSH::evaluate_L1dag_L2dag_S);
}