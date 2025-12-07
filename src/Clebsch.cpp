/**
 * @file Clebsch.cpp
 * @brief Clebsch-Gordan 系数实现
 * 参考 GremlinEq/src/swsh/SWSHCGUtil.cc
 */

 #include "Clebsch.h"
 #include <vector>
 #include <iostream>
 
 // 辅助：计算 log(n!)
 double Clebsch::ln_factorial(int n) {
     if (n < 0) return 0.0; // 简单的错误处理
     static std::vector<double> cache = {0.0}; // log(0!) = 0
     if (n < (int)cache.size()) return cache[n];
     
     double val = cache.back();
     for (int i = cache.size(); i <= n; ++i) {
         val += std::log((double)i);
         cache.push_back(val);
     }
     return cache[n];
 }
 
 // 核心：计算 CG 系数 <j1, j2, m1, m2 | J, M>
 // 公式参考 GremlinEq，基于 Hamermesh 教材
 Real Clebsch::cgcof(int j1, int j2, int m1, int m2, int J, int M) {
     // 选择规则检查
     if (m1 + m2 != M) return 0.0;
     if (std::abs(j1 - j2) > J || (j1 + j2) < J) return 0.0;
     if (std::abs(m1) > j1 || std::abs(m2) > j2 || std::abs(M) > J) return 0.0;
 
     // 求和变量 lambda (不要与物理特征值混淆)
     // 范围: max(0, -(J-j2+m1), -(J-j1-m2)) <= lambda <= min(j1+j2-J, j1-m1, j2+m2)
     // GremlinEq 实现比较直接，遍历所有可能的 lambda，利用阶乘对负数无效来自动截断
     
     // 为了稳健，我们先计算预因子 (prefactor)
     double ln_pre = 0.5 * (ln_factorial(J + M) + ln_factorial(J - M) +
                            ln_factorial(J - j2 + j1) + ln_factorial(J - j1 + j2) +
                            ln_factorial(j1 + j2 - J) - ln_factorial(j1 + j2 + J + 1));
     
     // 加上另外一部分预因子
     ln_pre += 0.5 * (ln_factorial(j1 + m1) + ln_factorial(j1 - m1) +
                      ln_factorial(j2 + m2) + ln_factorial(j2 - m2));
 
     double sum = 0.0;
     
     // 确定 lambda 遍历范围
     int min_lam = 0;
     int max_lam = j1 + j2 - J; // GremlinEq 循环上限
     
     for (int lam = min_lam; lam <= max_lam; ++lam) {
         // 分母中的项，必须非负
         int term1 = j1 + j2 - J - lam;
         int term2 = j1 - m1 - lam;
         int term3 = j2 + m2 - lam;
         int term4 = J - j2 + m1 + lam;
         int term5 = J - j1 - m2 + lam;
         
         if (term1 < 0 || term2 < 0 || term3 < 0 || term4 < 0 || term5 < 0) continue;
         
         double ln_term = - (ln_factorial(lam) + ln_factorial(term1) + ln_factorial(term2) + 
                             ln_factorial(term3) + ln_factorial(term4) + ln_factorial(term5));
         
         double term = std::exp(ln_term);
         if (lam % 2 != 0) term = -term; // (-1)^lambda
         
         sum += term;
     }
     
     // 最终还要乘以前面的 sqrt(2J+1) 和预因子
     return std::sqrt(2.0 * J + 1.0) * std::exp(ln_pre) * sum;
 }
 
 // 计算 <s, p, m | cos(theta) | s, q, m>
 // GremlinEq: xbrac
 Real Clebsch::integral_x(int s, int p, int q, int m) {
     // 公式: C(q, 1, m, 0, p, m) * C(q, 1, -s, 0, p, -s) * sqrt((2q+1)/(2p+1))
     // 注意：GremlinEq 代码里写的是 cgcof(q, 1, m, 0, p, m)，意味着它是从 q 耦合到 p
     // <p | x | q>
     
     Real term1 = cgcof(q, 1, m, 0, p, m);
     Real term2 = cgcof(q, 1, -s, 0, p, -s);
     Real factor = std::sqrt((2.0 * q + 1.0) / (2.0 * p + 1.0));
     
     return term1 * term2 * factor;
 }
 
 // 计算 <s, p, m | cos^2(theta) | s, q, m>
 // GremlinEq: xsqrbrac
 Real Clebsch::integral_x_sqr(int s, int p, int q, int m) {
     // 公式: C(q, 2, m, 0, p, m) * C(q, 2, -s, 0, p, -s) * (2/3) * sqrt((2q+1)/(2p+1))
     // 加上对角项修正: if (q == p) ans += 1.0/3.0
     
     Real term1 = cgcof(q, 2, m, 0, p, m);
     Real term2 = cgcof(q, 2, -s, 0, p, -s);
     Real factor = (2.0 / 3.0) * std::sqrt((2.0 * q + 1.0) / (2.0 * p + 1.0));
     
     Real ans = term1 * term2 * factor;
     if (q == p) ans += 1.0 / 3.0;
     
     return ans;
 }