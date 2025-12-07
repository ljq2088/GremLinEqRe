/**
 * @file Clebsch.h
 * @brief Clebsch-Gordan 系数工具类
 * 参考 GremlinEq/include/SWSH.h (Clebsch class)
 */

 #ifndef CLEBSCH_H
 #define CLEBSCH_H
 
 #include <cmath>
 
 using Real = double;
 
 class Clebsch {
 public:
     // 计算积分 <s, p, m | cos(theta) | s, q, m>
     // 对应 GremlinEq: xbrac
     static Real integral_x(int s, int p, int q, int m);
 
     // 计算积分 <s, p, m | cos^2(theta) | s, q, m>
     // 对应 GremlinEq: xsqrbrac
     static Real integral_x_sqr(int s, int p, int q, int m);
 
 private:
     // 基础 Clebsch-Gordan 系数计算
     // <j1, j2, m1, m2 | J, M>
     static Real cgcof(int j1, int j2, int m1, int m2, int J, int M);
     
     // 阶乘的对数 (辅助函数)
     static double ln_factorial(int n);
 };
 
 #endif // CLEBSCH_H