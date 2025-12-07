# MST Method for Radial Teukolsky Equation

本文档基于 Fujita & Tagoshi (2004) [arXiv:gr-qc/0410018] 以及 GremlinEq 的实现。

## 1. 核心思想

MST (Mano-Suzuki-Takasugi) 方法通过将径向 Teukolsky 方程的齐次解展开为超几何函数级数（在视界附近收敛）或库伦波函数级数（在无穷远收敛），将微分方程问题转化为求解级数系数 $a_n$ 的代数问题。

关键在于引入一个**重整化角动量 (Renormalized Angular Momentum)** $\nu$，它是使得级数在两端同时收敛的特征值。

## 2. 递推系数 (MST Coefficients)

展开系数 $a_n$ 满足三项递推关系：
$$\alpha_n^\nu a_{n+1} + \beta_n^\nu a_n + \gamma_n^\nu a_{n-1} = 0$$

GremlinEq 采用去分母后的多项式形式以提高数值精度（与 FT04 Eq. 2-10 相比差一个公因子，不影响方程的根）。我们在 `TeukolskyRadial.cpp` 中实现的公式如下：

令 $N = n + \nu$。

**Alpha ($\alpha_n$):**
$$\alpha_n \propto i \epsilon \kappa (N) (2N-1) [(N-1)^2 + \epsilon^2] (N+1+i\tau)$$

**Beta ($\beta_n$):**
$$\beta_n \propto 4 \cdot [ M_1 (M_1 + B_1) + B_2 ] \cdot (N+1.5) (N-0.5)$$
其中：
* $M_1 = N(N+1)$
* $B_1 = 2\epsilon^2 - \epsilon m q - \lambda - 2$
* $B_2 = \epsilon(\epsilon - mq)(4 + \epsilon^2)$

**Gamma ($\gamma_n$):**
$$\gamma_n \propto -i \epsilon \kappa (N+1) (2N+3) [(N+2)^2 + \epsilon^2] (N-i\tau)$$

*(注：具体系数需对照代码 `coeff_alpha` 等实现，确保与 GremlinEq `fraction_macros.h` 一致)*

## 3. 连分式与超越方程

为了确定 $\nu$，我们需要求解超越方程 $g(\nu) = 0$：
$$\beta_0^\nu + \alpha_0^\nu R_1(\nu) + \gamma_0^\nu L_{-1}(\nu) = 0$$

其中 $R_n$ 和 $L_n$ 是连分式 (FT04 Eq. 2-14, 2-15)：
$$R_n = \frac{a_n}{a_{n-1}} = -\frac{\gamma_n}{\beta_n + \alpha_n R_{n+1}} = -\frac{\gamma_n}{\beta_n - \frac{\alpha_n \gamma_{n+1}}{\beta_{n+1} - \dots}}$$
$$L_n = \frac{a_n}{a_{n+1}} = -\frac{\alpha_n}{\beta_n + \gamma_n L_{n-1}} = -\frac{\alpha_n}{\beta_n - \frac{\alpha_{n-1} \gamma_n}{\beta_{n-1} - \dots}}$$

在代码中，`continued_fraction(nu, direction)` 函数负责计算这些连分式的值。

## 4. 物理参数定义

* $\epsilon = 2 M \omega$
* $q = a/M$
* $\kappa = \sqrt{1-q^2}$
* $\tau = (\epsilon - mq)/\kappa$