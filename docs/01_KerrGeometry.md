# Kerr Geometry & Geodesics Implementation

本文档记录了 `KerrGeo` 类中实现的物理公式，主要参考自 Hughes 的 GremlinEq 代码库 (`src/utility/GKG.cc`, `include/Globals.h`)。

## 1. 守恒量与参数

对于克尔黑洞（质量 $M=1$），测试粒子的运动由以下守恒量描述：
* **能量 ($E$)**
* **轴向角动量 ($L_z$)**
* **Carter 常数 ($Q$)**

### 圆形赤道轨道 (Circular Equatorial Orbits)
对于给定半径 $r$ 和自旋 $a$ 的圆形赤道轨道，其参数计算如下（引入辅助变量 $v = 1/\sqrt{r}$）：

**顺行 (Prograde):**
$$\Omega_\phi = \frac{1}{a + r^{3/2}}$$
$$E = \frac{1 - 2v^2 + av^3}{\sqrt{1 - 3v^2 + 2av^3}}$$
$$L_z = r v \frac{1 - 2av^3 + a^2v^4}{\sqrt{1 - 3v^2 + 2av^3}}$$
$$Q = 0$$

**逆行 (Retrograde):**
$$\Omega_\phi = \frac{-1}{r^{3/2} - a}$$
$$E = \frac{1 - 2v^2 - av^3}{\sqrt{1 - 3v^2 - 2av^3}}$$
$$L_z = -r v \frac{1 + 2av^3 + a^2v^4}{\sqrt{1 - 3v^2 - 2av^3}}$$
$$Q = 0$$

---

## 2. 解析势函数 (Analytical Potentials)

这是计算 Teukolsky 方程源项的核心。为了保证精度，我们实现了势函数的**解析导数**，而非使用数值差分。

### 2.1 径向势 $R(r)$

定义：
$$R(r) = [E(r^2+a^2) - aL_z]^2 - \Delta [ K + r^2 ]$$
其中：
* $\Delta = r^2 - 2r + a^2$
* $K = (L_z - aE)^2 + Q$ （分离常数相关项）

**一阶导数 $R'(r)$:**
直接对 $r$ 求导：
$$R'(r) = 2[E(r^2+a^2) - aL_z] \cdot (2Er) - \left[ \Delta'(r)(K+r^2) + \Delta(r)(2r) \right]$$
其中 $\Delta'(r) = 2r - 2$。

**二阶导数 $R''(r)$:**
在 `GremlinEq` 中使用了化简后的多项式形式以提高效率：
$$R''(r) = 2 \left[ a^2(E^2-1) - L_z^2 - Q \right] + r \cdot \left[ 12 + 12(E^2-1)r \right]$$

### 2.2 角向势 $\Theta(z)$

定义变量 $z = \cos\theta$。
$$\Theta(z) = Q - L_z^2 \cot^2\theta - a^2(1-E^2)\cos^2\theta$$
转换为 $z$ 的形式：
$$\Theta(z) = Q - L_z^2 \frac{z^2}{1-z^2} - a^2(1-E^2)z^2$$

> **注意**：GremlinEq 的 `GKG.cc` 实现中输入参数名为 `z` 但逻辑暗示其为 $\cos^2\theta$。为避免歧义，我们的 `KerrGeo::potential_theta(z)` 明确要求输入为 $z=\cos\theta$。