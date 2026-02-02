# 4-3-Newton-方法 (Newton's method)

这是一份数值计算学习笔记，参考了 Tobin A. Driscoll and Richard J. Braun 的教材 [*Fundamentals of Numerical Computation* (2023)](https://tobydriscoll.net/fnc-julia/home.html).

> 这份笔记主要是翻译了原文内容，并删改或重新表述部分内容，希望能进一步减少初学者的学习障碍.

**#1 Newton 方法的切线思想**

在求根问题里，**Newton's method** 是最核心的方法之一. 它的关键想法是：在当前近似点 $x_k$ 处用切线去近似 $f$，然后用 "切线的根" 作为新的近似 $x_{k+1}$.

> **Demo:** Tangent line approximation
> Suppose we want to find a root of $f(x)=x e^x-2$. From the graph, there is a root near $x=1$, so we take $x_1=1$ as an initial guess.
>
> ```Python
> import numpy as np
> import matplotlib.pyplot as plt
>
> def f(x):
>     return x * np.exp(x) - 2.0
>
> def dfdx(x):
>     return np.exp(x) * (x + 1.0)
>
> xs = np.linspace(0.0, 1.5, 600)
> plt.plot(xs, f(xs), label="function")
> plt.axhline(0.0, color="k", lw=1)
> plt.grid(True, axis="y", alpha=0.3)
> plt.xlabel("x")
> plt.ylabel("y")
> plt.ylim(-2, 4)
> plt.legend(loc="upper left")
> plt.title("Tangent line approximation")
>
> x1 = 1.0
> y1 = f(x1)
> m1 = dfdx(x1)
> tangent1 = lambda x: y1 + m1 * (x - x1)
>
> plt.scatter([x1], [y1], label="initial point")
> plt.plot(xs, tangent1(xs), ls="--", label="tangent line")
>
> x2 = x1 - y1 / m1
> print("x2 =", x2)
> plt.scatter([x2], [0.0], label="tangent root")
> plt.legend()
> plt.show()
> ```
> The tangent line root $x_2$ is typically closer to the true root than $x_1$, so we repeat the same construction at $(x_2,f(x_2))$.

**#2 Newton 迭代公式**

用一般记号表达：给定当前近似 $x_k$，我们用切线构造线性模型

$$
q(x)=f(x_k)+f'(x_k)(x-x_k).
$$

求 $q(x)=0$ 是容易的，并由 $q(x_{k+1})=0$ 得到 Newton 迭代：

> **Algorithm:** Newton's method
> Given a function $f$, its derivative $f'$, and an initial value $x_1$, iteratively define
> $$
> x_{k+1}=x_k-\frac{f(x_k)}{f'(x_k)},\qquad k=1,2,\dots.
> $$

**#3 收敛性：误差的平方递推**

从 **4-2-不动点迭代** 的经验可以看出：即使迭代形式非常简洁，"是否收敛" 与 "收敛得多快" 都需要单独分析. 对 Newton 方法来说，收敛阶的推导相对直接.

设迭代 $x_{k+1}=x_k-\frac{f(x_k)}{f'(x_k)}$ 的极限为 $r$，并且 $f(r)=0$. 令误差为

$$
\epsilon_k=x_k-r.
$$

把 $x_k=r+\epsilon_k$ 代入迭代式：

$$
\epsilon_{k+1}+r
=
\epsilon_k+r-\frac{f(r+\epsilon_k)}{f'(r+\epsilon_k)}.
$$

假设 $|\epsilon_k|\to 0$，并且 $f$ 在 $r$ 附近足够光滑 (至少二阶连续可导). 对 $f(r+\epsilon_k)$ 与 $f'(r+\epsilon_k)$ 在 $r$ 处做 Taylor 展开：

$$
f(r+\epsilon_k)
=
f(r)+\epsilon_k f'(r)+\frac{1}{2}\epsilon_k^2 f''(r)+O(\epsilon_k^3),
$$

$$
f'(r+\epsilon_k)
=
f'(r)+\epsilon_k f''(r)+O(\epsilon_k^2).
$$

再利用 $f(r)=0$，并额外假设 $r$ 是 **simple root** (即 $f'(r)\ne 0$)，可以得到

$$
\epsilon_{k+1}
=
\frac{1}{2}\frac{f''(r)}{f'(r)}\epsilon_k^2+O(\epsilon_k^3).
$$

因此在渐近区间内，Newton 方法的误差会被大致平方.

> **Observation:** Error squaring in Newton's method
> Asymptotically, each iteration of Newton's method roughly squares the error.

**#4 二次收敛与数值判别**

为了形式化刻画 "误差平方" 这件事，我们引入二次收敛.

> **Definition:** Quadratic convergence
> Suppose a sequence $x_k$ approaches limit $x^*$. If the error sequence $\epsilon_k=x_k-x^*$ satisfies
> $$
> \lim_{k\to\infty}\frac{|\epsilon_{k+1}|}{|\epsilon_k|^2}=L
> $$
> for a positive constant $L$, then the sequence has quadratic convergence to the limit.

线性收敛常见的经验判据是：在 log-linear 图中误差趋向一条直线. 二次收敛下，误差下降会越来越陡，因此不会出现稳定直线段. 作为一种数值检验，注意 $|\epsilon_{k+1}|\approx K|\epsilon_k|^2$ 会推出

$$
\frac{\log|\epsilon_{k+1}|}{\log|\epsilon_k|}\to 2,\qquad (k\to\infty).
$$

> **Demo:** Quadratic convergence in practice
> We revisit $f(x)=x e^x-2$ and compute Newton iterates using extended precision so that we can observe several squaring steps beyond machine precision.
>
> ```Python
> import mpmath as mp
>
> mp.mp.dps = 80  # extended precision
>
> def f(x):
>     return x * mp.e**x - 2
>
> def dfdx(x):
>     return mp.e**x * (x + 1)
>
> x = mp.mpf("1.0")
> xs = [x]
> for _ in range(7):
>     xs.append(xs[-1] - f(xs[-1]) / dfdx(xs[-1]))
>
> r = xs[-1]  # take the last iterate as a highly accurate root estimate
>
> err = [abs(xk - r) for xk in xs[:-1]]
> logerr = [mp.log(e) for e in err]
> ratios = [logerr[i+1] / logerr[i] for i in range(len(logerr) - 1)]
>
> print("errors:")
> for e in err:
>     print(e)
>
> print("log(err[k+1])/log(err[k]) ratios:")
> for q in ratios:
>     print(q)
> ```
> The ratios of $\log|\epsilon_{k+1}|$ to $\log|\epsilon_k|$ should trend toward 2, which is strong numerical evidence of quadratic convergence.

> **Note:** 在普通双精度浮点数里，Newton 方法往往在很少几步后就把误差压到机器精度附近，此时继续迭代会很快 "停滞" 在舍入误差水平上. 因此上面的 Demo 用扩展精度来观察更多轮的渐近行为.

推导二次收敛以及上面的数值现象时，我们隐含使用了几个关键假设：

1. $f$ 在根附近足够光滑，使得 Taylor 展开成立 (通常说 $f$ 具有足够的 smoothness).
2. $r$ 必须是 simple root，即 $f'(r)\ne 0$；在多重根处，二次收敛会失效.
3. 我们默认迭代序列确实会收敛. 但在具体问题里，找到一个能让 Newton 迭代收敛的初值往往是最困难的部分；后续会在 **4-6-拟 Newton 方法** 里继续处理这个问题.

**#5 一个可复用的实现：把问题与算法分离**

在实现 Newton 方法时，我们通常把 "问题定义 (f 与 f')" 与 "算法 (Newton 迭代)" 分开：同一段算法代码可以复用在不同的求根问题上；并且我们也更容易对同一问题尝试多种求根算法.

另外还有一个实际问题：如何决定停止迭代. 真实误差 $|x_k-r|$ 不可直接计算，因此常用两个替代指标：

- 相邻两次近似之差 $|x_k-x_{k-1}|$ (把它当作误差的 proxy).
- 残差 $|f(x_k)|$ (在 **4-1-求根问题** 的语境下，它更接近一种可控的反向误差指标).

再加上最大迭代次数作为安全阀，就构成一个可用的停止准则.

> **Function:** newton
> **Newton's method for a scalar rootfinding problem**
> ```Python
> import numpy as np
>
> def newton(f, dfdx, x1, *, maxiter=40, ftol=None, xtol=None):
>     """
>     newton(f, dfdx, x1, *, maxiter=40, ftol=..., xtol=...)
>
>     Use Newton's method to find a root of f starting from x1.
>     Returns an array of all iterates.
>     """
>     eps = np.finfo(float).eps
>     if ftol is None:
>         ftol = 100 * eps
>     if xtol is None:
>         xtol = 100 * eps
>
>     x = float(x1)
>     xs = [x]
>     y = float(f(x))
>     dx = np.inf  # for the first pass
>
>     k = 0
>     while (abs(dx) > xtol) and (abs(y) > ftol):
>         k += 1
>         if k > maxiter:
>             print("Warning: Maximum number of iterations reached.")
>             break
>
>         dydx = float(dfdx(x))
>         if dydx == 0.0:
>             print("Warning: Derivative is zero; Newton step undefined.")
>             break
>         dx = -y / dydx
>         x = x + dx
>         xs.append(x)
>         y = float(f(x))
>
>     return np.array(xs)
> ```

> **Note:** 当 $f'(x_k)=0$ 或非常接近 0 时，Newton 步 $\Delta x=-f(x_k)/f'(x_k)$ 会变得不可用或极不稳定. 这种情形下通常需要换初值，或改用带 bracket 的方法 (见 **4-4-基于插值的方法**)，或用阻尼 / quasi-Newton (见 **4-6-拟 Newton 方法**).

**#6 用 Newton 方法数值计算反函数**

当我们想数值求一个函数的反函数值时，本质上是在解一个方程. 例如考虑

$$
g(x)=e^x-x.
$$

给定 $y$ 时，要计算 $g^{-1}(y)$，等价于求根

$$
f_y(x)=g(x)-y=0.
$$

> **Demo:** Function and its inverse
> ```Python
> import numpy as np
> import matplotlib.pyplot as plt
>
> def g(x):
>     return np.exp(x) - x
>
> def dgdx(x):
>     return np.exp(x) - 1.0
>
> ys = np.linspace(g(0.0), g(2.0), 200)
> xs = np.zeros_like(ys)
>
> for i, y in enumerate(ys):
>     fy = lambda x, y=y: g(x) - y
>     it = newton(fy, dgdx, x1=y)
>     xs[i] = it[-1]
>
> grid = np.linspace(0.0, 2.0, 400)
> plt.figure()
> plt.plot(grid, g(grid), label="g(x)", lw=2)
> plt.plot(ys, xs, label=r"$g^{-1}(y)$", lw=2)
> plt.plot(grid, grid, ls="--", color="k", lw=1, label="y=x")
> plt.gca().set_aspect("equal", adjustable="box")
> plt.title("Function and its inverse")
> plt.xlabel("x")
> plt.ylabel("y")
> plt.grid(True, alpha=0.3)
> plt.legend()
> plt.show()
> ```
> The inverse curve is constructed by solving $g(x)=y$ for many values of $y$.
