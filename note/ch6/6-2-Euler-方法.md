# 6-2-Euler-方法 (Euler's method)

这是一份数值计算学习笔记，参考了 Tobin A. Driscoll and Richard J. Braun 的教材 [*Fundamentals of Numerical Computation* (2023)](https://tobydriscoll.net/fnc-julia/home.html).

> 这份笔记主要是翻译了原文内容，并删改或重新表述部分内容，希望能进一步减少初学者的学习障碍.

**#1 IVP 的离散与记号**

我们从一阶初值问题出发：

$$
u'(t)=f(t,u(t)),\quad a\le t\le b,\qquad u(a)=u_0.
$$

我们用有限个节点上的数值来表示数值解. 先只考虑等距节点：

$$
t_i=a+ih,\qquad h=\frac{b-a}{n},\qquad i=0,\ldots,n.
$$

其中 $h$ 称为 **step size** (步长).

因为我们在节点处得到的是近似解而不是真解，需要小心区分记号. 从现在开始：

- $\hat{u}(t)$ 表示 IVP 的精确解.
- $u_i\approx \hat{u}(t_i)$ 表示数值方法在节点 $t_i$ 上得到的近似值.

由于初值 $u(a)=u_0$ 是精确给定的，因此不必区分 $u_0$ 是精确解还是数值解.

为了推导第一个方法，我们把节点值 $\{u_i\}$ 用分段线性插值连接起来. 在 $t_i<t<t_{i+1}$ 上，这个插值函数的斜率为

$$
\frac{u_{i+1}-u_i}{t_{i+1}-t_i}
=
\frac{u_{i+1}-u_i}{h}.
$$

接下来，我们把它与微分方程的结构对齐：把左侧看成 $u'(t_i)$ 的一个前向差分近似，并令它等于 $f(t_i,u_i)$.

**#2 Euler 方法**

由

$$
\frac{u_{i+1}-u_i}{h}=f(t_i,u_i)
$$

整理得到 Euler 方法：

> **Algorithm:** Euler's method for an IVP
> Given the IVP $u'=f(t,u)$, $u(a)=u_0$, and the nodes $t_i=a+ih$, iteratively compute
> $$
> u_{i+1}=u_i+h f(t_i,u_i),\qquad i=0,\ldots,n-1.
> $$
> Then $u_i$ is approximately the value of the solution at $t=t_i$.

Euler 方法按时间从左到右推进：用当前的 $u_i$ 显式算出下一步 $u_{i+1}$.

> **Function:** euler
> **Euler's method for an initial-value problem**
> ```Python
> import numpy as np
>
> def euler(f, tspan, u0, n):
>     """
>     Apply Euler's method to solve u' = f(t,u) on tspan=(a,b) with u(a)=u0,
>     using n time steps.
>
>     Returns:
>       t: (n+1,) array of times
>       u: (n+1, ...) array of solution values (scalar or vector)
>     """
>     a, b = float(tspan[0]), float(tspan[1])
>     h = (b - a) / n
>     t = a + h * np.arange(n + 1)
>
>     u0 = np.asarray(u0, dtype=float)
>     u = np.empty((n + 1,) + u0.shape, dtype=float)
>     u[0] = u0
>     for i in range(n):
>         u[i + 1] = u[i] + h * np.asarray(f(t[i], u[i]), dtype=float)
>     return t, u
> ```

> **Note:** 这里把 $u_0$ 统一转成 `numpy` 数组，因此同一份实现同时覆盖标量与向量情形.

**#3 局部截断误差**

现在用 Taylor 展开来评估 Euler 方法的一步误差. 设 $\hat{u}(t)$ 是精确解，并假设我们在 $t=t_i$ 恰好取到了精确值：$u_i=\hat{u}(t_i)$. 那么 Euler 公式给出的下一步是

$$
u_{i+1}=u_i+h f(t_i,u_i)=\hat{u}(t_i)+h f\bigl(t_i,\hat{u}(t_i)\bigr).
$$

对精确解做 Taylor 展开：

$$
\hat{u}(t_{i+1})
=
\hat{u}(t_i)+h\hat{u}'(t_i)+\frac{1}{2}h^2\hat{u}''(t_i)+O(h^3).
$$

由于精确解满足微分方程 $\hat{u}'(t_i)=f\bigl(t_i,\hat{u}(t_i)\bigr)$，因此

$$
\hat{u}(t_{i+1})-\Bigl[\hat{u}(t_i)+h f\bigl(t_i,\hat{u}(t_i)\bigr)\Bigr]
=
\frac{1}{2}h^2\hat{u}''(t_i)+O(h^3).
$$

这说明 Euler 方法的一步误差量级是 $O(h^2)$ (当我们把一步误差写成 "下一步值的误差" 时).

**#4 一步法、截断误差、相容性**

我们先引入更一般的一步法形式.

> **Definition:** One-step IVP method
> A one-step method for the IVP $u'=f(t,u)$ is a formula of the form
> $$
> u_{i+1}=u_i+h\,\phi(t_i,u_i,h),\qquad i=0,\ldots,n-1.
> $$

Euler 方法就是其中的特例：$\phi(t,u,h)=f(t,u)$.

与 **5-5-有限差分的收敛性** 类似，我们把截断误差定义为：把精确解代回一步公式后得到的残差 (并按步长做归一化).

> **Definition:** Truncation error of a one-step IVP method
> The local truncation error (LTE) of the one-step method is
> $$
> \tau_{i+1}(h):=\frac{\hat{u}(t_{i+1})-\hat{u}(t_i)}{h}-\phi\bigl(t_i,\hat{u}(t_i),h\bigr).
> $$
> The method is called consistent if $\tau_{i+1}(h)\to 0$ as $h\to 0$.

> **Lemma:** Consistency condition
> If $\phi(t,u,0)=f(t,u)$ for all $t$ and $u$, then the one-step method is consistent.

**#5 收敛性与全局误差**

给定精确解 $\hat{u}(t)$，数值解在节点处的全局误差可以视为向量

$$
\bigl[\hat{u}(t_i)-u_i\bigr]_{i=0,\ldots,n}.
$$

有时 "全局误差" 也指这向量的最大范数，或指它在终点时刻的分量.

从定义看，从 $t_i$ 走到 $t_{i+1}$ 的局部误差是 $h\tau_{i+1}(h)$. 若要用步长 $h$ 从 $a$ 走到 $b$，需要 $n=(b-a)/h=O(h^{-1})$ 步. 因此把 $\tau_{i+1}(h)$ 定义成 "每单位步长的局部误差" 后，它已经把 "误差要积累 $O(n)$ 次" 这件事先扣掉了一部分.

但全局误差并不是简单地把局部误差相加. 正如 **6-1-IVP-基础** 里初值扰动的讨论所示，每一步产生的误差会像扰动一样随时间演化，可能被放大或衰减. 因此，分析全局误差需要同时考虑：

- 局部误差的累加.
- 局部误差在动力系统流 (flow) 下的放大效应.

下面的定理给出了一个典型结果.

> **Theorem:** Global error bound for a one-step method
> Suppose that the unit local truncation error satisfies
> $$
> |\tau_{i+1}(h)|\le Ch^p,
> $$
> and that
> $$
> \left|\frac{\partial \phi}{\partial u}\right|\le L
> $$
> for all $t\in[a,b]$, all $u$, and all $h>0$. Then the global error satisfies
> $$
> |\hat{u}(t_i)-u_i|
> \le
> \frac{Ch^p}{L}\bigl[e^{L(t_i-a)}-1\bigr]
> = O(h^p),
> $$
> as $h\to 0$.

这个定理也支撑了下面的精度阶定义.

> **Definition:** Order of accuracy of a one-step IVP method
> If the local truncation error satisfies $\tau_{i+1}(h)=O(h^p)$ for a positive integer $p$, then $p$ is the order of accuracy of the formula.

我们可以把上面的定理理解为：在常见假设下，全局误差与 LTE 具有相同的精度阶. 但要注意：$O(h^p)$ 里隐藏的常数会随时间指数增长. 当时间区间固定 (例如 $b-a$ 有界) 时，这不会破坏结论；但如果我们把时间推进到非常长的区间，则没有同样的保证.

**#6 一个收敛性实验**

> **Demo:** Convergence of Euler's method.
> We consider the IVP
> $$
> u'=\sin\bigl((u+t)^2\bigr),\quad 0\le t\le 4,\qquad u(0)=-1.
> $$
> ```Python
> import numpy as np
> import matplotlib.pyplot as plt
> from scipy.integrate import solve_ivp
>
> def f(t, u):
>     return np.sin((t + u)**2)
>
> def rhs(t, y):
>     return [f(t, y[0])]
>
> a, b = 0.0, 4.0
> u0 = -1.0
>
> # Euler solutions with different step counts.
> for n in [20, 50]:
>     t, u = euler(f, (a, b), u0, n)
>     plt.plot(t, u, marker="o", ms=3, label=f"n={n}")
>
> # A high-accuracy reference solution for comparison.
> ref = solve_ivp(rhs, (a, b), [u0], method="DOP853", rtol=1e-13, atol=1e-15, dense_output=True)
> tt = np.linspace(a, b, 800)
> plt.plot(tt, ref.sol(tt)[0], color="k", lw=2, label="reference")
>
> plt.xlabel("t")
> plt.ylabel("u(t)")
> plt.title("Solution by Euler's method")
> plt.grid(True)
> plt.legend()
> plt.show()
> ```
>
> To study convergence, we compute an infinity-norm error against the reference solution on the Euler time nodes:
>
> ```Python
> import numpy as np
> import matplotlib.pyplot as plt
> from scipy.integrate import solve_ivp
>
> def f(t, u):
>     return np.sin((t + u)**2)
>
> def rhs(t, y):
>     return [f(t, y[0])]
>
> a, b = 0.0, 4.0
> u0 = -1.0
>
> ref = solve_ivp(rhs, (a, b), [u0], method="DOP853", rtol=1e-13, atol=1e-15, dense_output=True)
>
> ns = np.round(5 * 10 ** np.arange(0, 3.1, 0.5)).astype(int)  # 5,16,50,...,5000
> err = []
> for n in ns:
>     t, u = euler(f, (a, b), u0, int(n))
>     u_ref = ref.sol(t)[0]
>     err.append(np.max(np.abs(u_ref - u)))
>
> err = np.array(err)
> for n, e in zip(ns, err):
>     print(n, e)
>
> plt.loglog(ns, err, "o-", label="results")
> plt.loglog(ns, 0.05 * (ns / ns[0]) ** (-1), "--", label=r"$O(n^{-1})$")
> plt.xlabel("n")
> plt.ylabel("Inf-norm global error")
> plt.title("Convergence of Euler's method")
> plt.grid(True, which="both")
> plt.legend()
> plt.show()
> ```
> The error decreases by about a factor of 10 when n increases by a factor of 10, indicating first-order convergence.

> **Note:** 因为 $h=(b-a)/n$，所以 $O(h)=O(n^{-1})$. 因此在对数坐标下，误差曲线的斜率为 $-1$ 与 "一阶收敛" 是同一件事.
