# 6-1-IVP-基础 (Basics of IVPs)

这是一份数值计算学习笔记，参考了 Tobin A. Driscoll and Richard J. Braun 的教材 [*Fundamentals of Numerical Computation* (2023)](https://tobydriscoll.net/fnc-julia/home.html).

> 这份笔记主要是翻译了原文内容，并删改或重新表述部分内容，希望能进一步减少初学者的学习障碍.

**#1 一阶标量初值问题**

> **Definition:** Initial-value problem (scalar)
> $$
> u'(t)=f\bigl(t,u(t)\bigr),\quad a\le t\le b,\qquad u(a)=u_0.
> $$

我们把 $t$ 称为自变量 (independent variable)，把 $u$ 称为因变量 (dependent variable). 若

$$
u'=f(t,u)=g(t)+u h(t),
$$

则微分方程是线性的；否则是非线性的.

一个 IVP 的解是一个函数 $u(t)$，它同时满足微分方程 $u'(t)=f(t,u(t))$ 与初值条件 $u(a)=u_0$.

当 $t$ 表示时间时，有时会用 $\dot{u}$ (读作 "u-dot") 代替 $u'$.

**#2 一个种群模型例子**

> **Example:** Population growth and the logistic equation.
> Suppose $u(t)$ is the size of a population at time $t$. We idealize by allowing $u$ to take any real value. If we assume a constant per capita birth rate, then
> $$
> \frac{du}{dt}=ku,\qquad u(0)=u_0,
> $$
> for some $k>0$. The solution is $u(t)=e^{kt}u_0$, which is exponential growth.
>
> A more realistic model caps the growth due to finite resources. Suppose the death rate is proportional to the size of the population, indicating competition. Then
> $$
> \frac{du}{dt}=ku-ru^2,\qquad u(0)=u_0.
> $$
> This is the logistic equation. The solution relevant for population models has the form
> $$
> u(t)=\frac{k/r}{1+\left(\frac{k}{ru_0}-1\right)e^{-kt}}.
> $$
> For $k,r,u_0>0$, the solution smoothly varies from the initial population $u_0$ to a finite population $k/r$ that has been limited by competition.

线性问题可以用积分写出解析解. 例如对 $u'=g(t)+u h(t)$，定义积分因子

$$
\rho(t)=\exp\left[\int -h(t)\,dt\right],
$$

则解可以由

$$
\rho(t)u(t)=u_0+\int_a^t \rho(s)g(s)\,ds
$$

推导出来.

但在很多情况下，上面的积分无法用闭式计算. 一些非线性 ODE (例如可分离变量方程) 也可能有短公式，但通常会伴随难以处理的积分. 更常见的情况是：我们没有可用的解析表达式，因此必须求数值解.

ODE 也可能包含未知函数的高阶导数. 例如二阶常微分方程常写成

$$
u''(t)=f(t,u,u').
$$

一个二阶 IVP 在初始时刻需要两个条件才能完全确定解. 在后面的 **6-3-IVP-系统** 中，我们会把高阶 IVP 改写为一阶形式，因此后续只讨论一阶问题.

**#3 数值解的基本形态**

> **Demo:** Solving an IVP numerically.
> We solve the IVP
> $$
> u'=\sin\bigl((u+t)^2\bigr),\quad t\in[0,4],\qquad u(0)=-1.
> $$
> ```Python
> import numpy as np
> import matplotlib.pyplot as plt
> from scipy.integrate import solve_ivp
>
> def rhs(t, y):
>     u = y[0]
>     return [np.sin((t + u)**2)]
>
> sol = solve_ivp(
>     rhs,
>     t_span=(0.0, 4.0),
>     y0=[-1.0],
>     method="RK45",
>     rtol=1e-10,
>     atol=1e-12,
>     dense_output=True,
> )
>
> # The solution can be evaluated at other times via interpolation.
> print("u(1.0) =", float(sol.sol(1.0)[0]))
>
> # Discrete values chosen by the solver.
> print("t nodes:", sol.t)
> print("u values:", sol.y[0])
>
> plt.plot(sol.t, sol.y[0], label="discrete values")
> tt = np.linspace(0.0, 4.0, 400)
> plt.plot(tt, sol.sol(tt)[0], label="interpolated solution")
> plt.xlabel("t")
> plt.ylabel("u(t)")
> plt.title(r"$u'=\sin((t+u)^2)$")
> plt.grid(True)
> plt.legend()
> plt.show()
> ```
> The solver first computes approximate values at automatically chosen times, and then uses interpolation to evaluate the solution at other times.

**#4 存在性与唯一性**

有些看起来简单的 IVP 也可能并不存在 "对所有时间都成立" 的解. 例如对

$$
u'=(u+t)^2
$$

这类非线性方程，解可能在有限时间内 blow up (发散到无穷大).

> **Demo:** Finite-time blowup.
> ```Python
> import numpy as np
> import matplotlib.pyplot as plt
> from scipy.integrate import solve_ivp
>
> def rhs(t, y):
>     u = y[0]
>     return [(t + u)**2]
>
> sol = solve_ivp(
>     rhs,
>     t_span=(0.0, 1.0),
>     y0=[1.0],
>     method="RK45",
>     rtol=1e-10,
>     atol=1e-12,
> )
>
> print("status =", sol.status)
> print("message =", sol.message)
>
> plt.semilogy(sol.t, sol.y[0])
> plt.xlabel("t")
> plt.ylabel("u(t)")
> plt.title("Finite-time blowup (numerical)")
> plt.grid(True)
> plt.show()
> ```
> A failure message (e.g., step size forced too small) can indicate that the solution does not exist beyond some time.

我们也可以构造一个 IVP，使它有不止一个解.

> **Example:** Multiple solutions.
> The functions $u(t)=t^2$ and $u(t)\equiv 0$ both satisfy the differential equation $u'=2\sqrt{u}$ and the initial condition $u(0)=0$. Thus the corresponding IVP has more than one solution.

下面的标准定理给出一个容易检查、并能保证唯一解存在的条件 (但它不是最一般的条件，因此也可能漏掉一些确实有唯一解的情况).

> **Theorem:** Existence and uniqueness
> If the derivative $\partial f/\partial u$ exists and $\left|\partial f/\partial u\right|$ is bounded by a constant $L$ for all $t$ with $a\le t\le b$ and all $u$, then the initial-value problem has a unique solution for $t\in[a,b]$.

**#5 初值扰动与条件性**

在数值计算中，我们还需要关心 IVP 的条件性. 对一阶 IVP 来说，(数据) 至少包含两部分：函数 $f(t,u)$ 与初值 $u_0$. 讨论 "扰动一个数" 比 "扰动一个函数" 更直接，因此这里聚焦初值扰动 $u_0\mapsto u_0+\delta$ 对解的影响.

> **Theorem:** Dependence on initial value
> If the derivative $\partial f/\partial u$ exists and $\left|\partial f/\partial u\right|$ is bounded by a constant $L$ for all $t$ with $a\le t\le b$ and all $u$, then the solution $u(t;u_0+\delta)$ of $u'=f(t,u)$ with initial condition $u(a)=u_0+\delta$ satisfies
> $$
> \|u(t;u_0+\delta)-u(t;u_0)\|_\infty \le |\delta| e^{L(b-a)}
> $$
> for all sufficiently small $|\delta|$.

数值解必然会有误差，而这些误差可以视为对解的扰动. 上面的定理给出了一个上界：$e^{L(b-a)}$ 可以作为 "解对初值扰动" 的 (逐点) 绝对条件数上界. 但这个上界可能会严重高估某个具体问题的实际敏感性.

> **Demo:** A bound can be very pessimistic.
> Consider $u'=u$ and $u'=-u$. In both cases, $\partial f/\partial u=\pm 1$, so the bound from the theorem is $e^{b-a}$. But the behaviors differ.
> ```Python
> import numpy as np
> import matplotlib.pyplot as plt
>
> a, b = 0.0, 1.0
> t = np.linspace(a, b, 200)
>
> u0 = 1.0
> delta = 1e-3
>
> u_plus = (u0 + delta) * np.exp(t)
> u_base = u0 * np.exp(t)
> diff_grow = np.abs(u_plus - u_base)
>
> u_plus2 = (u0 + delta) * np.exp(-t)
> u_base2 = u0 * np.exp(-t)
> diff_decay = np.abs(u_plus2 - u_base2)
>
> print("max diff (u'=u)   =", float(diff_grow.max()))
> print("max diff (u'=-u)  =", float(diff_decay.max()))
> print("bound             =", abs(delta) * np.exp(b - a))
>
> plt.plot(t, diff_grow, label=r"$u'=u$")
> plt.plot(t, diff_decay, label=r"$u'=-u$")
> plt.xlabel("t")
> plt.ylabel(r"$|u(t;u_0+\\delta)-u(t;u_0)|$")
> plt.title("Sensitivity to initial value")
> plt.grid(True)
> plt.legend()
> plt.show()
> ```
> In the growing case the bound is sharp, while in the decaying case the true condition number is 1 (the maximum difference occurs at the initial time).

一般来说，初值扰动可能导致解轨线相互远离、相互靠近、或围绕原轨线振荡. 我们在后面的章节才会系统讨论这些行为及其对数值方法的影响.
