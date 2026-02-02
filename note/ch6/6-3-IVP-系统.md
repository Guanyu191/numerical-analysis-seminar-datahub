# 6-3-IVP-系统 (IVP systems)

这是一份数值计算学习笔记，参考了 Tobin A. Driscoll and Richard J. Braun 的教材 [*Fundamentals of Numerical Computation* (2023)](https://tobydriscoll.net/fnc-julia/home.html).

> 这份笔记主要是翻译了原文内容，并删改或重新表述部分内容，希望能进一步减少初学者的学习障碍.

**#1 为什么需要 IVP systems**

很少有应用只涉及一个因变量的初值问题. 更常见的是多个未知量共同演化，并由一个方程组来描述它们.

生态学中常见的一个模型变体是

> **Example:** A predator-prey model
> $$
> \begin{aligned}
> \frac{dy}{dt} &= y(1-\alpha y)-\frac{yz}{1+\beta y},\\[2mm]
> \frac{dz}{dt} &= -z+\frac{yz}{1+\beta y},
> \end{aligned}
> $$
> where $\alpha$ and $\beta$ are positive constants.
>
> This is a system of two differential equations for the unknown functions $y(t)$ and $z(t)$, which could represent a prey species (or susceptible host) and a predator species (or infected population). Both equations involve both unknowns, with no clear way to separate them.

为了把系统写得更统一，我们可以把两个因变量打包成一个向量值函数

$$
\mathbf{u}(t)=
\begin{bmatrix}
u_1(t)\\
u_2(t)
\end{bmatrix}
=
\begin{bmatrix}
y(t)\\
z(t)
\end{bmatrix}.
$$

对应的方程组可以写成分量形式 $u_1'(t)=f_1(t,\mathbf{u})$、$u_2'(t)=f_2(t,\mathbf{u})$，从而把系统视为一个向量值微分方程.

**#2 向量值 IVP 的定义**

> **Definition:** Vector-valued IVP / IVP system
> A vector-valued first-order initial-value problem (IVP) is
> $$
> \mathbf{u}'(t)=\mathbf{f}\bigl(t,\mathbf{u}(t)\bigr),\quad a\le t\le b,\qquad \mathbf{u}(a)=\mathbf{u}_0,
> $$
> where $\mathbf{u}(t)$ is $m$-dimensional.
>
> If $\mathbf{f}(t,\mathbf{u})=\mathbf{A}(t)\mathbf{u}+\mathbf{g}(t)$, the differential equation is linear; otherwise, it is nonlinear.

我们会交替使用 "IVP system" 与 "vector-valued IVP" 这两个说法. 把若干个标量 IVP 组装成上面的向量形式，通常只需要像前面的 predator-prey 例子那样做一个恰当的向量化定义.

**#3 系统的数值解**

把标量 IVP 求解器推广到系统情形通常很直接. 以 Euler 方法为例，它在系统形式下变成

$$
\mathbf{u}_{i+1}=\mathbf{u}_i+h\,\mathbf{f}(t_i,\mathbf{u}_i),\qquad i=0,\ldots,n-1.
$$

这只是把 Euler 公式同步应用到每个分量上. 因为向量加法、标量乘法等运算与标量情形一一对应，**6-2-Euler-方法** 里的 `euler` 实现不需要改动就能用于系统；实际要改的只是：初值与右端函数都要用向量来编码.

> **Demo:** Solving the predator-prey system.
> ```Python
> import numpy as np
> import matplotlib.pyplot as plt
> from scipy.integrate import solve_ivp
>
> def predprey_rhs(t, u, alpha, beta):
>     y, z = u
>     s = (y * z) / (1.0 + beta * y)   # appears in both equations
>     return np.array([y * (1.0 - alpha * y) - s, -z + s], dtype=float)
>
> # Parameters and IVP data.
> alpha, beta = 0.1, 0.25
> u0 = np.array([1.0, 0.01])
> tspan = (0.0, 60.0)
>
> # High-accuracy reference solution.
> sol = solve_ivp(
>     lambda t, u: predprey_rhs(t, u, alpha, beta),
>     t_span=tspan,
>     y0=u0,
>     method="DOP853",
>     rtol=1e-11,
>     atol=1e-13,
>     dense_output=True,
> )
>
> tt = np.linspace(tspan[0], tspan[1], 2000)
> uu = sol.sol(tt)
> plt.plot(tt, uu[0], label="prey")
> plt.plot(tt, uu[1], label="predator")
> plt.title("Predator-prey solution")
> plt.xlabel("t")
> plt.grid(True)
> plt.legend()
> plt.show()
>
> # Compare to Euler's method on a uniform grid.
> def euler(f, tspan, u0, n):
>     a, b = float(tspan[0]), float(tspan[1])
>     h = (b - a) / n
>     t = a + h * np.arange(n + 1)
>     u = np.empty((n + 1, len(u0)), dtype=float)
>     u[0] = u0
>     for i in range(n):
>         u[i + 1] = u[i] + h * f(t[i], u[i])
>     return t, u
>
> f = lambda t, u: predprey_rhs(t, u, alpha, beta)
> t_e, u_e = euler(f, tspan, u0, n=1200)
>
> plt.plot(tt, uu[0], color="C0", lw=2, label="reference prey")
> plt.plot(tt, uu[1], color="C1", lw=2, label="reference predator")
> plt.plot(t_e[::3], u_e[::3, 0], "o", ms=2, color="k", label="Euler prey")
> plt.plot(t_e[::3], u_e[::3, 1], "o", ms=2, color="gray", label="Euler predator")
> plt.title("Predator-prey: reference vs Euler")
> plt.xlabel("t")
> plt.grid(True)
> plt.legend()
> plt.show()
> ```
> The Euler solution can lose accuracy quickly on this problem.

当系统只有两个分量时，我们常把解画在相平面 (phase plane) 中：用 $u_1$ 与 $u_2$ 作坐标轴，把时间看作曲线参数.

> **Demo:** Predator-prey in the phase plane.
> ```Python
> import numpy as np
> import matplotlib.pyplot as plt
> from scipy.integrate import solve_ivp
>
> def predprey_rhs(t, u, alpha, beta):
>     y, z = u
>     s = (y * z) / (1.0 + beta * y)
>     return np.array([y * (1.0 - alpha * y) - s, -z + s], dtype=float)
>
> alpha, beta = 0.1, 0.25
> u0 = np.array([1.0, 0.01])
> tspan = (0.0, 60.0)
>
> sol = solve_ivp(
>     lambda t, u: predprey_rhs(t, u, alpha, beta),
>     t_span=tspan,
>     y0=u0,
>     method="DOP853",
>     rtol=1e-11,
>     atol=1e-13,
>     dense_output=True,
> )
>
> tt = np.linspace(tspan[0], tspan[1], 3000)
> uu = sol.sol(tt)
> plt.plot(uu[0], uu[1])
> plt.xlabel("y")
> plt.ylabel("z")
> plt.title("Predator-prey in the phase plane")
> plt.grid(True)
> plt.axis("equal")
> plt.show()
> ```
> In the phase plane the solution can approach a periodic one, represented by a closed loop.

在本章后续内容里，我们会像在标量情形下那样陈述各种方法，但默认它们同样适用于系统. 误差分析在系统情形可能更复杂，但关于精度阶与其他性质的结论，对系统与标量都是成立的；后续给出的代码也都支持系统输入.

**#4 高阶系统如何改写为一阶系统**

一旦我们能够求解一阶 ODE 系统，也就能求解更高阶的 ODE 系统. 原因是：我们总能把高阶问题系统地改写成更高维的一阶系统.

> **Example:** Turning a second-order IVP into a first-order system
> Consider
> $$
> y''+(1+y')^3y=0,\qquad y(0)=y_0,\qquad y'(0)=0.
> $$
> Define $u_1=y$ and $u_2=y'$. Then
> $$
> u_1'=u_2,\qquad u_2'=-(1+u_2)^3u_1,
> $$
> with initial condition $u_1(0)=y_0$, $u_2(0)=0$.

下面给出一个来自力学的例子：两只摆通过同一根杆耦合.

> **Example:** Coupled pendulums
> Two identical pendulums can be modeled as the second-order system
> $$
> \begin{aligned}
> \theta_1''(t)+\gamma \theta_1'(t)+\frac{g}{L}\sin(\theta_1)+k(\theta_1-\theta_2) &= 0,\\
> \theta_2''(t)+\gamma \theta_2'(t)+\frac{g}{L}\sin(\theta_2)+k(\theta_2-\theta_1) &= 0,
> \end{aligned}
> $$
> where $\theta_1,\theta_2$ are angles, $L$ is the length of each pendulum, $\gamma$ is a friction parameter, and $k$ describes the coupling torque.
>
> Introduce
> $$
> u_1=\theta_1,\quad u_2=\theta_2,\quad u_3=\theta_1',\quad u_4=\theta_2'.
> $$
> Then the first-order system is
> $$
> \begin{aligned}
> u_1' &= u_3,\\
> u_2' &= u_4,\\
> u_3' &= -\gamma u_3-\frac{g}{L}\sin(u_1)+k(u_2-u_1),\\
> u_4' &= -\gamma u_4-\frac{g}{L}\sin(u_2)+k(u_1-u_2).
> \end{aligned}
> $$
> To complete the IVP, one specifies $\theta_1(0)$, $\theta_1'(0)$, $\theta_2(0)$, and $\theta_2'(0)$.

上面两个例子展示的技巧总是可用. 若原问题里有一个标量因变量 $y$，并且在方程中出现了 $y$ 的若干阶导数，那么我们就为 $y$、$y'$、$y''$ 等各引入一个分量，直到 (但不包含) $y$ 出现的最高阶导数为止. 对原系统里的每个标量因变量都做同样的处理.

最终得到的一阶系统应当满足：

- 每个标量初值条件对应新向量 $\mathbf{u}$ 的一个分量.
- 很多方程来自 "低阶导数之间的平凡关系" (例如 $u_1'=u_2$ 这类).
- 剩下的方程来自原来的高阶方程 (提供最高阶导数的表达式).
- 一阶系统的标量方程个数与未知的一阶变量个数一致.

**#5 耦合摆的一个演示**

> **Demo:** Coupled pendulums (uncoupled vs coupled).
> ```Python
> import numpy as np
> import matplotlib.pyplot as plt
> from scipy.integrate import solve_ivp
>
> def couple_rhs(t, u, gamma, L, k):
>     g = 9.8
>     u1, u2, u3, u4 = u
>     return np.array([
>         u3,
>         u4,
>         -gamma*u3 - (g/L)*np.sin(u1) + k*(u2 - u1),
>         -gamma*u4 - (g/L)*np.sin(u2) + k*(u1 - u2),
>     ], dtype=float)
>
> u0 = np.array([1.25, -0.5, 0.0, 0.0])   # pulled opposite directions, released from rest
> tspan = (0.0, 50.0)
> gamma, L = 0.0, 0.5
>
> # Uncoupled case (k=0).
> sol0 = solve_ivp(
>     lambda t, u: couple_rhs(t, u, gamma, L, 0.0),
>     t_span=tspan,
>     y0=u0,
>     method="DOP853",
>     rtol=1e-11,
>     atol=1e-13,
>     dense_output=True,
> )
> tt = np.linspace(20.0, 50.0, 2000)
> uu = sol0.sol(tt)
> plt.plot(tt, uu[0], label=r"$\\theta_1$")
> plt.plot(tt, uu[1], label=r"$\\theta_2$")
> plt.title("Uncoupled pendulums")
> plt.xlabel("t")
> plt.grid(True)
> plt.legend()
> plt.show()
>
> # Coupled case (k=1).
> sol1 = solve_ivp(
>     lambda t, u: couple_rhs(t, u, gamma, L, 1.0),
>     t_span=tspan,
>     y0=u0,
>     method="DOP853",
>     rtol=1e-11,
>     atol=1e-13,
>     dense_output=True,
> )
> uu = sol1.sol(tt)
> plt.plot(tt, uu[0], label=r"$\\theta_1$")
> plt.plot(tt, uu[1], label=r"$\\theta_2$")
> plt.title("Coupled pendulums")
> plt.xlabel("t")
> plt.grid(True)
> plt.legend()
> plt.show()
> ```
> With coupling activated, the pendulums can swap energy back and forth.
