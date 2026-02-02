# 6-4-Runge-Kutta-方法 (Runge-Kutta methods)

这是一份数值计算学习笔记，参考了 Tobin A. Driscoll and Richard J. Braun 的教材 [*Fundamentals of Numerical Computation* (2023)](https://tobydriscoll.net/fnc-julia/home.html).

> 这份笔记主要是翻译了原文内容，并删改或重新表述部分内容，希望能进一步减少初学者的学习障碍.

**#1 Runge-Kutta 方法的基本想法**

现在讨论初值问题里最重要、也最常用的一类方法：**Runge-Kutta (RK) methods**. 它们属于 **6-2-Euler-方法** 中定义的一步法，但通常不直接写成那种抽象形式. RK 方法通过在每个时间步内多次评估 ODE 右端函数 $f(t,u)$，把精度从一阶提升到更高阶.

**#2 一个二阶方法: improved Euler (IE2)**

从精确解的 Taylor 展开出发：

$$
\hat{u}(t_{i+1})
=
\hat{u}(t_i)+h\hat{u}'(t_i)+\frac{1}{2}h^2\hat{u}''(t_i)+O(h^3).
$$

若用 $\hat{u}'$ 替换为 $f$，并且只保留右端前两项，就得到 Euler 方法. 若想更精确，就需要估计第三项.

注意对 $\hat{u}'(t)=f(t,\hat{u}(t))$ 求导 (这里用到多元链式法则，因为 $f$ 的两个自变量都依赖于 $t$)：

$$
\hat{u}''(t)
=
\frac{df}{dt}
=
\frac{\partial f}{\partial t}
+
\frac{\partial f}{\partial u}\frac{du}{dt}
=
f_t+f_u f.
$$

> **Note:** 这里 $f_t$ 与 $f_u$ 是对 $\partial f/\partial t$ 与 $\partial f/\partial u$ 的简写，默认在点 $(t,\hat{u}(t))$ 处取值. 对 IVP system，$f_u$ 变为一个 **Jacobian** 矩阵，此时 $f_u f$ 表示矩阵-向量乘法.

把它代回 Taylor 展开，可写成

$$
\hat{u}(t_{i+1})
=
\hat{u}(t_i)
+
h\Bigl[
f(t_i,\hat{u}(t_i))
+
\frac{h}{2}f_t(t_i,\hat{u}(t_i))
+
\frac{h}{2}f(t_i,\hat{u}(t_i))f_u(t_i,\hat{u}(t_i))
\Bigr]
+
O(h^3).
$$

我们不想直接计算并编码这些偏导数，因此采用 "approximate approximation". 观察到

$$
f(t_i+\alpha,\hat{u}(t_i)+\beta)
=
f(t_i,\hat{u}(t_i))
+
\alpha f_t(t_i,\hat{u}(t_i))
+
\beta f_u(t_i,\hat{u}(t_i))
+
O(\alpha^2+|\alpha\beta|+\beta^2).
$$

若选择

$$
\alpha=\frac{h}{2},\qquad \beta=\frac{h}{2}f(t_i,\hat{u}(t_i)),
$$

那么

$$
\hat{u}(t_{i+1})
=
\hat{u}(t_i)
+
h f\!\left(t_i+\frac{h}{2},\hat{u}(t_i)+\frac{h}{2}f(t_i,\hat{u}(t_i))\right)
+
O(h^3),
$$

这就给出了一个新的二阶一步法: improved Euler.

> **Definition:** Improved Euler method (IE2)
> $$
> u_{i+1}
> =
> u_i
> +
> h\, f\!\left(t_i+\frac{h}{2},\ u_i+\frac{h}{2}f(t_i,u_i)\right).
> $$

更清楚地看它的两阶段结构：令

$$
k_1 = h f(t_i,u_i),\qquad v=u_i+\frac{1}{2}k_1,
$$

然后

$$
k_2 = h f\!\left(t_i+\frac{h}{2},v\right),\qquad u_{i+1}=u_i+k_2.
$$

由于一步误差满足 $h\tau_{i+1}=O(h^3)$，所以 $\tau_{i+1}=O(h^2)$，因此 IE2 的精度阶为 2.

**#3 一个实现**

> **Function:** ie2
> **Improved Euler method for an IVP**
> ```Python
> import numpy as np
>
> def ie2(f, tspan, u0, n):
>     """
>     Improved Euler (2nd-order Runge-Kutta) for u' = f(t,u).
>     Returns arrays (t, u).
>     """
>     a, b = float(tspan[0]), float(tspan[1])
>     h = (b - a) / n
>     t = a + h * np.arange(n + 1)
>
>     u0 = np.asarray(u0, dtype=float)
>     u = np.empty((n + 1,) + u0.shape, dtype=float)
>     u[0] = u0
>     for i in range(n):
>         k1 = h * np.asarray(f(t[i], u[i]), dtype=float)
>         v = u[i] + 0.5 * k1
>         k2 = h * np.asarray(f(t[i] + 0.5*h, v), dtype=float)
>         u[i + 1] = u[i] + k2
>     return t, u
> ```

**#4 更一般的 RK 表示与 RK4**

一个 $s$-stage 的 RK 方法写成

$$
\begin{aligned}
k_1 &= h f(t_i,u_i),\\
k_2 &= h f(t_i+c_1h,\ u_i+a_{11}k_1),\\
k_3 &= h f(t_i+c_2h,\ u_i+a_{21}k_1+a_{22}k_2),\\
&\ \ \vdots\\
k_s &= h f(t_i+c_{s-1}h,\ u_i+a_{s-1,1}k_1+\cdots+a_{s-1,s-1}k_{s-1}),\\
u_{i+1} &= u_i+b_1k_1+\cdots+b_sk_s.
\end{aligned}
$$

这个方法由 stage 数 $s$ 以及常数 $a_{ij},b_j,c_i$ 完全确定. 它们常被写成一个 Butcher 表. 例如 IE2 的 Butcher 表是

$$
\begin{array}{c c c}
\hline
0\\
\frac{1}{2} & \frac{1}{2}\\
\hline
 & 0 & 1\\
\hline
\end{array}
$$

下面再给出两个两阶段、二阶的方法 (modified Euler 与 Heun)：

$$
\begin{array}{c c c}
\hline
0\\
1 & 1\\
\hline
 & \frac{1}{2} & \frac{1}{2}\\
\hline
\end{array}
\qquad
\begin{array}{c c c}
\hline
0\\
\frac{2}{3} & \frac{2}{3}\\
\hline
 & \frac{1}{4} & \frac{3}{4}\\
\hline
\end{array}
$$

> **Note:** Euler、improved Euler (IE2)、以及 modified Euler (ME2) 是三个不同的方法.

最常用的 RK 方法是下面这个四阶方法，通常就被称作 "四阶 RK"，我们记为 RK4.

> **Definition:** Fourth-order Runge-Kutta method (RK4)
> $$
> \begin{aligned}
> k_1 &= h f(t_i,u_i),\\
> k_2 &= h f\!\left(t_i+\frac{h}{2},u_i+\frac{k_1}{2}\right),\\
> k_3 &= h f\!\left(t_i+\frac{h}{2},u_i+\frac{k_2}{2}\right),\\
> k_4 &= h f(t_i+h,u_i+k_3),\\
> u_{i+1} &= u_i+\frac{1}{6}k_1+\frac{1}{3}k_2+\frac{1}{3}k_3+\frac{1}{6}k_4.
> \end{aligned}
> $$

> **Function:** rk4
> **Fourth-order Runge-Kutta for an IVP**
> ```Python
> import numpy as np
>
> def rk4(f, tspan, u0, n):
>     """
>     Classic 4th-order Runge-Kutta for u' = f(t,u).
>     Returns arrays (t, u).
>     """
>     a, b = float(tspan[0]), float(tspan[1])
>     h = (b - a) / n
>     t = a + h * np.arange(n + 1)
>
>     u0 = np.asarray(u0, dtype=float)
>     u = np.empty((n + 1,) + u0.shape, dtype=float)
>     u[0] = u0
>     for i in range(n):
>         k1 = h * np.asarray(f(t[i], u[i]), dtype=float)
>         k2 = h * np.asarray(f(t[i] + 0.5*h, u[i] + 0.5*k1), dtype=float)
>         k3 = h * np.asarray(f(t[i] + 0.5*h, u[i] + 0.5*k2), dtype=float)
>         k4 = h * np.asarray(f(t[i] + h, u[i] + k3), dtype=float)
>         u[i + 1] = u[i] + (k1 + 2*(k2 + k3) + k4) / 6.0
>     return t, u
> ```

**#5 精度与效率**

对多阶段方法来说，每一个 stage 通常需要评估一次 $f$；因此一个 $s$-stage 方法的一步需要 $s$ 次 $f$ 的评估. 一般把 "评估 $f$ 的次数" 视为主要计算成本.

误差会随 $n$ 的增加呈几何下降，因此用更多 stage 换取更高精度阶往往划算. 但 stage 数越大，并不意味着精度阶可以无限增长：当 $s=5,6,7$ 时，最高精度阶是 $s-1$；当 $s=8,9$ 时，最高精度阶降为 $s-2$，等等. 在很多应用中，四阶被认为是足够且性价比很高的选择.

> **Demo:** Convergence comparison by f-evaluations.
> We compare IE2 and RK4 on the IVP $u'=\sin((u+t)^2)$ over $0\le t\le 4$, $u(0)=-1$.
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
> ref = solve_ivp(rhs, (a, b), [u0], method="DOP853", rtol=1e-13, atol=1e-15, dense_output=True)
>
> ns = np.round(2 * 10 ** np.arange(0, 3.1, 0.5)).astype(int)
> err_ie2 = []
> err_rk4 = []
> for n in ns:
>     t, u = ie2(f, (a, b), u0, int(n))
>     err_ie2.append(np.max(np.abs(ref.sol(t)[0] - u)))
>
>     t, u = rk4(f, (a, b), u0, int(n))
>     err_rk4.append(np.max(np.abs(ref.sol(t)[0] - u)))
>
> err_ie2 = np.array(err_ie2)
> err_rk4 = np.array(err_rk4)
>
> plt.loglog(2*ns, err_ie2, "o-", label="IE2")
> plt.loglog(4*ns, err_rk4, "o-", label="RK4")
> plt.xlabel("f-evaluations")
> plt.ylabel("inf-norm error")
> plt.title("Convergence of RK methods")
> plt.grid(True, which="both")
> plt.legend()
> plt.show()
> ```
> The fourth-order variant can be more efficient over a wide range of accuracies.
