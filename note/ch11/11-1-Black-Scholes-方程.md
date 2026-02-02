# 11-1-Black-Scholes-方程 (Black-Scholes equation)

这是一份数值计算学习笔记，参考了 Tobin A. Driscoll and Richard J. Braun 的教材 [*Fundamentals of Numerical Computation* (2023)](https://tobydriscoll.net/fnc-julia/home.html).

> 这份笔记主要是翻译了原文内容，并删改或重新表述部分内容，希望能进一步减少初学者的学习障碍.

**#1 期权定价与 Black-Scholes 方程**

设在 $t=0$ 时我们买入一只股票，其股价记为 $S(t)$. 在之后的某个时刻，如果 $S(t)>S(0)$，我们卖出就能获利；但如果 $S(t)<S(0)$，我们就会亏损，甚至可能亏掉全部投入. 因此我们可能希望降低这种风险.

一种做法是买入看涨期权 (**call option**) 而不是直接买股票. 这是一份合约：它规定了固定的到期时间 (或称 **strike time**) $T$ 与执行价 (或称 **strike price**) $K$，使我们在 $T$ 时刻有权以价格 $K$ 从合约发行方买入该股票.

如果 $S(T)>K$，那么我们可以用 $K$ 买入并立刻按 $S(T)$ 卖出，利润为 $S(T)-K$. 另一方面，如果 $S(T)\le K$，行权没有优势，因为股票的市场价值不如合约保证的买入价. 但此时我们最多损失买入期权时付出的费用. 这些讨论可以用一个收益函数 (**payoff function**) 来概括：

$$
H(S)=\max\{S-K,0\}=(S-K)_{+}.
$$

问题变成：这份期权合约的 "公平" 市场价格是多少. 这个问题的一个著名近似答案来自 Black-Scholes 方程：

$$
\frac{\partial v}{\partial t}+\frac{1}{2}\sigma^2S^2\frac{\partial^2 v}{\partial S^2}+rS\frac{\partial v}{\partial S}-rv=0.
$$

其中 $r$ 是无风险利率 (例如非常安全的投资可以获得的利率)，$\sigma$ 是股票的波动率 (可以理解为收益率的标准差量级). 在 Black-Scholes 模型中，$r$ 与 $\sigma$ 都被假设为已知常数.

期权价值 $v(S,t)$ 同时依赖时间 $t$ 与股价 $S$，并且它们可以被视为相互独立的自变量. 在到期时刻，收益条件 $v(S,T)=H(S)$ 被施加；我们的目标是把方程 "沿时间反方向" 解回去，从而得到 $v(S,0)$. 这类方程是随时间演化的偏微分方程 (**time-dependent PDE**, 也称 evolutionary PDE).

**#2 初始条件与边界条件**

Black-Scholes 方程的自变量是 $t$ 与 $S$. 由于它对时间只有一阶导数，我们需要给出解的一个初始值.

为了让时间按更常见的方式 "向前流动"，我们引入新变量 $\eta=T-t$. 此时方程变为

$$
-v_{\eta}+\frac{1}{2}\sigma^2S^2v_{SS}+rSv_S-rv=0,
$$

并且定义在 $0\le \eta\le T$ 上. 这里采用了常见记号：用下标表示偏导数.

在新时间变量下，初始条件为

$$
v(S,0)=H(S),
$$

目标是求出所有 $\eta>0$ 时刻的 $v(S,\eta)$. 之后我们会把 $\eta$ 重新记作 $t$，但需要记住：在这一应用里，这个 $t$ 的方向与真实时间相反.

接下来还需要给出 $S$ 的定义域. 股票价格不可能为负，因此左端点是 $S=0$. 理论上 $S$ 没有上界；为了便于计算，我们把定义域截断到某个正数 $S_{\max}$.

由于方程对 $S$ 有二阶导数，我们还需要在 $S$ 的两端各给出一个条件. 在 $S=0$ 时，股票与期权都没有价值，因此令 $v=0$. 对于 $S=S_{\max}$，根据收益函数 $H$ 的形状，初始时刻 $v$ 在 $S_{\max}$ 处的斜率为 1；我们选择对所有时间都强制这一条件. 边界条件总结为

$$
v(0,t)=0,\qquad v_S(S_{\max},t)=1.
$$

左端是齐次 Dirichlet 条件，右端是非齐次 Neumann 条件.

因此，这个问题由 PDE、初始条件、以及边界条件共同组成. 这种 "初始-边界值问题" 常被称为 **initial-boundary-value problem** (IBVP). 它的数值求解通常同时需要初值问题 (IVP) 与边值问题 (BVP) 的方法成分.

**#3 热方程**

Black-Scholes 方程可以通过变量替换化为更简单的典型 PDE.

> **Definition:** Heat equation
>
> The heat equation or diffusion equation in one dimension is
> $$
> u_t = k u_{xx},
> $$
> where $k$ is a constant diffusion coefficient.

热方程是抛物型 PDE (**parabolic PDE**) 的代表模型. 扩散过程的一个关键特征是：速度与解的梯度成正比，因此解中的快速变化会很快被 "抹平".

> **Observation:**
> Solutions of the heat equation smooth out quickly and become as flat as the boundary conditions allow.

> **Example:**
> Consider the following diffusion problem on $0\le x\le 1$:
> $$
> \begin{aligned}
> \text{PDE: }& u_t = u_{xx}, && 0<x<1,\ t>0,\\
> \text{BC: }& u(0,t)=u(1,t)=0, && t\ge 0,\\
> \text{IC: }& u(x,0)=\sin(\pi x), && 0\le x\le 1.
> \end{aligned}
> $$
> We can verify that
> $$
> \hat u(x,t)=e^{-\pi^2 t}\sin(\pi x)
> $$
> satisfies the PDE, the initial condition, and the boundary conditions. First compute
> $$
> \frac{\partial \hat u}{\partial t} = -\pi^2 e^{-\pi^2 t}\sin(\pi x),\qquad
> \frac{\partial^2 \hat u}{\partial x^2} = -\pi^2 e^{-\pi^2 t}\sin(\pi x).
> $$
> These are equal, so the PDE holds. Substituting $x=0$ or $x=1$ gives $\hat u=0$, so the boundary conditions hold. Finally, substituting $t=0$ recovers $\hat u(x,0)=\sin(\pi x)$.

在某些情形下，我们能写出热方程 (从而也包括 Black-Scholes 方程) 的显式解. 这些公式不仅提供很多洞见，有时也能直接用来求解. 但这类显式解通常很 "脆弱"：模型只要稍作修改，它们就可能变得不适用甚至失效. 相比之下，数值方法往往对这类改动更稳健.

**#4 一个朴素的数值解法**

回到 Black-Scholes 方程，把它改写为

$$
v_t=\frac{1}{2}\sigma^2S^2v_{SS}+rSv_S-rv,
$$

并配上初始条件 $v(S,0)=H(S)$ 与边界条件 $v(0,t)=0$、$v_S(S_{\max},t)=1$. 我们先不做任何变量变换技巧或其他额外洞见，直接尝试数值求解.

令

$$
x_i=ih,\quad h=\frac{S_{\max}}{m},\quad i=0,\dots,m,\qquad
t_j=j\tau,\quad \tau=\frac{T}{n},\quad j=0,\dots,n.
$$

这里我们用更常见的 $x,t$ 代替 $S,\eta$. 得到一个网格函数 $\mathbf{V}$，其元素 $V_{ij}\approx v(x_i,t_j)$. 由初始条件，我们设置对所有 $i$ 都有 $V_{i,0}=H(x_i)$.

暂时假设空间索引 $i$ 在两侧都没有边界. 用简单的有限差分替换各项导数，可以得到

$$
\frac{V_{i,j+1}-V_{i,j}}{\tau}
=
\frac{\sigma^2 x_i^2}{2}\frac{V_{i+1,j}-2V_{i,j}+V_{i-1,j}}{h^2}
+r x_i\frac{V_{i+1,j}-V_{i-1,j}}{2h}
-rV_{i,j}.
$$

把它整理为显式推进形式：

$$
\begin{aligned}
V_{i,j+1}
&=V_{i,j}
+\frac{\lambda\sigma^2x_i^2}{2}\Bigl(V_{i+1,j}-2V_{i,j}+V_{i-1,j}\Bigr)
+\frac{r x_i\mu}{2}\Bigl(V_{i+1,j}-V_{i-1,j}\Bigr)
-r\tau V_{i,j},
\end{aligned}
$$

其中

$$
\lambda=\frac{\tau}{h^2},\qquad \mu=\frac{\tau}{h}.
$$

把 $j=0$ 代入后，右端对每个 $i$ 都是已知量，因此可以得到 $V_{i,1}$. 然后再用 $j=1$ 得到 $V_{i,2}$，如此向前推进.

现在把 $x$ 的边界条件加入. $V_{0,j+1}$ 恒为 0，所以在 $i=0$ 处我们不需要用显式格式去计算. 难点在右端 $i=m$ (即 $x_i=S_{\max}$)：上式会用到并不存在的 "虚拟点" $V_{m+1,j}$. 这时需要使用 Neumann 条件. 如果 $V_{m+1,j}$ 存在，我们可以把 $v_S(S_{\max},t)=1$ 离散为

$$
\frac{V_{m+1,j}-V_{m-1,j}}{2h}=1,
$$

从而解出虚拟点 $V_{m+1,j}$，并把它代回显式格式的右端.

> **Demo:** A finite-difference solver for the Black-Scholes IBVP
> We set parameters, discretize, and then march forward in (transformed) time.
>
> ```Python
> import numpy as np
> import matplotlib.pyplot as plt
>
> def solve_black_scholes_fd(Smax=8.0, T=6.0, K=3.0, sigma=0.06, r=0.08, m=200, n=1000):
>     h = Smax / m
>     x = h * np.arange(m + 1)
>     tau = T / n
>     lam = tau / h**2
>     mu = tau / h
>
>     V = np.zeros((m + 1, n + 1))
>     V[:, 0] = np.maximum(0.0, x - K)  # payoff H(x)
>
>     for j in range(n):
>         # Fictitious value from Neumann condition: (V_{m+1}-V_{m-1})/(2h)=1.
>         Vfict = 2.0 * h + V[m - 1, j]
>         Vj = np.empty(m + 2)
>         Vj[:-1] = V[:, j]
>         Vj[-1] = Vfict
>
>         # Dirichlet at x=0 keeps V[0, j+1] = 0.
>         for i in range(1, m + 1):
>             diff1 = Vj[i + 1] - Vj[i - 1]
>             diff2 = Vj[i + 1] - 2.0 * Vj[i] + Vj[i - 1]
>             V[i, j + 1] = (
>                 Vj[i]
>                 + (lam * sigma**2 * x[i] ** 2 / 2.0) * diff2
>                 + (r * x[i] * mu / 2.0) * diff1
>                 - r * tau * Vj[i]
>             )
>
>     t = np.linspace(0.0, T, n + 1)
>     return x, t, V
>
> x, t, V = solve_black_scholes_fd(T=6.0)
>
> idx = np.arange(0, V.shape[1], 250)
> for j in idx:
>     plt.plot(x, V[:, j], label=f\"t = {t[j]:.2f}\")
> plt.title(\"Black-Scholes solution (finite differences)\")
> plt.xlabel(\"stock price\")
> plt.ylabel(\"option value\")
> plt.grid(True, alpha=0.3)
> plt.legend(loc=\"upper left\")
> plt.show()
> ```
>
> The curves are easy to interpret if we remember that the transformed time variable means time until strike.
>
> > **Note:** 如果我们希望把演化过程做成动画，可以用 `matplotlib.animation` 并保存为 mp4，但通常需要系统安装 `ffmpeg`.

结果很好解释：此处的时间变量代表 "距离到期还有多久". 当我们离到期时刻很近时，期权价值应该接近收益函数 $H$. 当距离到期还有更久时，股价有更大的概率上涨，因此对固定的 $S$ 来说，期权价值会变大.

上面的 Demo 看起来一切顺利. 但问题很快会出现.

> **Demo:** The same scheme can become unstable for a longer horizon
> We repeat the same computation but extend the simulation time to $T=8$.
>
> ```Python
> x, t, V = solve_black_scholes_fd(T=8.0)
> print(\"max(V)\", V.max())  # huge values indicate blow-up
>
> idx = np.arange(0, V.shape[1], 250)
> for j in idx:
>     plt.plot(x, V[:, j], label=f\"t = {t[j]:.2f}\")
> plt.title(\"This 'solution' is nonsense (blow-up)\")
> plt.xlabel(\"stock price\")
> plt.ylabel(\"option value\")
> plt.ylim(0, 6)
> plt.grid(True, alpha=0.3)
> plt.legend(loc=\"upper left\")
> plt.show()
> ```

这个所谓的 "解" 会出现爆炸式增长，显然不可信. 这暗示着数值格式存在不稳定性；关于不稳定性的来源，我们会在本章后续讨论. 在此之前，我们先考虑一条更一般、更稳健的策略来求解随时间演化的 PDE.
