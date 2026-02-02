# 6-5-自适应-Runge-Kutta (Adaptive Runge-Kutta)

这是一份数值计算学习笔记，参考了 Tobin A. Driscoll and Richard J. Braun 的教材 [*Fundamentals of Numerical Computation* (2023)](https://tobydriscoll.net/fnc-julia/home.html).

> 这份笔记主要是翻译了原文内容，并删改或重新表述部分内容，希望能进一步减少初学者的学习障碍.

**#1 为什么需要自适应步长**

初值问题方法的推导与分析通常假设固定步长 $h$. 虽然 **6-2-Euler-方法** 的误差定理保证了当 $h\to 0$ 时误差具有 $O(h^p)$ 的行为，但这个界里包含不可知的常数，因此并不能直接告诉我们在某个具体步长下误差大约是多少.

更重要的是，固定步长在很多问题里并不高效：就像 **5-7-自适应积分** 一样，理想策略往往是 "哪里变化快就用小步，哪里变化慢就用大步". 对 IVP 来说，难点在于右端项 $f(t,u)$ 依赖于解本身，因此细节与积分问题有很大不同.

**#2 步长预测：用两种阶的 RK 同时估计误差**

设我们在同一个时间步上，同时运行两种不同阶的 RK 方法：一个阶为 $p$，得到 $u_{i+1}$；另一个阶为 $p+1$，得到 $\tilde u_{i+1}$. 在大多数情况下，$\tilde u_{i+1}$ 会显著更准确，因此可以用两者差值来估计低阶方法的局部误差：

$$
E_i(h)=|\tilde u_{i+1}-u_{i+1}|.
$$

对向量 IVP 则用范数替代绝对值.

接下来问一个 "回看" 问题：如果我们希望误差目标是 $\epsilon$，那刚才应该用多大的步长？假设局部截断误差满足 $E_i(h)\approx C h^{p+1}$，那么对 $qh$ 有

$$
E_i(qh)\approx q^{p+1}E_i(h).
$$

把它与目标 $E_i(qh)\approx \epsilon$ 匹配，得到经典的步长因子预测：

$$
q\approx \left(\frac{\epsilon}{E_i(h)}\right)^{1/(p+1)}.
$$

有些观点认为，应该控制更接近全局误差的量 $E_i(h)/h$，则会导向

$$
q\le \left(\frac{\epsilon}{E_i(h)}\right)^{1/p}.
$$

两种选择各有拥护者；实际经验中，前一种 (基于 $1/(p+1)$ ) 更常被采用.

**#3 自适应步长算法框架**

> **Algorithm:** Adaptive step size for an IVP
> Given a solution estimate $u_i$ at $t=t_i$ and a step size $h$:
> 1. Produce estimates $u_{i+1}$ and $\tilde u_{i+1}$, and estimate the error.
> 2. If the error is small enough, adopt $\tilde u_{i+1}$ as the solution value at $t=t_i+h$, then increment $i$.
> 3. Replace $h$ by $qh$, with $q$ given by a step size prediction rule.
> 4. Repeat until $t=b$.

这只是框架，还需要明确：怎么用最小成本得到一对不同阶的结果 (Step 1)，以及如何定义 "误差足够小" 与如何避免步长失控.

**#4 Embedded Runge-Kutta：共享 stages 的两套公式**

如果我们真的分别运行两种 RK 方法来得到 $u_{i+1}$ 与 $\tilde u_{i+1}$，那么每个时间步要做的 $f(t,u)$ 评估次数会显著增加. 为了降低自适应的边际成本，我们使用 embedded RK formulas：两套不同阶的公式共享同一批内部 stages，只是用不同的线性组合得到不同阶的输出.

一个常用的嵌入对是 Bogacki–Shampine (BS23)：

$$
\begin{array}{c|cccc}
\hline
0\\
\frac{1}{2} & \frac{1}{2}\\
\frac{3}{4} & 0 & \frac{3}{4}\\
1 & \frac{2}{9} & \frac{1}{3} & \frac{4}{9}\\
\hline
 & \frac{2}{9} & \frac{1}{3} & \frac{4}{9} & 0\\
 & \frac{7}{24} & \frac{1}{4} & \frac{1}{3} & \frac{1}{8}\\
\hline
\end{array}
$$

表的上半部分给出 4 个 stages. 最后两行给出两种不同的加权组合：一行产生三阶近似，另一行产生二阶近似；两者差值就提供误差估计.

**#5 一个实现: rk23**

下面给出一个 embedded 二/三阶自适应求解器 `rk23`. 它采用 BS23 公式对，并使用 $q=(\epsilon/E)^{1/3}$ 类型的步长更新，同时加入保守因子与步长增长限制.

> **Function:** rk23
> **Adaptive IVP solver based on embedded RK formulas**
> ```Python
> import numpy as np
>
> def rk23(f, tspan, u0, tol, maxiter=10_000):
>     """
>     rk23(f, tspan, u0, tol)
>
>     Adaptive embedded RK23 method for u' = f(t,u).
>     Returns arrays (t, u).
>     """
>     a, b = float(tspan[0]), float(tspan[1])
>     t = [a]
>
>     u0 = np.asarray(u0, dtype=float)
>     u = [u0.copy() if u0.shape != () else float(u0)]
>
>     # Initial step size guess (p=2 -> exponent 1/(p+1)=1/3).
>     h = 0.5 * float(tol) ** (1.0 / 3.0)
>
>     # First stage (FSAL will reuse the last stage of accepted step).
>     s1 = np.asarray(f(t[0], np.asarray(u[0], dtype=float)), dtype=float)
>
>     i = 0
>     while t[i] < b and i < maxiter:
>         # Detect underflow of the step size.
>         if t[i] + h == t[i]:
>             print(f"Warning: stepsize too small near t={t[i]}")
>             break
>
>         # Don't step past the end.
>         h = min(h, b - t[i])
>
>         ui = np.asarray(u[i], dtype=float)
>
>         # BS23 stages (s1 is already known).
>         s2 = np.asarray(f(t[i] + 0.5 * h, ui + 0.5 * h * s1), dtype=float)
>         s3 = np.asarray(f(t[i] + 0.75 * h, ui + 0.75 * h * s2), dtype=float)
>
>         # 3rd-order solution (b = [2/9, 1/3, 4/9, 0]).
>         unew = ui + h * (2 * s1 + 3 * s2 + 4 * s3) / 9.0
>
>         # 4th stage at the endpoint (needed for error estimate and FSAL).
>         s4 = np.asarray(f(t[i] + h, unew), dtype=float)
>
>         # Difference between the embedded 3rd- and 2nd-order formulas.
>         err = h * (-5 * s1 / 72.0 + s2 / 12.0 + s3 / 9.0 - s4 / 8.0)
>         E = float(np.linalg.norm(err, ord=np.inf))
>
>         maxerr = float(tol) * (1.0 + float(np.linalg.norm(ui, ord=np.inf)))
>
>         # Accept the proposed step?
>         if E < maxerr:
>             t.append(t[i] + h)
>             u.append(unew.copy() if unew.shape != () else float(unew))
>             i += 1
>             s1 = s4  # FSAL property
>
>         # Adjust step size (protect against E=0).
>         if E == 0.0:
>             q = 4.0
>         else:
>             q = 0.8 * (maxerr / E) ** (1.0 / 3.0)
>             q = min(q, 4.0)
>         h = q * h
>
>     return np.array(t, dtype=float), np.array(u, dtype=float)
> ```

**#6 Demo：自适应步长如何响应解的特征**

> **Demo:** Adaptive IVP solution
> The solution of $u' = e^{t-u\\sin u}$ changes abruptly near $t\\approx 2.4$, and adaptive step sizes vary over orders of magnitude.
>
> ```Python
> import numpy as np
> import matplotlib.pyplot as plt
>
> f = lambda t, u: np.exp(t - u * np.sin(u))
> t, u = rk23(f, (0.0, 5.0), u0=0.0, tol=1e-5)
>
> plt.plot(t, u, marker="o", ms=2)
> plt.xlabel("t")
> plt.ylabel("u(t)")
> plt.title("Adaptive IVP solution")
> plt.grid(True, alpha=0.3)
> plt.show()
>
> dt = np.diff(t)
> plt.semilogy(t[:-1], dt)
> plt.xlabel("t")
> plt.ylabel("step size")
> plt.title("Adaptive step sizes")
> plt.grid(True, which="both", alpha=0.3)
> plt.show()
>
> print("minimum step size =", dt.min())
> print("average step size =", dt.mean())
> ```
> Compared to a uniform step size chosen for the most difficult region, adaptivity can reduce the number of steps dramatically.

> **Demo:** Finite-time blowup
> For an IVP with finite-time blowup, the adaptively chosen steps become very small as we approach the singularity, and failure of the adaptivity can indicate where it occurs.
>
> ```Python
> import numpy as np
> import matplotlib.pyplot as plt
>
> f = lambda t, u: (t + u) ** 2
> t, u = rk23(f, (0.0, 1.0), u0=1.0, tol=1e-5)
>
> plt.semilogy(t, u)
> plt.xlabel("t")
> plt.ylabel("u(t)")
> plt.title("Finite-time blowup (adaptive RK23)")
> plt.grid(True, which="both", alpha=0.3)
> plt.show()
> ```

有些问题里，自适应步长会清晰对应到解的某些可见特征 (例如快速跃迁、接近 blowup). 但也存在所谓 stiff problems：时间步看起来 "不合理地小"，却并不能从解的可见变化直接解释. 这类问题需要不同类型的求解器，在后续章节会继续讨论.
