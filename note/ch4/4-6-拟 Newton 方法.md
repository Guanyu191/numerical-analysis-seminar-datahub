# 4-6-拟 Newton 方法 (Quasi-Newton methods)

这是一份数值计算学习笔记，参考了 Tobin A. Driscoll and Richard J. Braun 的教材 [*Fundamentals of Numerical Computation* (2023)](https://tobydriscoll.net/fnc-julia/home.html).

> 这份笔记主要是翻译了原文内容，并删改或重新表述部分内容，希望能进一步减少初学者的学习障碍.

**#1 为什么需要 quasi-Newton**

Newton 法是求解方程与最优化问题的基础方法，但它的 "纯" 形式并不理想. 最主要的两个问题是：计算 Jacobian 矩阵既麻烦又昂贵，而且从许多初始点出发时迭代会发散. **quasi-Newton methods** (拟 Newton 方法) 会在不改变基本思路的前提下，对 Newton 法做一些改造来缓解这些问题.

**#2 用有限差分近似 Jacobian**

在一维情形，我们已经见过一个替代 "直接算导数" 的办法. 事后回看，可以把割线法理解为：把 Newton 公式中的 $f'(x_k)$ 换成差商

$$
f'(x_k)\approx \frac{f(x_k)-f(x_{k-1})}{x_k-x_{k-1}}.
$$

如果序列 $\{x_k\}$ 收敛到某个根 $r$，那么上面的差商也会收敛到 $f'(r)$.

在方程组情形，替换 Jacobian 的评估更麻烦：我们需要对 $n$ 个变量分别求偏导. 回忆在 **4-5-非线性方程组的 Newton 法** 里，Jacobian 的第 $j$ 列可以写成

$$
\mathbf{J}(\mathbf{x})\mathbf{e}_j
=
\begin{bmatrix}
\frac{\partial f_1}{\partial x_j}\\
\frac{\partial f_2}{\partial x_j}\\
\vdots\\
\frac{\partial f_n}{\partial x_j}
\end{bmatrix},
$$

其中 $\mathbf{e}_j$ 表示单位矩阵的第 $j$ 列 (也就是第 $j$ 个标准基向量). 受一维差商启发，我们只扰动 $x_j$，其余变量保持不变，用下面的有限差分来近似：

$$
\mathbf{J}(\mathbf{x})\mathbf{e}_j
\approx
\frac{\mathbf{f}(\mathbf{x}+\delta \mathbf{e}_j)-\mathbf{f}(\mathbf{x})}{\delta},
\quad j=1,\ldots,n.
$$

关于步长 $\delta$ 的选取，在第 5 章会给出解释：通常把 $\delta$ 选在 $\sqrt{\epsilon}$ 的量级，其中 $\epsilon$ 表示对 $\mathbf{f}$ 的评估中预期的噪声或不确定性. 如果噪声唯一来源是浮点舍入误差，那么可以用 $\delta\approx \sqrt{\epsilon_{\rm mach}}$.

> **Function:** fdjac
> **Finite-difference approximation of a Jacobian**
> ```Python
> import numpy as np
>
> def fdjac(f, x0, y0=None):
>     """
>     Compute a finite-difference approximation of the Jacobian matrix of f at x0.
>     If y0 = f(x0) is provided, it will be reused to save one function evaluation.
>     """
>     x0 = np.asarray(x0, dtype=float)
>     if y0 is None:
>         y0 = f(x0)
>     y0 = np.asarray(y0, dtype=float)
>
>     delta = np.sqrt(np.finfo(float).eps) * max(np.linalg.norm(x0), 1.0)
>
>     # Scalar input/output (still return a 1x1 "Jacobian"). 
>     if x0.ndim == 0 or x0.size == 1:
>         J = (np.asarray(f(x0 + delta), dtype=float) - y0) / delta
>         return np.atleast_2d(J)
>
>     m, n = y0.size, x0.size
>     J = np.zeros((m, n), dtype=float)
>     x = x0.astype(float, copy=True)
>     for j in range(n):
>         x[j] += delta
>         J[:, j] = (np.asarray(f(x), dtype=float) - y0) / delta
>         x[j] -= delta
>     return J
> ```
>
> **Note:** 代码里用 `max(||x0||, 1)` 做尺度归一化，是为了避免在 $x_0$ 很小 (例如接近 $\mathbf{0}$) 时步长过小.

**#3 Broyden 更新**

有限差分 Jacobian 很容易实现，但从上面的公式可以看出：每次迭代都要额外评估 $\mathbf{f}$ 共 $n$ 次 (每一列一次). 在某些应用里，这会慢到无法接受. 更糟的是，随着迭代逐渐收敛，根的近似值变化不大，Jacobian 矩阵也应当变化不大；此时重复做这些函数评估就显得浪费. 这正适合引入 "approximate approximation" 的思想：寻找一种足够好、但更便宜的方式，用上一轮的信息来更新 Jacobian.

回忆 Newton 迭代来自线性模型

$$
\mathbf{f}(\mathbf{x}_{k+1})
\approx
\mathbf{f}(\mathbf{x}_k)+\mathbf{J}(\mathbf{x}_k)(\mathbf{x}_{k+1}-\mathbf{x}_k)
=
\boldsymbol{0}.
$$

令 Newton 步 $\mathbf{s}_k=\mathbf{x}_{k+1}-\mathbf{x}_k$，令 $\mathbf{y}_k=\mathbf{f}(\mathbf{x}_k)$. 现在我们用一个矩阵 $\mathbf{A}_k$ 来近似 Jacobian，那么 $\mathbf{s}_k$ 由下面的线性系统定义：

$$
\mathbf{A}_k\mathbf{s}_k=-\mathbf{y}_k.
$$

得到 $\mathbf{x}_{k+1}$ 后，我们希望把 $\mathbf{A}_k$ 更新为 $\mathbf{A}_{k+1}$. 一维割线法会假设导数满足差商关系；在向量情形中，更自然的改写方式是

$$
\mathbf{y}_{k+1}-\mathbf{y}_k
=
\mathbf{A}_{k+1}(\mathbf{x}_{k+1}-\mathbf{x}_k)
=
\mathbf{A}_{k+1}\mathbf{s}_k.
$$

这仍不足以唯一确定 $\mathbf{A}_{k+1}$. 但如果我们额外要求 $\mathbf{A}_{k+1}-\mathbf{A}_k$ 是一个秩为 1 的矩阵，就会得到下面的更新公式.

> **Definition:** Broyden update formula
> $$
> \mathbf{A}_{k+1}
> =
> \mathbf{A}_k
> +
> \frac{1}{\mathbf{s}_k^{T}\mathbf{s}_k}
> \bigl(\mathbf{y}_{k+1}-\mathbf{y}_k-\mathbf{A}_k\mathbf{s}_k\bigr)\mathbf{s}_k^{T}.
> $$

注意 $\mathbf{A}_{k+1}-\mathbf{A}_k$ 与两个向量的外积成正比，因此计算它不需要额外评估 $\mathbf{f}$. 在合理的假设下，只要使用 Broyden 更新，得到的 $\{\mathbf{x}_k\}$ 会呈现超线性收敛；即便 $\mathbf{A}_k$ 序列本身未必收敛到真正的 Jacobian.

实际使用中，通常会用有限差分在 $k=1$ 处初始化 Jacobian. 如果某一步用更新后的 Jacobian 计算出的步长没能让残差下降得足够多，那么我们就用有限差分重新初始化 Jacobian 并重新计算该步.

**#4 Levenberg 方法**

许多求根问题里最困难的部分，是找到一个能让迭代进入收敛区的初始点. 在 Newton 迭代中，无论我们使用精确 Jacobian、有限差分 Jacobian，还是迭代更新得到的 Jacobian，隐含构造的线性模型都会在远离根时变得越来越不准确，最终几乎无法反映真实的函数行为.

与其在每一步都试图精细地分析线性模型的准确性，实践中更常见的是采用简单的策略. 例如：当我们根据线性模型算出一个建议步长之后，先问一个二值问题：走这一步是否会让情况更好？因为我们要找的是 $\mathbf{f}$ 的根，所以可以用 **backward error** (残差) $\|\mathbf{f}\|$ 来量化这个问题：如果残差没有变小，我们就拒绝这一步，并寻找替代步长.

下面考虑带参数的线性系统：

$$
\bigl(\mathbf{A}_k^{T}\mathbf{A}_k+\lambda \mathbf{I}\bigr)\mathbf{s}_k
=
-\mathbf{A}_k^{T}\mathbf{y}_k.
$$

> **Algorithm:** Levenberg's method
> Given $\mathbf{f}$, a starting value $\mathbf{x}_1$, and a scalar $\lambda$, for each $k=1,2,3,\ldots$
> 1. Compute $\mathbf{y}_k=\mathbf{f}(\mathbf{x}_k)$, and let $\mathbf{A}_k$ be an exact or approximate Jacobian matrix.
> 2. Solve $\bigl(\mathbf{A}_k^{T}\mathbf{A}_k+\lambda \mathbf{I}\bigr)\mathbf{s}_k=-\mathbf{A}_k^{T}\mathbf{y}_k$ for $\mathbf{s}_k$.
> 3. Let $\hat{\mathbf{x}}=\mathbf{x}_k+\mathbf{s}_k$.
> 4. If the residual is reduced at $\hat{\mathbf{x}}$, then let $\mathbf{x}_{k+1}=\hat{\mathbf{x}}$.
> 5. Update $\lambda$ and update $\mathbf{A}_k$ to $\mathbf{A}_{k+1}$.

这个方法的直觉可以从极端情形看出来.

如果 $\lambda=0$，那么上式变为

$$
\mathbf{A}_k^{T}\mathbf{A}_k\mathbf{s}_k=-\mathbf{A}_k^{T}\mathbf{y}_k,
$$

它等价于常规的 Newton 或 quasi-Newton 步 $\mathbf{A}_k\mathbf{s}_k=-\mathbf{y}_k$.

另一方面，当 $\lambda\to\infty$ 时，上式逼近

$$
\lambda \mathbf{s}_k=-\mathbf{A}_k^{T}\mathbf{y}_k.
$$

为了解释这个式子，定义标量残差函数

$$
\phi(\mathbf{x})
=
\mathbf{f}(\mathbf{x})^{T}\mathbf{f}(\mathbf{x})
=
\|\mathbf{f}(\mathbf{x})\|^2.
$$

求解 $\mathbf{f}(\mathbf{x})=\boldsymbol{0}$ 等价于最小化 $\phi(\mathbf{x})$. 计算可得它的梯度为

$$
\nabla \phi(\mathbf{x})
=
2\mathbf{J}(\mathbf{x})^{T}\mathbf{f}(\mathbf{x}).
$$

因此当 $\mathbf{A}_k=\mathbf{J}(\mathbf{x}_k)$ 时，$\mathbf{s}_k$ 与 $-\nabla\phi(\mathbf{x}_k)$ 同向，也就是最速下降方向. 在非病态的情形下，沿这个方向走一个足够小的步长，会保证 $\phi$ 下降，也就是残差下降.

换句话说，参数 $\lambda$ 让我们能在两种行为之间平滑切换：在根附近，用接近 Newton 的步长可获得很快的局部收敛；在远离根时，用更接近梯度下降的小步长来保证迭代取得进展.

**#5 一个组合实现**

把有限差分、Jacobian 更新、以及 Levenberg 步长控制结合起来，是相互独立的决策. 下面的实现展示了它们如何组合在一起，也是目前为止逻辑最复杂的例程之一.

每次循环会先用上面的 Levenberg 线性系统提出一个步长 $\mathbf{s}_k$. 然后检查使用该步长是否会使 $\|\mathbf{f}\|$ 下降：如果下降，就接受新点，降低 $\lambda$ 以更接近 Newton 行为，并用 Broyden 公式廉价更新 Jacobian；如果不下降，就增大 $\lambda$ 以更接近梯度下降，并在必要时用有限差分重新计算 Jacobian.

> **Function:** levenberg
> **Quasi-Newton method for nonlinear systems**
> ```Python
> import numpy as np
>
> def levenberg(f, x1, maxiter=40, ftol=1e-12, xtol=1e-12):
>     """
>     Use Levenberg's quasi-Newton iteration to find a root of f starting from x1.
>     Returns the history of root estimates as a list of numpy arrays.
>     """
>     x1 = np.asarray(x1, dtype=float)
>     xs = [x1.astype(float, copy=True)]
>
>     yk = np.asarray(f(xs[0]), dtype=float)
>     s = np.inf
>
>     A = fdjac(f, xs[0], yk)   # start with a FD Jacobian
>     jac_is_new = True
>     lam = 10.0
>
>     k = 0
>     while (np.linalg.norm(s) > xtol) and (np.linalg.norm(yk) > ftol):
>         # Compute the proposed step from (A^T A + lam I) s = -A^T y.
>         n = A.shape[1]
>         B = A.T @ A + lam * np.eye(n)
>         z = A.T @ yk
>         s = -np.linalg.solve(B, z)
>
>         xhat = xs[k] + s
>         yhat = np.asarray(f(xhat), dtype=float)
>
>         if np.linalg.norm(yhat) < np.linalg.norm(yk):    # accept
>             lam = lam / 10.0   # get closer to Newton
>
>             # Broyden update of the Jacobian.
>             denom = float(s @ s)
>             if denom != 0.0:
>                 A = A + np.outer((yhat - yk - A @ s), s) / denom
>                 jac_is_new = False
>
>             xs.append(xhat)
>             yk = yhat
>             k += 1
>         else:                                        # don't accept
>             lam = 4.0 * lam     # get closer to gradient descent
>
>             # Re-initialize the Jacobian if it's out of date.
>             if not jac_is_new:
>                 A = fdjac(f, xs[k], yk)
>                 jac_is_new = True
>
>         if k == maxiter:
>             break
>
>     return xs
> ```

在某些问题里，这个简单逻辑可能会让 $\lambda$ 在大与小之间来回振荡；在实践中有更复杂也更稳健的控制策略. 另外，上面的线性系统通常会再做一点修改，从而得到著名的 **Levenberg-Marquardt algorithm**，它在某些问题中表现更好.

> **Demo:** Solving a nonlinear system without coding a Jacobian.
> ```Python
> import numpy as np
>
> def f(x):
>     x1, x2, x3 = x
>     return np.array([
>         np.exp(x2 - x1) - 2.0,
>         x1 * x2 + x3,
>         x2 * x3 + x1**2 - x2,
>     ])
>
> x1 = np.array([0.0, 0.0, 0.0])
> xs = levenberg(f, x1, maxiter=40, ftol=1e-12, xtol=1e-12)
> r = xs[-1]
> print("backward error =", np.linalg.norm(f(r)))
>
> # Compare convergence rate via log(error) ratios.
> logerr = [np.log(np.linalg.norm(r - xk)) for xk in xs[:-1]]
> ratios = [logerr[k+1] / logerr[k] for k in range(len(logerr) - 1)]
> print("ratios =", ratios)
> ```
> The ratios typically settle between 1 and 2, indicating a convergence rate between linear and quadratic.

> **Note:** 这个方法的一个主要收益是：我们只需要写 $\mathbf{f}$，而不需要显式写 Jacobian (对照 **4-5-非线性方程组的 Newton 法** 里的 `newtonsys`). 同时，收敛速度通常介于线性与二次之间，类似于一维割线法.
