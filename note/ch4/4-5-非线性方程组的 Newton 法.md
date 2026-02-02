# 4-5-非线性方程组的 Newton 法 (Newton for nonlinear systems)

这是一份数值计算学习笔记，参考了 Tobin A. Driscoll and Richard J. Braun 的教材 [*Fundamentals of Numerical Computation* (2023)](https://tobydriscoll.net/fnc-julia/home.html).

> 这份笔记主要是翻译了原文内容，并删改或重新表述部分内容，希望能进一步减少初学者的学习障碍.

**#1 多维求根问题**

当变量与方程都变成多维时，求根问题会困难得多.

> **Definition:** Multidimensional rootfinding problem
> Given a continuous vector-valued function $\mathbf{f}$ mapping from $\mathbb{R}^n$ into $\mathbb{R}^n$, find a vector $\mathbf{r}$ such that
> $$
> \begin{aligned}
> f_1(r_1,\dots,r_n) &= 0,\\
> f_2(r_1,\dots,r_n) &= 0,\\
> &\vdots\\
> f_n(r_1,\dots,r_n) &= 0.
> \end{aligned}
> $$

具体问题常常用标量变量和标量方程来描述，但写成上面的向量形式会更统一.

> **Example:** The steady state of interactions between the population $w(t)$ of a predator species and the population $h(t)$ of a prey species might be modeled as
> $$
> \begin{aligned}
> ah - b h w &= 0,\\
> -cw + d w h &= 0
> \end{aligned}
> $$
> for positive parameters $a,b,c,d$. To cast this in the vector form, we could define $\mathbf{x}=[h,w]$, $f_1(x_1,x_2) = ax_1 - bx_1x_2$, and $f_2(x_1,x_2)= -c x_2 + d x_1 x_2$.

上面这个例子可以手算解出来，但在实际问题里，哪怕只是证明解的存在性与唯一性，通常都很难.

**#2 线性模型与 Jacobian**

把求根方法推广到方程组时，我们仍然沿用同一条基本思路：构造一个容易处理的模型来近似原函数. 起点依然是线性模型. 我们从多元 Taylor 展开出发：

$$
\mathbf{f}(\mathbf{x}+\mathbf{h})
=
\mathbf{f}(\mathbf{x})+\mathbf{J}(\mathbf{x})\mathbf{h}+O(\|\mathbf{h}\|^2),
$$

其中 $\mathbf{J}$ 称为 $\mathbf{f}$ 的 **Jacobian matrix**，定义为

$$
\mathbf{J}(\mathbf{x})
=
\begin{bmatrix}
\rule[2mm]{0pt}{1em}\frac{\partial f_1}{\partial x_1} & \frac{\partial f_1}{\partial x_2} & \cdots & \frac{\partial f_1}{\partial x_n}\\[2mm]
\frac{\partial f_2}{\partial x_1} & \frac{\partial f_2}{\partial x_2} & \cdots & \frac{\partial f_2}{\partial x_n}\\[1mm]
\vdots & \vdots & & \vdots\\[1mm]
\rule[-3mm]{0pt}{1em}\frac{\partial f_n}{\partial x_1} & \frac{\partial f_n}{\partial x_2} & \cdots & \frac{\partial f_n}{\partial x_n}
\end{bmatrix}
=
\left[\frac{\partial f_i}{\partial x_j}\right]_{\,i,j=1,\ldots,n}.
$$

由于 Jacobian 在 Taylor 公式里扮演的角色，我们也会把 $\mathbf{J}(\mathbf{x})$ 记作 $\mathbf{f}\,'(\mathbf{x})$. 和其他导数一样，它也是 $\mathbf{x}$ 的函数.

> **Example:** Let
> $$
> \begin{aligned}
> f_1(x_1,x_2,x_3) &= -x_1\cos(x_2) - 1,\\
> f_2(x_1,x_2,x_3) &= x_1x_2 + x_3,\\
> f_3(x_1,x_2,x_3) &= e^{-x_3}\sin(x_1+x_2) + x_1^2 - x_2^2.
> \end{aligned}
> $$
> Then
> $$
> \mathbf{J}(\mathbf{x}) =
> \begin{bmatrix}
> -\cos(x_2) & x_1\sin(x_2) & 0\\
> x_2 & x_1 & 1\\
> e^{-x_3}\cos(x_1+x_2)+2x_1 & e^{-x_3}\cos(x_1+x_2)-2x_2 & -e^{-x_3}\sin(x_1+x_2)
> \end{bmatrix}.
> $$
> If we were to start writing out the terms in the Taylor expansion, we would begin with
> $$
> \begin{aligned}
> f_1(x_1+h_1,x_2+h_2,x_3+h_3)
> &=
> -x_1\cos(x_2)-1 -\cos(x_2)h_1 + x_1\sin(x_2)h_2 + O\bigl(\|\mathbf{h}\|^2\bigr),\\
> f_2(x_1+h_1,x_2+h_2,x_3+h_3)
> &=
> x_1x_2 + x_3 + x_2h_1 +x_1h_2 + h_3 + O\bigl(\|\mathbf{h}\|^2\bigr),
> \end{aligned}
> $$
> and so on.

Taylor 展开里 $\mathbf{f}(\mathbf{x})+\mathbf{J}(\mathbf{x})\mathbf{h}$ 这部分，就是 $\mathbf{f}$ 在 $\mathbf{x}$ 附近的线性项. 如果 $\mathbf{f}$ 本身就是线性的，例如 $\mathbf{f}(\mathbf{x})=\mathbf{A}\mathbf{x}-\mathbf{b}$，那么 Jacobian 就是常数矩阵 $\mathbf{A}$，高阶项也会消失.

**#3 多维 Newton 迭代**

现在我们有了向量系统的线性模型，就可以把 Newton 法推广到多维. 在根的近似值 $\mathbf{x}_k$ 处，把 $\mathbf{h}=\mathbf{x}-\mathbf{x}_k$ 代入 Taylor 线性项，得到

$$
\mathbf{f}(\mathbf{x})
\approx
\mathbf{q}(\mathbf{x})
=
\mathbf{f}(\mathbf{x}_k)+\mathbf{J}(\mathbf{x}_k)(\mathbf{x}-\mathbf{x}_k).
$$

我们把下一步迭代值 $\mathbf{x}_{k+1}$ 定义为线性模型的根：$\mathbf{q}(\mathbf{x}_{k+1})=\boldsymbol{0}$，即

$$
\boldsymbol{0}
=
\mathbf{f}(\mathbf{x}_k)+\mathbf{J}(\mathbf{x}_k)(\mathbf{x}_{k+1}-\mathbf{x}_k).
$$

整理得到

$$
\mathbf{x}_{k+1}
=
\mathbf{x}_k-\bigl[\mathbf{J}(\mathbf{x}_k)\bigr]^{-1}\mathbf{f}(\mathbf{x}_k).
$$

注意 $\mathbf{J}^{-1}\mathbf{f}$ 在这里扮演了一维情形 $f/f'$ 的角色 (在一维时两者确实一致). 但在数值计算中，我们不会去算矩阵逆，而是解线性系统.

> **Algorithm:** Multidimensional Newton's method
> Given $\mathbf{f}$ and a starting value $\mathbf{x}_1$, for each $k=1,2,3,\ldots$
> 1. Compute $\mathbf{y}_k = \mathbf{f}(\mathbf{x}_k)$ and $\mathbf{A}_k=\mathbf{f}\,'(\mathbf{x}_k)$.
> 2. Solve the linear system $\mathbf{A}_k\mathbf{s}_k = -\mathbf{y}_k$ for the **Newton step** $\mathbf{s}_k$.
> 3. Let $\mathbf{x}_{k+1} = \mathbf{x}_k + \mathbf{s}_k$.

把一维 Newton 法的级数分析推广到向量情形，可以得到：在合适条件下，如果迭代真的收敛，那么它在任意向量范数下也具有二次收敛.

**#4 实现与一个收敛性实验**

下面给出 Newton 方程组版本的一个直接实现：输入残差函数与 Jacobian，输出整个迭代历史.

> **Function:** newtonsys
> **Newton's method for a system of equations**
> ```Python
> import numpy as np
>
> def newtonsys(f, jac, x1, maxiter=40, ftol=None, xtol=None):
>     """
>     Use Newton's method to find a root of a system of equations, starting from x1.
>     The functions f and jac should return the residual vector and the Jacobian matrix.
>     Returns the history of root estimates as a list of numpy arrays.
>     """
>     x1 = np.asarray(x1, dtype=float)
>     if ftol is None:
>         ftol = 1000 * np.finfo(float).eps
>     if xtol is None:
>         xtol = 1000 * np.finfo(float).eps
>
>     xs = [x1.astype(float, copy=True)]
>     y = f(xs[0])
>     J = jac(xs[0])
>     dx = np.inf
>     k = 0
>     while (np.linalg.norm(dx) > xtol) and (np.linalg.norm(y) > ftol):
>         try:
>             dx = -np.linalg.solve(J, y)     # Newton step
>         except np.linalg.LinAlgError:
>             print("Warning: Jacobian is singular; aborting.")
>             break
>         xs.append(xs[k] + dx)           # append to history
>         k += 1
>         y = f(xs[k])
>         J = jac(xs[k])
>         if k == maxiter:
>             print("Warning: Maximum number of iterations reached.")
>             break
>     return xs
> ```

> **Demo:** Quadratic convergence in a nonlinear system.
> ```Python
> import numpy as np
>
> def func(x):
>     x1, x2, x3 = x
>     return np.array([
>         np.exp(x2 - x1) - 2.0,
>         x1 * x2 + x3,
>         x2 * x3 + x1**2 - x2,
>     ])
>
> def jac(x):
>     x1, x2, x3 = x
>     e = np.exp(x2 - x1)
>     return np.array([
>         [-e,  e,  0.0],
>         [ x2, x1, 1.0],
>         [ 2*x1, x3 - 1.0, x2],
>     ])
>
> xs = newtonsys(func, jac, x1=[0.0, 0.0, 0.0], maxiter=40, ftol=1e-14, xtol=1e-14)
> r = xs[-1]
> print("residual =", np.linalg.norm(func(r)))
>
> # Look at log(error) ratios approaching 2 for quadratic convergence.
> logerr = [np.log(np.linalg.norm(r - xk)) for xk in xs[:-1]]
> ratios = [logerr[k+1] / logerr[k] for k in range(len(logerr) - 1)]
> print("ratios =", ratios)
> ```
> The ratio tends to 2 when quadratic convergence is observed.

> **Note:** 原文用高精度算术来获得更长的收敛序列. 在双精度浮点下，Newton 法很快就会进入舍入误差主导的区间，因此 `ratios` 在后期可能不再稳定.
