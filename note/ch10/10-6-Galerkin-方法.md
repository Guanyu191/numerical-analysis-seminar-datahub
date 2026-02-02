# 10-6-Galerkin-方法 (The Galerkin method)

这是一份数值计算学习笔记，参考了 Tobin A. Driscoll and Richard J. Braun 的教材 [*Fundamentals of Numerical Computation* (2023)](https://tobydriscoll.net/fnc-julia/home.html).

> 这份笔记主要是翻译了原文内容，并删改或重新表述部分内容，希望能进一步减少初学者的学习障碍.

**#1 从配点到 Galerkin**

在有限差分里，我们定义了一种配点方法：要求微分方程的近似在一组有限节点上成立. 这一节给出一种替代思路：不再用差分近似导数，而是用积分条件来刻画问题.

这一节的推导只限于线性边值问题

$$
u'' = p(x)u' + q(x)u + r(x),\qquad a\le x\le b,\qquad u(a)=0,\ u(b)=0.
$$

我们假设它可以等价地写成

$$
-\frac{d}{dx}\bigl[c(x)u'(x)\bigr] + s(x)u(x) = f(x),\qquad u(a)=0,\ u(b)=0.
$$

原则上，这样的变换总是可行的 (至少在形式上). 非零边界值也能并入. 这个框架也可以适配 Neumann 条件. 对非线性问题，我们通常会用 Newton 迭代把它转化成一串线性问题来求解.

**#2 弱形式 (Weak formulation)**

令上面的方程乘上一个通用函数 $\psi(x)$ (称为测试函数 **test function**)，并在 $x$ 上积分：

$$
\int_a^b f(x)\psi(x)\,dx
=
\int_a^b\bigl[-(c(x)u'(x))' \psi(x) + s(x)u(x)\psi(x)\bigr]\,dx.
$$

对第一项做分部积分可得

$$
\int_a^b f(x)\psi(x)\,dx
=
\bigl[-c(x)u'(x)\psi(x)\bigr]_a^b
+\int_a^b\bigl[c(x)u'(x)\psi'(x) + s(x)u(x)\psi(x)\bigr]\,dx.
$$

现在对测试函数做一个重要且方便的约束：要求 $\psi(a)=\psi(b)=0$. 这样边界项消失，我们得到

$$
\int_a^b \bigl[c(x)u'(x)\psi'(x) + s(x)u(x)\psi(x)\bigr]\,dx
=
\int_a^b f(x)\psi(x)\,dx,
$$

这被称为原微分方程的弱形式 (**weak form**).

> **Definition:** Weak solution
> If $u(x)$ is a function such that the weak form is satisfied for all valid choices of $\psi$, we say that $u$ is a weak solution of the BVP.

任何满足原微分方程的解 (可以称为强形式 **strong form**) 都是弱解，但反过来不一定成立. 虽然弱形式看起来更绕，但在很多数学模型里，它反而更基础.

**#3 Galerkin 条件**

目标是用一个有限维问题去逼近弱形式. 取一组线性无关的函数 $\phi_1,\dots,\phi_m$，并满足 $\phi_i(a)=\phi_i(b)=0$. 如果我们把测试函数限制在它们的线性组合里

$$
\psi(x)=\sum_{i=1}^{m} z_i\phi_i(x),
$$

代入弱形式并整理，可写成

$$
\sum_{i=1}^{m} z_i
\left[
\int_a^b \bigl(c(x)u'(x)\phi_i'(x) + s(x)u(x)\phi_i(x)\bigr)\,dx
\;-\;
\int_a^b f(x)\phi_i(x)\,dx
\right]
=0.
$$

一种满足这个条件的方式是让括号中的项对每个 $i$ 都为零：

$$
\int_a^b \bigl(c(x)u'(x)\phi_i'(x) + s(x)u(x)\phi_i(x)\bigr)\,dx
=
\int_a^b f(x)\phi_i(x)\,dx,\qquad i=1,\dots,m.
$$

由于 $\phi_i$ 的线性无关性，这也是唯一的可能性，因此我们不再需要显式考虑 $z_i$.

接下来我们也用同一组基函数来表示近似解：

$$
u(x)=\sum_{j=1}^{m} w_j\phi_j(x).
$$

把它代回上面的积分条件，并把未知量按 $w_j$ 收集，可得一个线性系统

$$
(\mathbf{K}+\mathbf{M})\mathbf{w}=\mathbf{f},
$$

其中矩阵与右端向量定义为

$$
K_{ij}=\int_a^b c(x)\phi_i'(x)\phi_j'(x)\,dx,\qquad
M_{ij}=\int_a^b s(x)\phi_i(x)\phi_j(x)\,dx,
$$

$$
f_i=\int_a^b f(x)\phi_i(x)\,dx.
$$

矩阵 $\mathbf{K}$ 被称为刚度矩阵 (**stiffness matrix**)，$\mathbf{M}$ 被称为质量矩阵 (**mass matrix**). 由定义可知它们都是对称的. 最后需要做的选择是：怎么选基函数 $\phi_i$，并如何计算这些积分.

**#4 一个三角基函数的例子**

> **Example:**
> Suppose we are given $-u'' + 4u = x$ with $u(0)=u(\pi)=0$. We could choose the basis functions $\phi_k(x)=\\sin(kx)$ for $k=1,2,3$. Then
> $$
> M_{ij}=4\\int_0^{\\pi}\\sin(ix)\\sin(jx)\\,dx,
> \\qquad
> K_{ij}=ij\\int_0^{\\pi}\\cos(ix)\\cos(jx)\\,dx,
> \\qquad
> f_i=\\int_0^{\\pi}x\\sin(ix)\\,dx.
> $$
> With some calculation, one finds
> $$
> \\mathbf{M}=2\\pi\\begin{bmatrix}1&0&0\\\\0&1&0\\\\0&0&1\\end{bmatrix},
> \\qquad
> \\mathbf{K}=\\frac{\\pi}{2}\\begin{bmatrix}1&0&0\\\\0&4&0\\\\0&0&9\\end{bmatrix},
> \\qquad
> \\mathbf{f}=\\pi\\begin{bmatrix}1\\\\-1/2\\\\1/3\\end{bmatrix}.
> $$
> Solving the diagonal linear system gives an approximate solution
> $$
> \\frac{2}{5}\\sin(x) - \\frac{1}{8}\\sin(2x) + \\frac{2}{39}\\sin(3x).
> $$

**#5 有限元 (Finite elements) 与 hat 函数基**

一个更通用的选择，是使用 **5-2-分段线性插值** 中出现的分段线性 hat 函数作为基函数. 设节点为 $a=t_0<t_1<\cdots<t_n=b$，并记

$$
h_i=t_i-t_{i-1},\qquad i=1,\dots,n.
$$

取 $m=n-1$，并令 $\phi_i$ 为内部节点对应的 hat 函数：

$$
\phi_i(x)=H_i(x)=
\begin{cases}
\dfrac{x-t_{i-1}}{h_i}, & x\in[t_{i-1},t_i],\\[2mm]
\dfrac{t_{i+1}-x}{h_{i+1}}, & x\in[t_i,t_{i+1}],\\
0, & \text{otherwise}.
\end{cases}
$$

这些函数是 cardinal 的：$H_i(t_i)=1$ 且当 $i\\ne j$ 时 $H_i(t_j)=0$. 因此如果我们写

$$
u(x)=\sum_{j=1}^{n-1} u_j H_j(x),
$$

则系数 $u_j$ 就是数值解在节点 $t_j$ 的取值. 由于每个 hat 函数只在相邻两个小区间上非零，所有积分都可以拆成对每个子区间的局部积分，这会带来稀疏的矩阵结构. 这种把局部贡献拼装成全局系统的过程称为组装 (**assembly process**). 具有这种局部化结构的 Galerkin 方法就是有限元方法 (**finite element method**, FEM).

**#6 线性有限元的一个实现**

由于 hat 函数是局部支撑的，我们可以把弱形式里的积分拆成对每个子区间 $I_k=[t_{k-1},t_k]$ 的局部积分求和. 例如

$$
K_{ij}=\sum_{k=1}^{n}\int_{I_k} c(x)H_i'(x)H_j'(x)\,dx,
\qquad i,j=1,\dots,n-1,
$$

$$
M_{ij}=\sum_{k=1}^{n}\int_{I_k} s(x)H_i(x)H_j(x)\,dx,
\qquad i,j=1,\dots,n-1,
$$

$$
f_i=\sum_{k=1}^{n}\int_{I_k} f(x)H_i(x)\,dx,
\qquad i=1,\dots,n-1.
$$

由于在 $I_k$ 上只有相邻的两个 hat 函数 $H_{k-1}$ 与 $H_k$ 可能非零，因此每个区间只会对 $\mathbf{K}$ 与 $\mathbf{M}$ 的一个 $2\\times 2$ 子块产生贡献，对右端向量产生两个分量的贡献；把这些局部贡献加到全局矩阵与向量上的过程就是组装 (**assembly process**).

原则上我们可以在每个区间上做数值积分. 但这里用一个常见近似：把系数函数 $c(x)$、$s(x)$、$f(x)$ 在区间 $I_k$ 上用端点平均值替代，例如

$$
c(x)\\approx \\bar{c}_k=\\frac{c(t_{k-1})+c(t_k)}{2},\qquad x\\in I_k,
$$

类似地定义 $\bar{s}_k$、$\bar{f}_k$. 这样一来，各个局部积分只依赖节点位置与这些平均值.

在 $I_k$ 上有

$$
H_{k-1}'(x)=-\frac{1}{h_k},\qquad H_k'(x)=\frac{1}{h_k},
\qquad h_k=t_k-t_{k-1},
$$

因此刚度矩阵的局部贡献可以写成

$$
\frac{\bar{c}_k}{h_k}
\begin{bmatrix}
1 & -1\\
-1 & 1
\end{bmatrix}.
$$

类似地，质量矩阵与载荷向量在 $I_k$ 上的局部贡献分别是

$$
\frac{\bar{s}_k h_k}{6}
\begin{bmatrix}
2 & 1\\
1 & 2
\end{bmatrix},
\qquad
\frac{\bar{f}_k h_k}{2}
\begin{bmatrix}
1\\
1
\end{bmatrix}.
$$

这些就是下面代码里 `Ke`、`Me`、`fe` 的来源.

> **Demo:** Piecewise linear FEM for a linear BVP
> We implement the piecewise linear FEM for the weak form and assemble the global system from local contributions.
>
> ```Python
> import numpy as np
> import matplotlib.pyplot as plt
>
> def fem(c, s, f, a, b, n):
>     h = (b - a) / n
>     x = a + h * np.arange(n + 1)
>
>     # Local templates on each subinterval.
>     Ke = np.array([[1.0, -1.0], [-1.0, 1.0]])
>     Me = (1.0 / 6.0) * np.array([[2.0, 1.0], [1.0, 2.0]])
>     fe = 0.5 * np.array([1.0, 1.0])
>
>     cval = c(x)
>     sval = s(x)
>     fval = f(x)
>
>     cbar = 0.5 * (cval[:-1] + cval[1:])
>     sbar = 0.5 * (sval[:-1] + sval[1:])
>     fbar = 0.5 * (fval[:-1] + fval[1:])
>
>     m = n - 1  # interior unknowns
>     K = np.zeros((m, m))
>     M = np.zeros((m, m))
>     rhs = np.zeros(m)
>
>     for k in range(1, n + 1):
>         Ke_k = (cbar[k - 1] / h) * Ke
>         Me_k = (sbar[k - 1] * h) * Me
>         fe_k = (fbar[k - 1] * h) * fe
>         nodes = np.array([k - 1, k])
>
>         for a_loc in range(2):
>             j = nodes[a_loc]
>             if 1 <= j <= n - 1:
>                 J = j - 1
>                 rhs[J] += fe_k[a_loc]
>                 for b_loc in range(2):
>                     i = nodes[b_loc]
>                     if 1 <= i <= n - 1:
>                         I = i - 1
>                         K[J, I] += Ke_k[a_loc, b_loc]
>                         M[J, I] += Me_k[a_loc, b_loc]
>
>     u_int = np.linalg.solve(K + M, rhs)
>     u = np.zeros(n + 1)
>     u[1:n] = u_int
>     return x, u
> ```

> **Demo:** Solving a BVP by finite elements
> We solve
> $$
> -\\bigl(x^2 u'\\bigr)' + 4u = \\sin(\\pi x),\\qquad u(0)=u(1)=0,
> $$
> where $c(x)=x^2$, $s(x)=4$, $f(x)=\\sin(\\pi x)$.
>
> ```Python
> import numpy as np
> import matplotlib.pyplot as plt
>
> c = lambda x: x**2
> s = lambda x: 4.0 * np.ones_like(x)
> f = lambda x: np.sin(np.pi * x)
>
> x, u = fem(c, s, f, 0.0, 1.0, n=50)
> plt.plot(x, u)
> plt.xlabel("x")
> plt.ylabel("u")
> plt.title("Solution by finite elements")
> plt.grid(True, alpha=0.3)
> plt.show()
> ```
>
> Because piecewise linear interpolation on a uniform grid of size $h$ is $O(h^2)$ accurate, the FEM method based on linear interpolation as implemented here has accuracy similar to the second-order finite-difference method.
