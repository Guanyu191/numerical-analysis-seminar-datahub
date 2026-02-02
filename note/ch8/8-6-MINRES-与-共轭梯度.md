# 8-6-MINRES-与-共轭梯度 (MINRES and conjugate gradients)

这是一份数值计算学习笔记，参考了 Tobin A. Driscoll and Richard J. Braun 的教材 [*Fundamentals of Numerical Computation* (2023)](https://tobydriscoll.net/fnc-julia/home.html).

> 这份笔记主要是翻译了原文内容，并删改或重新表述部分内容，希望能进一步减少初学者的学习障碍.

**#1 对称性带来的特化: Lanczos 迭代**

我们已经多次看到：某些矩阵性质会显著简化线性代数问题. 其中最重要的一条是

$$
\mathbf{A}^{*}=\mathbf{A},
$$

也就是 $\mathbf{A}$ 为 Hermitian (实数情形就是对称矩阵). Arnoldi 迭代在这一情形下有一个特别有用的特化. 本节会描述由此得到的算法，但不展开实现细节.

从 Arnoldi 的关键恒等式

$$
\mathbf{A}\mathbf{Q}_m=\mathbf{Q}_{m+1}\mathbf{H}_m
$$

出发，左乘 $\mathbf{Q}_m^{*}$ 得

$$
\mathbf{Q}_m^{*}\mathbf{A}\mathbf{Q}_m
=
\mathbf{Q}_m^{*}\mathbf{Q}_{m+1}\mathbf{H}_m
=
\widetilde{\mathbf{H}}_m,
$$

其中 $\widetilde{\mathbf{H}}_m$ 是取 $\mathbf{H}_m$ 的前 $m$ 行得到的 $m\times m$ 矩阵.

如果 $\mathbf{A}$ 是 Hermitian，那么左边 $\mathbf{Q}_m^{*}\mathbf{A}\mathbf{Q}_m$ 也是 Hermitian，因此 $\widetilde{\mathbf{H}}_m$ 也必为 Hermitian. 另一方面，$\widetilde{\mathbf{H}}_m$ 还具有上 Hessenberg 结构. 把 "Hermitian" 与 "上 Hessenberg" 合在一起，会强迫它变成三对角矩阵.

> **Observation:** Arnoldi becomes tridiagonal for Hermitian matrices
> For a hermitian (or real symmetric) matrix, the upper Hessenberg matrix produced by the Arnoldi iteration is tridiagonal.

因此 Arnoldi 迭代的逐列关系会简化为一个三项递推：

$$
\mathbf{A}\mathbf{q}_m
=
H_{m-1,m}\mathbf{q}_{m-1}+H_{m,m}\mathbf{q}_m+H_{m+1,m}\mathbf{q}_{m+1}.
$$

和推导 Arnoldi 时一样：当已知前 $m$ 个向量时，我们可以解出 $\mathbf{H}$ 的第 $m$ 列系数，并进一步得到 $\mathbf{q}_{m+1}$. 得到的过程称为 Lanczos 迭代.

它最重要的工程优势是: Arnoldi 每步需要 $O(m)$ 次投影消去，而 Lanczos 每步只需要 $O(1)$ 次 (因为只涉及相邻两列). 这使得对对称问题而言，通常不需要像 GMRES 那样依赖重启来控制成本.

> **Note:** 理论上，Lanczos 看起来只是 Arnoldi 的一个小改动. 但想得到数值稳定的实现，还需要额外的分析与技巧. 这里不展开细节.

**#2 MINRES**

当 $\mathbf{A}$ 为 Hermitian，且 Arnoldi 被 Lanczos 取代后，与 GMRES 对应的算法称为 MINRES.

MINRES 和 GMRES 一样：它在逐步增长的 Krylov 子空间中最小化残差 $\|\mathbf{b}-\mathbf{A}\mathbf{x}\|_2$.

MINRES 的理论性质也比 GMRES 更容易处理. 下面的结果依赖较深的逼近理论. 回忆: Hermitian 矩阵的特征值都是实数.

> **Theorem:** Convergence of MINRES (indefinite case)
> Suppose $\mathbf{A}$ is hermitian, invertible, and indefinite. Divide its eigenvalues into positive and negative sets $\Lambda_{+}$ and $\Lambda_{-}$, and define
> $$
> \kappa_{+}=\frac{\max_{\lambda\in\Lambda_{+}}|\lambda|}{\min_{\lambda\in\Lambda_{+}}|\lambda|},
> \qquad
> \kappa_{-}=\frac{\max_{\lambda\in\Lambda_{-}}|\lambda|}{\min_{\lambda\in\Lambda_{-}}|\lambda|}.
> $$
> Then $\mathbf{x}_m$, the $m$th solution estimate of MINRES, satisfies
> $$
> \frac{\|\mathbf{r}_m\|_2}{\|\mathbf{b}\|_2}
> \le
> \left(\frac{\sqrt{\kappa_{+}\kappa_{-}}-1}{\sqrt{\kappa_{+}\kappa_{-}}+1}\right)^{\lfloor m/2\rfloor}.
> $$

> **Example:** A guaranteed iteration count from the bound
> Suppose $\mathbf{A}$ has $\kappa_{+}=60$ and $\kappa_{-}=15$. Then to guarantee a reduction in the relative residual of $10^{-3}$, we require
> $$
> \left(\frac{\sqrt{900}-1}{\sqrt{900}+1}\right)^{\lfloor m/2\rfloor}\le 10^{-3},
> $$
> which implies $m\ge 208$.

> **Demo:** MINRES on an indefinite system (SciPy)
> We build a symmetric indefinite matrix, estimate the bound parameter, and compare the observed convergence to the bound.
>
> ```Python
> import numpy as np
> import matplotlib.pyplot as plt
> import scipy.sparse as sp
> import scipy.sparse.linalg as spla
>
> def poisson_2d(n):
>     # Discrete 2D -Delta on an n-by-n interior grid, scaled by h^{-2}.
>     h = 1.0 / (n + 1)
>     e = np.ones(n)
>     T = sp.diags([-e, 4*e, -e], [-1, 0, 1], shape=(n, n), format="csr") / (h*h)
>     I = sp.eye(n, format="csr")
>     S = sp.diags([-e, -e], [-1, 1], shape=(n, n), format="csr") / (h*h)
>     A = sp.kron(I, T) + sp.kron(S, I)
>     return A.tocsr()
>
> rng = np.random.default_rng(0)
>
> A0 = poisson_2d(10)          # size 100x100, SPD
> A = A0 - 20.0 * sp.eye(A0.shape[0], format="csr")  # shift to make it indefinite
>
> # Eigenvalues for bound parameters (dense is fine at size 100).
> lam = np.linalg.eigvalsh(A.toarray())
> neg = lam < 0
> pos = ~neg
> kappa_minus = np.max(np.abs(lam[neg])) / np.min(np.abs(lam[neg]))
> kappa_plus = np.max(lam[pos]) / np.min(lam[pos])
> rho = (np.sqrt(kappa_minus * kappa_plus) - 1) / (np.sqrt(kappa_minus * kappa_plus) + 1)
> print("kappa_- =", kappa_minus, " kappa_+ =", kappa_plus, " rho =", rho)
>
> b = rng.standard_normal(A.shape[0])
>
> hist = []
> def cb(xk):
>     rk = b - A @ xk
>     hist.append(np.linalg.norm(rk, 2))
>
> x, info = spla.minres(A, b, rtol=1e-10, maxiter=51, callback=cb)
> relres = np.array(hist) / np.linalg.norm(b, 2)
>
> m = np.arange(len(relres))
> plt.semilogy(m, relres, label="observed")
> plt.semilogy(m, rho ** (m / 2.0), ls="--", label="upper bound")
> plt.xlabel("m")
> plt.ylabel("relative residual")
> plt.title("Convergence of MINRES")
> plt.grid(True, which="both", alpha=0.3)
> plt.legend()
> plt.show()
> ```
>
> The theoretical bound is often pessimistic, but it can still predict the slow linear phase controlled by the eigenvalues.

**#3 共轭梯度 (Conjugate gradients, CG)**

当在对称性之外，我们还有正定性时，会得到最著名的 Krylov 子空间方法之一：共轭梯度 (CG).

设 $\mathbf{A}$ 为 Hermitian 正定 (HPD). 则 $\mathbf{A}$ 有 Cholesky 分解，在复数情形下可写为 $\mathbf{A}=\mathbf{R}^{*}\mathbf{R}$. 因此对任意向量 $\mathbf{u}$，

$$
\mathbf{u}^{*}\mathbf{A}\mathbf{u}
=
(\mathbf{R}\mathbf{u})^{*}(\mathbf{R}\mathbf{u})
=
\|\mathbf{R}\mathbf{u}\|_2^2,
$$

它非负，且在 $\mathbf{A}$ 非奇异时仅当 $\mathbf{u}=\mathbf{0}$ 才为 0. 这允许我们定义一个由 $\mathbf{A}$ 诱导的向量范数：

$$
\|\mathbf{u}\|_{\mathbf{A}}=(\mathbf{u}^{*}\mathbf{A}\mathbf{u})^{1/2}.
$$

> **Definition:** Method of conjugate gradients (CG)
> Suppose $\mathbf{A}$ is hermitian and positive definite. For each $m=1,2,3,\dots$, minimize $\|\mathbf{x}_m-\mathbf{x}\|_{\mathbf{A}}$ for $\mathbf{x}$ in the Krylov subspace $\mathcal{K}_m$.

**#4 收敛、条件数与一个常见经验量级**

CG 与 MINRES 的收敛都强烈依赖于 $\mathbf{A}$ 的特征值. 在 HPD 情形下，特征值为正实数，且等于奇异值，因此 2-范数条件数 $\kappa$ 就是 "最大特征值 / 最小特征值".

> **Theorem:** MINRES and CG convergence (definite case)
> Let $\mathbf{A}$ be real and SPD with 2-norm condition number $\kappa$. For MINRES define
> $$
> R(m)=\frac{\|\mathbf{r}_m\|_2}{\|\mathbf{b}\|_2},
> $$
> and for CG define
> $$
> R(m)=\frac{\|\mathbf{x}_m-\mathbf{x}\|_{\mathbf{A}}}{\|\mathbf{x}\|_{\mathbf{A}}},
> $$
> where $\mathbf{r}_m$ and $\mathbf{x}_m$ are the residual and solution approximation associated with the space $\mathcal{K}_m$. Then
> $$
> R(m)\le 2\left(\frac{\sqrt{\kappa}-1}{\sqrt{\kappa}+1}\right)^{m}.
> $$

该定理对 MINRES 与 CG 给出相似的收敛刻画，只是衡量量不同：一个是残差，一个是 $\mathbf{A}$-范数意义下的误差. 由于它给的是上界，在实践里两种方法常常都能看到早期的超线性阶段.

当 $\kappa$ 很大时，上界中的因子

$$
\frac{\sqrt{\kappa}-1}{\sqrt{\kappa}+1}
$$

会非常接近 1，因此预期收敛会变慢. 把该因子做 Taylor 近似，可以得到一个常用经验量级.

> **Observation:** Iteration count grows like $\sqrt{\kappa}$
> As a rule of thumb, the number of iterations required for MINRES or CG to converge is $O(\sqrt{\kappa})$, where $\kappa$ is the condition number.

> **Demo:** Comparing MINRES and CG on SPD problems (SciPy)
> We solve diagonal SPD systems with different condition numbers and compare the residual convergence of MINRES and CG.
>
> ```Python
> import numpy as np
> import matplotlib.pyplot as plt
> import scipy.sparse as sp
> import scipy.sparse.linalg as spla
>
> def run_methods(kappa, n=2000, tol=1e-6, maxiter=1000):
>     # Diagonal SPD matrix with eigenvalues in [1/kappa, 1].
>     lam = np.linspace(1.0 / kappa, 1.0, n)
>     A = sp.diags(lam, 0, format="csr")
>     x_true = np.arange(1, n + 1, dtype=float) / n
>     b = A @ x_true
>
>     curves = {}
>
>     for name, solver in [("minres", spla.minres), ("cg", spla.cg)]:
>         hist = []
>         def cb(xk):
>             rk = b - A @ xk
>             hist.append(np.linalg.norm(rk) / np.linalg.norm(b))
>         if name == "minres":
>             x, info = solver(A, b, rtol=tol, maxiter=maxiter, callback=cb)
>         else:
>             x, info = solver(A, b, rtol=tol, atol=0.0, maxiter=maxiter, callback=cb)
>         curves[name] = np.array(hist)
>
>     return curves
>
> plt.figure(figsize=(7, 3))
> for kappa in [100, 2500]:
>     curves = run_methods(kappa)
>     for name, relres in curves.items():
>         plt.semilogy(relres, label=f"{name}, kappa={kappa}")
> plt.title("Convergence of MINRES and CG (diagonal SPD)")
> plt.xlabel("iteration")
> plt.ylabel("relative residual norm")
> plt.grid(True, which="both", alpha=0.3)
> plt.legend()
> plt.show()
> ```
>
> Increasing the condition number typically slows convergence, and the iteration count often scales roughly like $\sqrt{\kappa}$.
