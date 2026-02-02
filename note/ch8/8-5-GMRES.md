# 8-5-GMRES (GMRES)

这是一份数值计算学习笔记，参考了 Tobin A. Driscoll and Richard J. Braun 的教材 [*Fundamentals of Numerical Computation* (2023)](https://tobydriscoll.net/fnc-julia/home.html).

> 这份笔记主要是翻译了原文内容，并删改或重新表述部分内容，希望能进一步减少初学者的学习障碍.

**#1 Arnoldi 迭代的关键用途：解线性系统**

Arnoldi 迭代最重要的用途之一，是用来求解方阵线性系统

$$
\mathbf{A}\mathbf{x}=\mathbf{b}.
$$

在 **8-4-Krylov-子空间** 中，我们尝试用 Krylov 子空间做降维：把线性系统替换为

$$
\min_{\mathbf{x}\in\mathcal{K}_m}\|\mathbf{A}\mathbf{x}-\mathbf{b}\|_2
=
\min_{\mathbf{z}\in\mathbb{C}^m}\|\mathbf{A}\mathbf{K}_m\mathbf{z}-\mathbf{b}\|_2,
$$

其中 $\mathbf{K}_m$ 由 $\mathbf{A}$ 与种子向量 $\mathbf{b}$ 生成. 但由于 $\mathbf{K}_m$ 的列向量会逐渐变得近似共线，这个基往往非常病态，从而导致数值不稳定.

Arnoldi 迭代给出同一子空间的正交归一基. 令

$$
\mathbf{x}=\mathbf{Q}_m\mathbf{z},
$$

则问题变为

$$
\min_{\mathbf{z}\in\mathbb{C}^m}\|\mathbf{A}\mathbf{Q}_m\mathbf{z}-\mathbf{b}\|_2.
$$

由 Arnoldi 的基本恒等式

$$
\mathbf{A}\mathbf{Q}_m=\mathbf{Q}_{m+1}\mathbf{H}_m,
$$

可得

$$
\min_{\mathbf{z}\in\mathbb{C}^m}\|\mathbf{Q}_{m+1}\mathbf{H}_m\mathbf{z}-\mathbf{b}\|_2.
$$

又因为 Arnoldi 的第一列满足 $\mathbf{q}_1$ 是 $\mathbf{b}$ 的单位倍数，所以

$$
\mathbf{b}=\|\mathbf{b}\|_2\mathbf{q}_1=\|\mathbf{b}\|_2\mathbf{Q}_{m+1}\mathbf{e}_1.
$$

因此上式等价于

$$
\min_{\mathbf{z}\in\mathbb{C}^m}\left\|\mathbf{Q}_{m+1}\left(\mathbf{H}_m\mathbf{z}-\|\mathbf{b}\|_2\mathbf{e}_1\right)\right\|_2.
$$

由于 $\mathbf{Q}_{m+1}$ 的列向量正交归一，对任意 $\mathbf{w}\in\mathbb{C}^{m+1}$ 都有

$$
\|\mathbf{Q}_{m+1}\mathbf{w}\|_2=\|\mathbf{w}\|_2,
$$

于是我们把一个 $n$ 维的最小二乘问题，降成了一个 $(m+1)\times m$ 的小问题：

$$
\min_{\mathbf{z}\in\mathbb{C}^m}\left\|\mathbf{H}_m\mathbf{z}-\|\mathbf{b}\|_2\mathbf{e}_1\right\|_2.
$$

把该问题的解记为 $\mathbf{z}_m$，并令

$$
\mathbf{x}_m=\mathbf{Q}_m\mathbf{z}_m,
$$

则 $\mathbf{x}_m$ 就是 $m$-th GMRES 近似解.

**#2 GMRES 的定义**

> **Algorithm:** GMRES
> Given $n\times n$ matrix $\mathbf{A}$ and $n$-vector $\mathbf{b}$:
> For $m=1,2,\dots$, let $\mathbf{x}_m=\mathbf{Q}_m\mathbf{z}_m$, where $\mathbf{z}_m$ solves
> $$
> \min_{\mathbf{z}\in\mathbb{C}^m}\left\|\mathbf{H}_m\mathbf{z}-\|\mathbf{b}\|_2\mathbf{e}_1\right\|_2,
> $$
> and $\mathbf{Q}_m,\mathbf{H}_m$ arise from the Arnoldi iteration.

GMRES 的名字来自 "Generalized Minimum RESidual"：它用 Arnoldi 迭代在逐步增长的 Krylov 子空间里最小化残差

$$
\mathbf{r}_m=\mathbf{b}-\mathbf{A}\mathbf{x}_m.
$$

在精确算术下，如果 $m=n$ (更准确地说，当 Krylov 子空间最终覆盖整个空间) ，GMRES 应当得到精确解. 但实际目标通常是让 $\|\mathbf{r}_m\|_2$ 足够小，从而在某个 $m\ll n$ 时就停止.

> **Demo:** Residual curve for GMRES
> We repeat the linear-system experiment from **8-4-Krylov-子空间**, but now we use an Arnoldi basis so that the residual decreases smoothly toward machine precision.
>
> ```Python
> import numpy as np
> import matplotlib.pyplot as plt
>
> def arnoldi(A, u, m):
>     A = np.asarray(A)
>     u = np.asarray(u)
>     n = u.size
>     dtype = complex if np.iscomplexobj(A) or np.iscomplexobj(u) else float
>     Q = np.zeros((n, m + 1), dtype=dtype)
>     H = np.zeros((m + 1, m), dtype=dtype)
>     Q[:, 0] = u / np.linalg.norm(u, 2)
>     for j in range(m):
>         v = A @ Q[:, j]
>         for i in range(j + 1):
>             H[i, j] = np.vdot(Q[:, i], v)  # conjugate dot for complex vectors
>             v = v - H[i, j] * Q[:, i]
>         H[j + 1, j] = np.linalg.norm(v, 2)
>         Q[:, j + 1] = v / H[j + 1, j]
>     return Q, H
>
> rng = np.random.default_rng(0)
> n = 100
> lam = 10.0 + np.arange(1, n + 1)
> A = np.triu(rng.random((n, n)), 1) + np.diag(lam)
> b = rng.random(n)
>
> mmax = 60
> Q, H = arnoldi(A, b, mmax)
>
> resid = np.zeros(mmax + 1)
> resid[0] = np.linalg.norm(b, 2)
> for m in range(1, mmax + 1):
>     rhs = np.zeros(m + 1)
>     rhs[0] = np.linalg.norm(b, 2)
>     z, *_ = np.linalg.lstsq(H[: m + 1, :m], rhs, rcond=None)
>     x = Q[:, :m] @ z
>     resid[m] = np.linalg.norm(b - A @ x, 2)
>
> plt.semilogy(np.arange(mmax + 1), resid, "o-", ms=3)
> plt.xlabel("m")
> plt.ylabel("norm of mth residual")
> plt.title("Residual for GMRES")
> plt.grid(True, which="both", alpha=0.3)
> plt.show()
> ```

**#3 一个最基础的 GMRES 实现**

下面给出一个用于演示的基础实现：它每次都做一步 Arnoldi，然后解一次小的最小二乘问题来得到新的 $\mathbf{x}_m$，并记录残差范数.

> **Function:** gmres_demo
> **GMRES for a linear system (demo-only)**
> ```Python
> import numpy as np
>
> def gmres_demo(A, b, m, seed=0):
>     A = np.asarray(A)
>     b = np.asarray(b)
>     n = b.size
>
>     dtype = complex if np.iscomplexobj(A) or np.iscomplexobj(b) else float
>     Q = np.zeros((n, m + 1), dtype=dtype)
>     H = np.zeros((m + 1, m), dtype=dtype)
>
>     bnorm = np.linalg.norm(b, 2)
>     Q[:, 0] = b / bnorm
>
>     x = np.zeros(n, dtype=dtype)  # initial solution is zero
>     residual = np.zeros(m + 1, dtype=float)
>     residual[0] = bnorm
>
>     for j in range(m):
>         # Next step of Arnoldi iteration.
>         v = A @ Q[:, j]
>         for i in range(j + 1):
>             H[i, j] = np.vdot(Q[:, i], v)
>             v = v - H[i, j] * Q[:, i]
>         H[j + 1, j] = np.linalg.norm(v, 2)
>         Q[:, j + 1] = v / H[j + 1, j]
>
>         # Solve the minimum-residual problem.
>         rhs = np.zeros(j + 2)
>         rhs[0] = bnorm
>         z, *_ = np.linalg.lstsq(H[: j + 2, : j + 1], rhs, rcond=None)
>         x = Q[:, : j + 1] @ z
>         residual[j + 1] = np.linalg.norm(A @ x - b, 2)
>
>     return x, residual
> ```

**#4 收敛与重启**

由于 Krylov 子空间是嵌套的 $\mathcal{K}_m\subset \mathcal{K}_{m+1}$，因此 GMRES 的残差范数 (作为在空间上最小化得到的量) 不会随迭代增加.

但除此之外，要对 GMRES 的收敛给出普遍而有力的结论并不容易. 有时会出现次线性与超线性的阶段性收敛，且强烈依赖于矩阵的特征值分布.

GMRES 的一个工程难点是：随着 $m$ 增大，需要存储的 $\mathbf{Q}_{m+1}$ 的列数增大，$\mathbf{H}_m$ 的新元素也增多. 因此工作量与存储量通常随 $m$ 呈二次增长，某些应用中会变得不可承受. 这也是 GMRES 常配合重启 (restarting) 使用的原因.

设 $\widehat{\mathbf{x}}$ 是 $\mathbf{A}\mathbf{x}=\mathbf{b}$ 的一个近似解. 令 $\mathbf{x}=\mathbf{u}+\widehat{\mathbf{x}}$，代回原方程得

$$
\mathbf{A}\mathbf{u}
=
\mathbf{b}-\mathbf{A}\widehat{\mathbf{x}}
=
\mathbf{r},
$$

其中 $\mathbf{r}$ 是当前残差. 因此，只要我们解出 $\mathbf{A}\mathbf{u}=\mathbf{r}$，就得到对 $\widehat{\mathbf{x}}$ 的一个修正. 重启 GMRES 的核心就是：用当前残差重新生成一个低维 Krylov 子空间并继续最小化.

重启保证了每次循环的成本上界. 代价是：每次重启都会丢弃此前迭代中积累的 Krylov 子空间信息，残差最小化过程又从低维空间重新开始，从而可能显著减慢甚至导致停滞.

> **Demo:** Restarted GMRES trade-off (SciPy)
> We compare unrestarted GMRES and several restart values on a sparse Poisson matrix.
>
> ```Python
> import numpy as np
> import matplotlib.pyplot as plt
> import scipy.sparse as sp
> import scipy.sparse.linalg as spla
>
> def poisson_2d(n):
>     # 2D 5-point Laplacian on an n-by-n grid, Dirichlet boundary.
>     e = np.ones(n)
>     T = sp.diags([e, -4*e, e], [-1, 0, 1], shape=(n, n), format="csr")
>     I = sp.eye(n, format="csr")
>     A = sp.kron(I, T) + sp.kron(sp.diags([e, e], [-1, 1], shape=(n, n)), I)
>     return A.tocsr()
>
> A = poisson_2d(50)
> n = A.shape[0]
> b = np.ones(n)
>
> reltol = 1e-12
> maxiter = 100
>
> def run(restart):
>     hist = []
>     def cb(rk):
>         hist.append(float(rk))
>     x, info = spla.gmres(A, b, restart=restart, rtol=reltol, atol=0.0, maxiter=maxiter, callback=cb, callback_type="pr_norm")
>     return np.array(hist), info
>
> plt.figure()
> for restart in [n, 20, 40, 60]:
>     hist, info = run(restart)
>     plt.semilogy(hist, label=f"restart = {restart}")
> plt.title("Convergence of restarted GMRES")
> plt.xlabel("iteration")
> plt.ylabel("residual norm")
> plt.ylim(1e-8, 1e2)
> plt.grid(True, which="both", alpha=0.3)
> plt.legend()
> plt.show()
> ```
>
> Decreasing the restart value often worsens convergence per iteration, but reduces the cost per iteration. The best restart choice is hard to predict in general.

除了重启之外，也存在一些避免 GMRES/Arnoldi 迭代中计算量持续增长的变体. 比较常见的缩写包括 CGS, BiCGSTAB, QMR. 本仓库暂不展开.

> **Note:** GMRES 的前身之一是 MINRES，会在 **8-6-MINRES-与-共轭梯度** 出现.
