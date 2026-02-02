# 8-4-Krylov-子空间 (Krylov subspaces)

这是一份数值计算学习笔记，参考了 Tobin A. Driscoll and Richard J. Braun 的教材 [*Fundamentals of Numerical Computation* (2023)](https://tobydriscoll.net/fnc-julia/home.html).

> 这份笔记主要是翻译了原文内容，并删改或重新表述部分内容，希望能进一步减少初学者的学习障碍.

**#1 Krylov 矩阵与 Krylov 子空间**

幂迭代与反迭代有一个明显的缺陷：给定一个种子向量 $\mathbf{u}$，它们生成的向量序列

$$
\mathbf{u},\ \mathbf{A}\mathbf{u},\ \mathbf{A}^2\mathbf{u},\ \dots
$$

在每一步里只使用最新的向量来构造特征向量近似. 直觉上，如果我们允许在 "所有历史向量的线性组合" 中搜索，效果应该不会更差，甚至可能显著更好.

也就是说，我们希望在下面这个矩阵的值域 (列空间) 中寻找近似解：

$$
\mathbf{K}_m=
\begin{bmatrix}
\mathbf{u} & \mathbf{A}\mathbf{u} & \mathbf{A}^2\mathbf{u} & \cdots & \mathbf{A}^{m-1}\mathbf{u}
\end{bmatrix}.
$$

> **Definition:** Krylov matrix and subspace
> Given $n\times n$ matrix $\mathbf{A}$ and $n$-vector $\mathbf{u}$, the $m$th Krylov matrix is the $n\times m$ matrix $\mathbf{K}_m$.
> The range (i.e., column space) of this matrix is the $m$th Krylov subspace $\mathcal{K}_m$.
>
> In general, we expect that the dimension of the Krylov subspace $\mathcal{K}_m$, which is the rank of $\mathbf{K}_m$, equals $m$, though it may be smaller.

> **Note:** "Krylov" 的常见英文发音近似 "kree-luv". 但在不同语境里也可能听到其他读法.

**#2 Krylov 子空间的基本性质**

Krylov 矩阵的一个吸引点是：它可以只靠重复的矩阵-向量乘法生成，因此能充分利用稀疏矩阵的 matvec 优势. 此外，它还有一些关键的数学性质.

> **Theorem:** Properties of a Krylov subspace
> Suppose $\mathbf{A}$ is $n\times n$, $0<m<n$, and a vector $\mathbf{u}$ is used to generate Krylov subspaces. If $\mathbf{x}\in \mathcal{K}_m$, then the following hold:
> 1. $\mathbf{x}=\mathbf{K}_m\mathbf{z}$ for some $\mathbf{z}\in\mathbb{C}^m$.
> 2. $\mathbf{x}\in\mathcal{K}_{m+1}$.
> 3. $\mathbf{A}\mathbf{x}\in\mathcal{K}_{m+1}$.

**#3 用 Krylov 子空间做降维**

问题 $\mathbf{A}\mathbf{x}=\mathbf{b}$ 与 $\mathbf{A}\mathbf{x}=\lambda\mathbf{x}$ 都发生在高维空间 $\mathbb{C}^n$ 中. 一种近似思路是：用一个维度远小于 $n$ 的子空间 $\mathcal{K}_m$ (其中 $m\ll n$) 来替代整个空间. 这就是 Krylov 子空间方法的核心.

以线性系统为例，我们可以用最小二乘意义来理解近似：

$$
\mathbf{A}\mathbf{x}_m \approx \mathbf{b},
\qquad \mathbf{x}_m\in \mathcal{K}_m.
$$

由上面的 Theorem，我们可令 $\mathbf{x}=\mathbf{K}_m\mathbf{z}$，从而把问题化为低维最小二乘：

$$
\min_{\mathbf{x}\in\mathcal{K}_m}\|\mathbf{A}\mathbf{x}-\mathbf{b}\|
=
\min_{\mathbf{z}\in\mathbb{C}^m}\|\mathbf{A}(\mathbf{K}_m\mathbf{z})-\mathbf{b}\|
=
\min_{\mathbf{z}\in\mathbb{C}^m}\|(\mathbf{A}\mathbf{K}_m)\mathbf{z}-\mathbf{b}\|.
$$

在这个问题里，一个自然的种子向量就是 $\mathbf{b}$.

> **Demo:** Least-squares over a Krylov subspace can stagnate
> We build $\mathbf{K}_m$ from a seed vector $\mathbf{b}$, solve the reduced least-squares problems, and observe that convergence can stagnate due to ill-conditioning of the Krylov basis.
>
> ```Python
> import numpy as np
> import matplotlib.pyplot as plt
>
> rng = np.random.default_rng(0)
>
> n = 100
> lam = 10.0 + np.arange(1, n + 1)  # prescribed eigenvalues on the diagonal
>
> # Upper-triangular matrix with known eigenvalues.
> A = np.triu(rng.random((n, n)), 1) + np.diag(lam)
> b = rng.random(n)
>
> # Build Krylov basis columns with renormalization after each matvec.
> mmax = 30
> K = np.zeros((n, mmax), dtype=float)
> K[:, 0] = b
> for j in range(mmax - 1):
>     v = A @ K[:, j]
>     nv = np.linalg.norm(v, 2)
>     K[:, j + 1] = v / nv
>
> # Solve reduced least-squares problems and record residuals.
> resid = np.zeros(mmax)
> AK_full = A @ K
> for m in range(1, mmax + 1):
>     z, *_ = np.linalg.lstsq(AK_full[:, :m], b, rcond=None)
>     x = K[:, :m] @ z
>     resid[m - 1] = np.linalg.norm(b - A @ x, 2)
>
> plt.semilogy(np.arange(mmax), resid, "o-", ms=3)
> plt.xlabel("m")
> plt.ylabel(r"$\\|b-Ax_m\\|_2$")
> plt.title("Residual for linear systems")
> plt.grid(True, which="both", alpha=0.3)
> plt.show()
> ```
>
> The residual often decreases at first, then stagnates after only a few digits have been gained.

**#4 Arnoldi 迭代：构造正交基**

上面 Demo 的停滞来自一个关键的数值缺陷: Krylov 矩阵的列向量会越来越平行 (幂迭代里也见过类似现象)，从而让 $\mathbf{K}_m$ 的条件数随 $m$ 增大而变得很大. 当我们用这样一组近似共线的向量去解最小二乘问题时，很容易发生数值消去，进而带来明显的误差.

与 "病态的基" 相反的极端是正交归一基. 设我们对 $\mathbf{K}_m$ 做 thin QR 分解：

$$
\mathbf{K}_m=\mathbf{Q}_m\mathbf{R}_m.
$$

那么 $\mathbf{Q}_m$ 的列向量 $\mathbf{q}_1,\dots,\mathbf{q}_m$ 就是 $\mathcal{K}_m$ 的正交归一基.

由上面的 Theorem 可知 $\mathbf{A}\mathbf{q}_m\in\mathcal{K}_{m+1}$，因此存在系数 $H_{1m},\dots,H_{m+1,m}$ 使得

$$
\mathbf{A}\mathbf{q}_m
=
H_{1m}\mathbf{q}_1+\cdots+H_{m+1,m}\mathbf{q}_{m+1}.
$$

利用正交归一性，我们可以取

$$
H_{im}=\mathbf{q}_i^{*}(\mathbf{A}\mathbf{q}_m),
\qquad i=1,\dots,m,
$$

然后令

$$
\mathbf{v}
=
(\mathbf{A}\mathbf{q}_m)-H_{1m}\mathbf{q}_1-\cdots-H_{mm}\mathbf{q}_m,
\qquad
H_{m+1,m}=\|\mathbf{v}\|_2,
\qquad
\mathbf{q}_{m+1}=\mathbf{v}/H_{m+1,m}.
$$

这就得到 Arnoldi 迭代.

> **Algorithm:** Arnoldi iteration
> Given matrix $\mathbf{A}$ and vector $\mathbf{u}$:
> 1. Let $\mathbf{q}_1=\mathbf{u}/\|\mathbf{u}\|$.
> 2. For $m=1,2,\dots$,
>    a. Use $H_{im}=\mathbf{q}_i^{*}(\mathbf{A}\mathbf{q}_m)$ for $i=1,\dots,m$.
>    b. Let $\mathbf{v}=(\mathbf{A}\mathbf{q}_m)-H_{1m}\mathbf{q}_1-\cdots-H_{mm}\mathbf{q}_m$.
>    c. Let $H_{m+1,m}=\|\mathbf{v}\|$.
>    d. Let $\mathbf{q}_{m+1}=\mathbf{v}/H_{m+1,m}$.
>
> The Arnoldi iteration finds nested orthonormal bases for a family of nested Krylov subspaces.

**#5 关键恒等式与上 Hessenberg 矩阵**

到目前为止，我们关注的是 $\mathbf{Q}_m$ 的列向量如何给出 Krylov 子空间的正交归一基. 但迭代过程中得到的 $H_{ij}$ 也同样重要.

把上面的关系对 $j=1,2,\dots,m$ 逐列写出来，我们会得到一个紧凑的矩阵恒等式：

$$
\mathbf{A}\mathbf{Q}_m=\mathbf{Q}_{m+1}\mathbf{H}_m,
$$

其中 $\mathbf{H}_m$ 是一个 $(m+1)\times m$ 的 "上三角加一条次对角线" 的矩阵.

> **Definition:** Upper Hessenberg matrix
> A matrix $\mathbf{H}$ is upper Hessenberg if $H_{ij}=0$ whenever $i>j+1$.

上面的恒等式是 Krylov 子空间方法中的基本恒等式.

**#6 Python 实现**

下面给出 Arnoldi 迭代的一个基础实现. 它只需要矩阵-向量乘法，因此同样可以用于稀疏矩阵.

> **Function:** arnoldi
> **Arnoldi iteration for Krylov subspaces**
> ```Python
> import numpy as np
>
> def arnoldi(A, u, m):
>     """
>     Perform the Arnoldi iteration for A starting with vector u, out to
>     the Krylov subspace of degree m.
>
>     Returns:
>       Q : (n, m+1) orthonormal basis matrix
>       H : (m+1, m) upper Hessenberg matrix
>     """
>     A = np.asarray(A)
>     u = np.asarray(u)
>     n = u.size
>     Q = np.zeros((n, m + 1), dtype=complex if np.iscomplexobj(A) or np.iscomplexobj(u) else float)
>     H = np.zeros((m + 1, m), dtype=complex if np.iscomplexobj(A) or np.iscomplexobj(u) else float)
>
>     Q[:, 0] = u / np.linalg.norm(u, 2)
>     for j in range(m):
>         v = A @ Q[:, j]
>         for i in range(j + 1):
>             H[i, j] = np.vdot(Q[:, i], v)  # conjugate dot
>             v = v - H[i, j] * Q[:, i]
>         H[j + 1, j] = np.linalg.norm(v, 2)
>         Q[:, j + 1] = v / H[j + 1, j]
>
>     return Q, H
> ```

> **Demo:** Orthonormality and span check
> We verify that $Q^{*}Q\approx I$, and that the columns of $Q$ span the same space as the corresponding Krylov matrix columns.
>
> ```Python
> import numpy as np
>
> rng = np.random.default_rng(0)
> A = rng.integers(1, 10, size=(6, 6)).astype(float)
> u = rng.standard_normal(6)
>
> Q, H = arnoldi(A, u, m=2)  # Q has 3 columns
> err_orth = np.linalg.norm(Q.conj().T @ Q - np.eye(Q.shape[1]), 2)
> print("||Q^* Q - I||_2 =", err_orth)
>
> K = np.column_stack([u, A @ u, A @ (A @ u)])
> rank_QK = np.linalg.matrix_rank(np.column_stack([Q, K]))
> print("rank([Q, K]) =", rank_QK)
> ```

> **Note:** 上面的实现对应的是逐步去投影的 Gram-Schmidt 风格 (modified Gram-Schmidt). 在精确算术下，它与更 "公式化" 的写法等价，但数值稳定性通常更好.
