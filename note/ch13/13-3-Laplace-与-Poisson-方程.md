# 13-3-Laplace 与 Poisson 方程 (Laplace and Poisson equations)

这是一份数值计算学习笔记，参考了 Tobin A. Driscoll and Richard J. Braun 的教材 [*Fundamentals of Numerical Computation* (2023)](https://tobydriscoll.net/fnc-julia/home.html).

> 这份笔记主要是翻译了原文内容，并删改或重新表述部分内容，希望能进一步减少初学者的学习障碍.

**#1 Laplace 与 Poisson 方程**

考虑二维热方程

$$
u_t=u_{xx}+u_{yy}.
$$

当时间足够长时，温度分布会停止变化. 这时的稳态解满足 $u_t=0$，因此必须满足

$$
u_{xx}+u_{yy}=0,
$$

这引出了我们的第三类典型 PDE：Laplace 方程，以及更一般的 Poisson 方程. 二维 Poisson 方程写作

> **Definition:** Laplace and Poisson equations
> The two-dimensional Poisson equation is
> $$
> u_{xx}+u_{yy}=f(x,y).
> $$
> A common notation is
> $$
> \Delta u = f,
> $$
> where $\Delta$ is the Laplacian operator. If $f\equiv 0$, then the equation is the Laplace equation.

这里的 $f$ 有时被称为 **forcing function**. Laplace/Poisson 方程是椭圆型 PDE (**elliptic PDE**) 的代表. 对所有线性、常系数、且最高不超过二阶导数的 PDE，都可以分类为抛物型、双曲型或椭圆型. 在上面的 Poisson 方程里没有时间变量；虽然变量名本身是任意的，但椭圆型 PDE 常用来描述系统的稳态.

要得到一个完全确定的问题，还必须给出边界条件. 因为 $x$ 与 $y$ 都是空间变量，这是我们第一次遇到 "边界条件不是只在两个端点施加" 的情形. 这里我们只考虑在整个边界上施加 Dirichlet 条件：

$$
u(x,y)=g(x,y)\quad \text{on the boundary}.
$$

**#2 Sylvester 方程**

把未知解用矩形网格上的取值表示为矩阵 $\mathbf{U}=\operatorname{mtx}(u)$. 对 $x$ 与 $y$ 方向分别选取二阶导数的差分矩阵或谱微分矩阵 $\mathbf{D}_{xx}$ 与 $\mathbf{D}_{yy}$. 则 Poisson 方程的离散形式为

$$
\mathbf{D}_{xx}\mathbf{U}+\mathbf{U}\mathbf{D}_{yy}^T=\mathbf{F},
$$

其中 $\mathbf{F}=\operatorname{mtx}(f)$. 这是一个未知矩阵同时出现在左乘与右乘中的矩阵方程，称为 Sylvester 方程 (**Sylvester equation**). 为了求解它，我们需要一个新的矩阵运算：Kronecker 积.

**#3 Kronecker 积**

> **Definition:** Kronecker product
> Let $\mathbf{A}$ be $m\times n$ and $\mathbf{B}$ be $p\times q$. The Kronecker product $\mathbf{A}\otimes\mathbf{B}$ is the $mp\times nq$ matrix given by
> $$
> \mathbf{A}\otimes\mathbf{B}
> =
> \begin{bmatrix}
> A_{11}\mathbf{B} & A_{12}\mathbf{B} & \cdots & A_{1n}\mathbf{B}\\
> A_{21}\mathbf{B} & A_{22}\mathbf{B} & \cdots & A_{2n}\mathbf{B}\\
> \vdots & \vdots & & \vdots\\
> A_{m1}\mathbf{B} & A_{m2}\mathbf{B} & \cdots & A_{mn}\mathbf{B}
> \end{bmatrix}.
> $$

> **Demo:** Kronecker product in NumPy
> ```Python
> import numpy as np
>
> A = np.array([[1, 2], [-2, 0]])
> B = np.array([[1, 10, 100], [-5, 5, 3]])
>
> print(np.kron(A, B))
> ```
>
> 输出是一个 $4\times 6$ 矩阵，它由 $A$ 的每个元素乘上整块 $B$ 拼成.

Kronecker 积满足很多看起来 "顺理成章" 的恒等式：

> **Theorem:** Kronecker product identities
> For matrices of compatible sizes, the following hold:
>
> 1. $\mathbf{A}\otimes(\mathbf{B}+\mathbf{C})=\mathbf{A}\otimes\mathbf{B}+\mathbf{A}\otimes\mathbf{C}$.
> 2. $(\mathbf{A}+\mathbf{B})\otimes\mathbf{C}=\mathbf{A}\otimes\mathbf{C}+\mathbf{B}\otimes\mathbf{C}$.
> 3. $(\mathbf{A}\otimes\mathbf{B})\otimes\mathbf{C}=\mathbf{A}\otimes(\mathbf{B}\otimes\mathbf{C})$.
> 4. $(\mathbf{A}\otimes\mathbf{B})^T=\mathbf{A}^T\otimes\mathbf{B}^T$.
> 5. $(\mathbf{A}\otimes\mathbf{B})^{-1}=\mathbf{A}^{-1}\otimes\mathbf{B}^{-1}$.
> 6. $(\mathbf{A}\otimes\mathbf{B})(\mathbf{C}\otimes\mathbf{D})=(\mathbf{A}\mathbf{C})\otimes(\mathbf{B}\mathbf{D})$.

对 **13-2-二维扩散与对流** 中定义的 `vec` 操作，且各矩阵尺寸可乘时，还有恒等式

$$
\operatorname{vec}(\mathbf{A}\mathbf{B}\mathbf{C}^T)=(\mathbf{C}\otimes\mathbf{A})\,\operatorname{vec}(\mathbf{B}).
$$

**#4 把 Poisson 方程改写为线性方程组**

我们用上面的 `vec` 恒等式把 Sylvester 形式改写成标准线性系统. 先把

$$
\mathbf{D}_{xx}\mathbf{U}+\mathbf{U}\mathbf{D}_{yy}^T=\mathbf{F}
$$

补上单位矩阵，写成

$$
\mathbf{D}_{xx}\mathbf{U}\mathbf{I}_y+\mathbf{I}_x\mathbf{U}\mathbf{D}_{yy}^T=\mathbf{F},
$$

其中 $\mathbf{I}_x$ 与 $\mathbf{I}_y$ 分别是 $(m+1)\times(m+1)$ 与 $(n+1)\times(n+1)$ 的单位矩阵. 对两边取 `vec` 并应用
$
\operatorname{vec}(\mathbf{A}\mathbf{B}\mathbf{C}^T)=(\mathbf{C}\otimes\mathbf{A})\operatorname{vec}(\mathbf{B})
$
可得

$$
\bigl[(\mathbf{I}_y\otimes\mathbf{D}_{xx})+(\mathbf{D}_{yy}\otimes\mathbf{I}_x)\bigr]\,\operatorname{vec}(\mathbf{U})
=\operatorname{vec}(\mathbf{F}).
$$

这已经是标准线性方程组 $\mathbf{A}\mathbf{u}=\mathbf{b}$ 的形式，其中未知向量 $\mathbf{u}=\operatorname{vec}(\mathbf{U})$ 的长度是 $N=(m+1)(n+1)$.

**#5 Dirichlet 边界条件作为线性约束**

到目前为止，我们还没有把边界条件施加到离散系统里. 处理方式与一维边值问题类似：对每个边界网格点，我们用 "把该点的值设为边界给定值" 来替换原本的 PDE 配点方程.

更具体地说，设边界网格点在 `vec` 编号下对应的索引集合为 $\mathcal{B}\subset\{1,\dots,N\}$. 对每个 $i\in\mathcal{B}$，我们把系统 $\mathbf{A}\mathbf{u}=\mathbf{b}$ 的第 $i$ 行替换为

$$
\mathbf{e}_i^T\mathbf{u}=g(x_i,y_i),
$$

其中 $\mathbf{e}_i$ 是第 $i$ 个标准基向量. 这相当于把 $\mathbf{A}$ 的对应行替换为单位矩阵的对应行；向量 $\mathbf{b}$ 的对应分量替换为边界值.

**#6 Demo：稀疏结构与边界行替换**

> **Demo:** Building the Poisson linear system and applying Dirichlet rows
> ```Python
> import numpy as np
> import scipy.sparse as sp
> import matplotlib.pyplot as plt
>
> def diffmat2(n, xspan):
>     a, b = float(xspan[0]), float(xspan[1])
>     h = (b - a) / n
>     x = a + h * np.arange(n + 1)
>
>     D2 = np.zeros((n + 1, n + 1))
>     for i in range(1, n):
>         D2[i, i - 1] = 1.0 / h**2
>         D2[i, i] = -2.0 / h**2
>         D2[i, i + 1] = 1.0 / h**2
>     D2[0, 0:4] = np.array([2.0, -5.0, 4.0, -1.0]) / h**2
>     D2[n, n - 3 : n + 1] = np.array([-1.0, 4.0, -5.0, 2.0]) / h**2
>     return x, sp.csr_matrix(D2)
>
> def vec(A):
>     return np.asarray(A).reshape(-1, order="F")
>
> def unvec(z, shape):
>     return np.asarray(z).reshape(shape, order="F")
>
> f = lambda x, y: x**2 - y + 2.0
> m, n = 6, 5
> x, Dxx = diffmat2(m, (0.0, 3.0))
> y, Dyy = diffmat2(n, (-1.0, 1.0))
>
> X, Y = np.meshgrid(x, y, indexing="ij")
> F = f(X, Y)
> N = (m + 1) * (n + 1)
>
> A = sp.kron(sp.eye(n + 1, format="csr"), Dxx, format="csr") + sp.kron(Dyy, sp.eye(m + 1, format="csr"), format="csr")
> b = vec(F)
>
> plt.figure(figsize=(5, 4))
> plt.spy(A, markersize=2)
> plt.title("System matrix before boundary conditions")
> plt.show()
>
> isboundary = np.ones((m + 1, n + 1), dtype=bool)
> isboundary[1:-1, 1:-1] = False
> idx = vec(isboundary)
>
> A = A.tolil()
> I = sp.eye(N, format="lil")
> A[idx, :] = I[idx, :]
> b[idx] = 0.0
> A = A.tocsr()
>
> plt.figure(figsize=(5, 4))
> plt.spy(A, markersize=2)
> plt.title("System matrix with Dirichlet rows")
> plt.show()
>
> u = sp.linalg.spsolve(A, b)
> U = unvec(u, (m + 1, n + 1))
> print(U)
> ```
>
> `spy` 图能直观看到 Kronecker 积与差分矩阵共同产生的稀疏结构；施加 Dirichlet 条件后，边界对应的那些行会被替换成单位阵的行.

**#7 一个通用实现：poissonfd**

> **Function:** poissonfd
> **Solve Poisson's equation on a rectangle by finite differences**
> ```Python
> import numpy as np
> import scipy.sparse as sp
> from scipy.sparse.linalg import spsolve
>
> def diffmat2(n, xspan):
>     a, b = float(xspan[0]), float(xspan[1])
>     h = (b - a) / n
>     x = a + h * np.arange(n + 1)
>
>     D2 = np.zeros((n + 1, n + 1))
>     for i in range(1, n):
>         D2[i, i - 1] = 1.0 / h**2
>         D2[i, i] = -2.0 / h**2
>         D2[i, i + 1] = 1.0 / h**2
>     D2[0, 0:4] = np.array([2.0, -5.0, 4.0, -1.0]) / h**2
>     D2[n, n - 3 : n + 1] = np.array([-1.0, 4.0, -5.0, 2.0]) / h**2
>     return x, sp.csr_matrix(D2)
>
> def vec(A):
>     return np.asarray(A).reshape(-1, order="F")
>
> def unvec(z, shape):
>     return np.asarray(z).reshape(shape, order="F")
>
> def poissonfd(f, g, m, xspan, n, yspan):
>     """Solve Poisson's equation on a rectangle by finite differences.
>
>     Solve u_xx + u_yy = f(x,y) on xspan x yspan with Dirichlet boundary condition u=g.
>
>     Returns:
>       x, y: grid vectors (length m+1 and n+1)
>       U:    solution values on the tensor-product grid, shape (m+1, n+1)
>     """
>     x, Dxx = diffmat2(m, xspan)
>     y, Dyy = diffmat2(n, yspan)
>     X, Y = np.meshgrid(x, y, indexing="ij")
>
>     F = f(X, Y)
>     N = (m + 1) * (n + 1)
>
>     A = sp.kron(sp.eye(n + 1, format="csr"), Dxx, format="csr") + sp.kron(Dyy, sp.eye(m + 1, format="csr"), format="csr")
>     b = vec(F)
>
>     isboundary = np.ones((m + 1, n + 1), dtype=bool)
>     isboundary[1:-1, 1:-1] = False
>     idx = vec(isboundary)
>
>     # Rescale Dirichlet rows to match the magnitude of the PDE rows.
>     i0 = 1 + (m + 1) * 1  # (i,j)=(1,1) in 0-based, i + (m+1)j in vec order
>     scale = float(np.max(np.abs(A.getrow(i0).data)))
>     if scale == 0.0:
>         scale = 1.0
>
>     A = A.tolil()
>     A[idx, :] = 0.0
>     A[idx, idx] = scale
>     b[idx] = scale * g(X.reshape(-1, order="F")[idx], Y.reshape(-1, order="F")[idx])
>     A = A.tocsr()
>
>     u = spsolve(A, b)
>     U = unvec(u, (m + 1, n + 1))
>     return x, y, U
> ```
>
> > **Note:** 对 Dirichlet 行做缩放相当于把边界条件写成 $\sigma u=\sigma g$. 这样做的动机是让边界行与 PDE 行的系数规模接近，从而改善线性系统的数值性态.

**#8 Demo：先选解再构造 forcing**

> **Demo:** A manufactured solution for Poisson's equation
> We choose an exact solution first and derive the forcing term.
>
> ```Python
> import numpy as np
> import matplotlib.pyplot as plt
>
> u_exact = lambda x, y: np.sin(3.0 * x * y - 4.0 * y)
> f = lambda x, y: -np.sin(3.0 * x * y - 4.0 * y) * (9.0 * y**2 + (3.0 * x - 4.0) ** 2)
> g = u_exact
>
> x, y, U = poissonfd(f, g, m=40, xspan=(0.0, 1.0), n=60, yspan=(0.0, 2.0))
>
> X, Y = np.meshgrid(x, y, indexing="ij")
> E = U - u_exact(X, Y)
>
> plt.figure(figsize=(5, 4))
> plt.contourf(x, y, U.T, levels=24, cmap="viridis")
> plt.gca().set_aspect("equal", adjustable="box")
> plt.title("Solution of Poisson's equation")
> plt.xlabel("x")
> plt.ylabel("y")
> plt.colorbar()
> plt.show()
>
> plt.figure(figsize=(5, 4))
> plt.contourf(x, y, E.T, levels=24, cmap="RdBu_r")
> plt.gca().set_aspect("equal", adjustable="box")
> plt.title("Error")
> plt.xlabel("x")
> plt.ylabel("y")
> plt.colorbar()
> plt.show()
> ```
>
> 误差应该是关于 $(x,y)$ 的光滑函数，并且在边界上为 0；否则很可能是边界条件实现出了问题.

**#9 精度与效率**

在上面的实现里，我们在两个方向都使用二阶有限差分. 为了简化讨论，假设 $m=n$. 这时解的误差可以预期为 $O(n^{-2})$.

线性系统矩阵 $\mathbf{A}$ 的大小是 $N=O(n^2)$. 由 Kronecker 积与带状差分矩阵的结构可知，$\mathbf{A}$ 的上下带宽都是 $O(n)$. 可以证明该矩阵对称且负定，因此我们可以对 $-\mathbf{A}$ 使用稀疏 Cholesky 分解，代价为 $O(n^2N)=O(n^4)$ 次运算.

当 $n$ 增大时，截断误差减小，但运算量增加. 运算量的增长快于误差的下降，因此获得高精度会很昂贵. 假设我们能使用的运行时间固定为 $T$，则 $n=O(T^{1/4})$，因此误差随运行时间的收敛可以估计为 $O(T^{-1/2})$. 例如，如果想把误差减少 10 倍，可能需要大约 100 倍的计算工作量.

如果改用 Chebyshev 谱离散，在解足够光滑时，我们预期误差能以 $K^{-n}$ (某个 $K>1$) 的速度收敛. 但此时系统矩阵不再稀疏也不再对称，用 LU 分解求解需要 $O(N^3)=O(n^6)$ 次浮点运算. 因而若以运行时间 $T$ 为自变量，可预期收敛速率大约是 $K^{-T^{1/6}}$.

这两种复杂度并不容易直接比较. 从渐近意义看，谱方法最终会更高效；但如果精度要求并不高，二阶有限差分可能反而更快. 对于矩形区域上的 Poisson 方程，还存在一些专门的快速解法，这里不展开.
