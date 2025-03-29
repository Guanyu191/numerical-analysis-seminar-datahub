import numpy as np


def polynomial(c, x):
    """1-3
    """
    result = 0
    for i, coefficient in enumerate(c):
        result += coefficient * (x) ** i
    return result


def vandermode_matrix(x):
    """2-1
    n = len(x)
    V = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            V[i, j] = x[i] ** j
    return V
    """
    V = np.array(
        [[x[i] ** j for j in range(len(x))] for i in range(len(x))]
    )
    return V


def forwardsub(L, b):
    """2-3
    Solve the lower triangular linear system with matrix `L` and
    right-hand side vector `b`.
    """
    n = L.shape[0]
    x = np.zeros(n)
    x[0] = b[0] / L[0, 0]
    for i in range(1, n):
        s = np.sum(L[i, :i] * x[:i])
        x[i] = (b[i] - s) / L[i, i]
    return x


def backsub(U, b):
    """2-3
    Solve the upper triangular linear system with matrix `U` and
    right-hand side vector `b`.
    """
    n = U.shape[0]
    x = np.zeros(n)
    x[-1] = b[-1] / U[-1, -1]
    for i in range(n - 2, -1, -1):
        s = np.sum(U[i, i + 1:] * x[i + 1:])
        x[i] = (b[i] - s) / U[i, i]
    return x


def lufact(A):
    """2-4
    Compute the LU factorization of square matrix `A`, 
    returning the factors.
    """
    n = A.shape[0]
    L = np.eye(n)  # ones on main diagonal, zeros elsewhere
    U = np.zeros((n, n))
    A_k = A.copy()

    # reduction by outer product
    for k in range(n - 1):
        U[k, :] = A_k[k, :]
        L[:, k] = A_k[:, k] / U[k, k]
        A_k = A_k - np.outer(L[:, k], U[k, :])

    U[-1, -1] = A_k[-1, -1]
    return L, U


if __name__ == "__main__":
    A = np.array([
        [2, 3, 4], 
        [4, 5, 10], 
        [4, 8, 2]
    ])
    L, U = lufact(A)
    print("L = \n", L)
    print("U = \n", U)


