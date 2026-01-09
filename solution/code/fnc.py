from functools import lru_cache
import sys
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


def plufact(A):
    """2-6
    Compute the PLU factorization of square matrix `A`, returning the
    triangular factors and a row permutation vector.
    """
    A = A.astype(float)
    n = A.shape[0]
    L = np.zeros((n, n))
    U = np.zeros((n, n))
    p = np.zeros(n).astype(int)
    A_k = A.copy()

    # Reduction by outer products
    for k in range(n - 1):
        p[k] = np.argmax(np.abs(A_k[:, k]))
        U[k, :] = A_k[p[k], :]
        L[:, k] = A_k[:, k] / U[k, k]
        A_k -= np.outer(L[:, k], U[k, :])

    p[-1] = np.argmax(np.abs(A_k[:, -1]))
    U[-1, -1] = A_k[p[-1], -1]
    L[:, -1] = A_k[:, -1] / U[-1, -1]
    return L[p, :], U, p


def lsqrfact(A, b):
    """3-3
    Solve a linear least-squares problem by QR factorization.
    Returns the minimizer of ||b-Ax||.
    """
    Q, R = np.linalg.qr(A)
    c = Q.T @ b
    x = backsub(R, c)
    return x




def newton(f, dfdx, x_0, 
           maxiter=40, 
           ftol=100*sys.float_info.epsilon, 
           xtol=100*sys.float_info.epsilon):
    """4-3
    newton(f, dfdx, x_0 [maxiter, ftol, xtol])

    Use Newton's method to find a root of `f` starting from `x_0`, where
    `dfdx` is the derivative of `f`. Returns a vector of root estimates.

    The optional keyword parameters set the maximum number of iterations
    and the stopping tolerance for values of `f` and changes in `x`.
    """
    x = [float(x_0)]
    y = f(x_0)
    delta_x = float('inf')  # for initial pass bellow
    k = 0

    while abs(delta_x) > xtol and abs(y) > ftol:
        dydx = dfdx(x[k])
        delta_x = -y / dydx  # Newton's step
        x.append(x[k] + delta_x)  # append new estimate

        k += 1
        y = f(x[k])

        if k == maxiter:
            print("Warning: Maximum number of iterations reached.")
            break

    return x


def secant(f, x_1, x_2, 
           maxiter=40, 
           ftol=100*sys.float_info.epsilon, 
           xtol=100*sys.float_info.epsilon):
    """4-4
    secant(f, x_1, x_2 [maxiter, ftol, xtol])

    Use the secant method to find a root of `f` starting from `x_1` and `x_2`.
    Returns a list of root estimates.

    The optional keyword parameters set the maximum number of iterations
    and the stopping tolerance for values of `f` and changes in `x`.
    """
    x = [float(x_1), float(x_2)]
    y_1 = f(x_1)
    delta_x = float('inf')  # for initial pass in the loop below
    y_2 = float('inf')
    k = 1

    while abs(delta_x) > xtol and abs(y_2) > ftol:
        y_2 = f(x[k])
        delta_x = -y_2 * (x[k] - x[k - 1]) / (y_2 - y_1)  # secant step
        x.append(x[k] + delta_x)  # append new estimate

        k += 1
        y_1 = y_2  # current f-value becomes the old one next time

        if k == maxiter:
            print("Warning: Maximum number of iterations reached.")
            break
    return x
    

def iqi(f, x_1, x_2, x_3, 
        maxiter=40, 
        ftol=100*sys.float_info.epsilon, 
        xtol=100*sys.float_info.epsilon):
    """4-4
    """
    # inverse quadratic interpolation
    x = [float(x_1), float(x_2), float(x_3)]
    y_1, y_2 = f(x_1), f(x_2)
    delta_x = float('inf')
    y_3 = float('inf')
    k = 1
    
    while abs(delta_x) > xtol and abs(y_3) > ftol:
        y_3 = f(x_3)
        y_values = np.array([y_1, y_2, y_3])
        x_values = np.array([x_1, x_2, x_3])
        coeffs = np.polyfit(y_values, x_values, 2)
        x_new = np.polyval(coeffs, 0)
        x.append(x_new)

        delta_x = x_new - x_3
        k += 1
        x_1, x_2, x_3 = x_2, x_3, x_new
        y_1, y_2 = y_2, y_3

        if k == maxiter:
            print("Warning: Maximum number of iterations reached.")
            break
    return x


import numpy as np

def newtonsys(f, jac, x_0, 
              maxiter=40, 
              ftol=1000*np.finfo(float).eps, 
              xtol=1000*np.finfo(float).eps):
    """4-5
    newtonsys(f, jac, x_0 [maxiter,ftol,xtol])

    Use Newton's method to find a root of a system of equations,
    starting from `x_0`. The functions `f` and `jac` should return the
    residual vector and the Jacobian matrix, respectively. Returns the
    history of root estimates as a vector of vectors.

    The optional keyword parameters set the maximum number of iterations
    and the stopping tolerance for values of `f` and changes in `x`.
    """
    x = [np.array(x_0, dtype=float)]
    y, J = f(x_0), jac(x_0)
    delta_x = np.inf  # for initial pass below
    k = 0

    while (np.linalg.norm(delta_x) > xtol) and (np.linalg.norm(y) > ftol):
        delta_x = -np.linalg.solve(J, y)  # Newton step
        x.append(x[k] + delta_x)  # append to history
        k += 1
        y, J = f(x[k]), jac(x[k])

        if k == maxiter:
            print("Warning: Maximum number of iterations reached.")
            break
    return x


def fdjac(f, x_0, y_0=None):
    """4-6
    fdjac(f, x_0 [y_0])

    Compute a finite-difference approximation of the Jacobian matrix for
    `f` at `x_0`, where `y_0`=`f(x_0)` may be given.
    """
    if y_0 is None:
        y_0 = f(x_0)
    # FD step size
    delta = np.sqrt(np.finfo(float).eps) * max(np.linalg.norm(x_0), 1)
    m, n = len(y_0), len(x_0)
    if n == 1:
        J = (f(x_0 + delta) - y_0) / delta
    else:
        J = np.zeros((m, n))
        x = x_0.copy()
        for j in range(n):
            x[j] += delta
            J[:, j] = (f(x) - y_0) / delta
            x[j] -= delta
    return J


def levenberg(f, x_0, maxiter=40, ftol=1e-12, xtol=1e-12):
    """4-6
    levenberg(f, x_0 [maxiter, ftol, xtol])

    Use Levenberg's quasi-Newton iteration to find a root of the system
    `f` starting from `x_0`. Returns the history of root estimates 
    as a vector of vectors.

    The optional keyword parameters set the maximum number of iterations
    and the stopping tolerance for values of `f` and changes in `x`.

    """
    x = [np.array(x_0, dtype=float)]
    y_k = f(x_0)
    k = 0
    s = np.inf
    A = fdjac(f, x[k], y_k)  # start with FD Jacobian
    jac_is_new = True

    lam = 10
    while (np.linalg.norm(s) > xtol) and (np.linalg.norm(y_k) > ftol):
        # Compute the proposed step.
        B = A.T @ A + lam * np.eye(len(A[0]))
        z = A.T @ y_k
        s = - np.linalg.solve(B, z)

        x_hat = x[k] + s
        y_hat = f(x_hat)

        # Do we accept the result?
        if np.linalg.norm(y_hat) < np.linalg.norm(y_k):  # accept
            lam = lam / 10  # get closer to Newton
            # Broyden update of the Jacobian.
            s = s.reshape(-1, 1)
            A += (y_hat.reshape(-1, 1) - y_k.reshape(-1, 1) - A @ s) @ (s.T / (s.T @ s))
            jac_is_new = False

            x.append(x_hat)
            y_k = y_hat
            k += 1
        else:  # don't accept
            # Get closer to gradient descent.
            lam = 4 * lam
            # Re-initialize the Jacobian if it's out of date.
            if not jac_is_new:
                A = fdjac(f, x[k], y_k)
                jac_is_new = True

        if k == maxiter:
            print("Warning: Maximum number of iterations reached.")
            break
    return x


def hatfun(t, k):
    """5-2
    Create a piecewise linear hat function, where `t` is a
    vector of n+1 interpolation nodes and `k` is an integer in 0:n
    giving the index of the node where the hat function equals one.
    """
    n = len(t) - 1
    def H(x):
        if k > 0 and t[k-1] <= x <= t[k]:
            return (x - t[k-1]) / (t[k] - t[k-1])
        elif k < n and t[k] <= x <= t[k+1]:
            return (t[k+1] - x) / (t[k+1] - t[k])
        else:
            return 0.0
    return np.vectorize(H)


def plinterp(t, y):
    """5-2
    Construct a piecewise linear interpolating function for data values in
    `y` given at nodes in `t`.
    """
    hats = [hatfun(t, k) for k in range(len(t))]
    def p(x):
        return sum(y[k] * hats[k](x) for k in range(len(t)))
    return np.vectorize(p)


def spinterp(t, y):
    """5-3
    Construct a cubic not-a-knot spline interpolating function for data
    values in `y` given at nodes in `t`.

    Parameters
    ----------
    t : 1-D array_like (length n+1, strictly increasing)
        Node vector.
    y : 1-D array_like (length n+1)
        Data values f(t_k).

    Returns
    -------
    S : callable
        S(x) evaluates the spline at x (scalar or NumPy array).

    Notes
    -----
    • Requires n >= 3 for non-degenerate not-a-knot conditions.  
    • The coefficient vector is ordered [a | b | c | d], each length n.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    n = t.size - 1
    if n < 3:
        raise ValueError("Need at least 4 nodes (n >= 3 segments) for not-a-knot spline")

    h  = np.diff(t)                  # segment lengths (size n)
    Z  = np.zeros((n, n))
    I  = np.eye(n)
    E  = I[:-1, :]                   # deletes last row
    J  = I - np.diag(np.ones(n-1), k=1)   # diag(1) + super-diag(-1)
    H  = np.diag(h)
    H2 = np.diag(h**2)
    H3 = np.diag(h**3)

    # Assemble block rows (same order as original Julia code).
    AL  = np.hstack([I,  Z,    Z,    Z   ])
    AR  = np.hstack([I,  H,    H2,   H3  ])
    A1  = E @ np.hstack([Z,  J,  2*H, 3*H2])
    A2  = E @ np.hstack([Z,  Z,   J, 3*H ])
    nakL = np.zeros((1, 4*n))
    nakL[0, 3*n   ] =  1
    nakL[0, 3*n+1 ] = -1
    nakR = np.zeros((1, 4*n))
    nakR[0, 4*n-2 ] =  1
    nakR[0, 4*n-1 ] = -1

    A = np.vstack([AL, AR, A1, A2, nakL, nakR])
    v = np.concatenate([y[:-1], y[1:],            # vL, vR
                        np.zeros(n-1), np.zeros(n-1),  # v1, v2
                        [0.0, 0.0]])              # two not-a-knot RHS zeros

    # Solve for coefficients.
    z = np.linalg.solve(A, v)
    a, b, c, d = np.split(z, 4)

    # Evaluation routine (vectorised).
    def S(x):
        x = np.asarray(x, dtype=float)
        out = np.full_like(x, np.nan, dtype=float)

        m = (x >= t[0]) & (x <= t[-1])
        if not np.any(m):
            return out if out.shape else np.nan

        xi  = x[m]
        idx = np.searchsorted(t, xi, side='right') - 1   # segment indices
        idx[idx == n] = n-1                              # hit the last node
        s   = xi - t[idx]                                # local coordinate

        out[m] = a[idx] + s*(b[idx] + s*(c[idx] + s*d[idx]))
        return out if out.shape else out.item()

    return S


def fdweights(t, m):
    """5-4
    Compute weights for the `m`th derivative of a function at zero using
    values at the nodes in vector `t`.
    """
    t = np.asarray(t, dtype=float)
    r = len(t) - 1          # highest node index

    @lru_cache(maxsize=None)
    def weight(m_, r_, k_):
        # Recursively compute a single weight.
        if m_ < 0 or m_ > r_:
            return 0.0
        if m_ == 0 and r_ == 0:
            return 1.0
        if k_ < r_:
            # “interior” recursion branch
            return (t[r_] * weight(m_, r_-1, k_) -
                    m_    * weight(m_-1, r_-1, k_)) / (t[r_] - t[k_])
        else:
            # k_ == r_  →  “last-node” recursion branch
            numer = np.prod(t[r_-1] - t[:r_-1]) if r_ > 1 else 1.0
            denom = np.prod(t[r_]   - t[:r_])   if r_ > 0 else 1.0
            beta  = numer / denom
            return beta * (m_ * weight(m_-1, r_-1, r_-1)
                           - t[r_-1] * weight(m_, r_-1, r_-1))

    return np.array([weight(m, r, k) for k in range(r + 1)], dtype=float)


if __name__ == "__main__":
    A = np.array([
        [2, -1], 
        [0, 1], 
        [-2, 2]
    ])
    b = np.array([1, -5, 6])

    print(lsqrfact(A, b))


