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


if __name__ == "__main__":
    x = [-1, 0, 2, 3, 4]
    print(vandermode_matrix(x))
