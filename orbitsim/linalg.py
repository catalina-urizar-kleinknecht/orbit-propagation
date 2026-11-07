"""
Dense linear solver: LU decomposition with partial pivoting.

Used by newton.py, which solves one linear system per Newton iteration.
Cholesky is not an option there: the Newton matrix I - h J is not symmetric.
"""

import numpy as np


def lu_factor(A):
    """LU decomposition with partial pivoting: P A = L U.

    Returns (LU, piv): L (unit diagonal, below) and U (on and above the
    diagonal) packed into one matrix, and the row permutation piv.
    """
    LU = np.array(A, dtype=float)
    n = LU.shape[0]
    piv = np.arange(n)
    for k in range(n - 1):
        # move the largest pivot candidate to row k
        p = k + np.argmax(np.abs(LU[k:, k]))
        if p != k:
            LU[[k, p]] = LU[[p, k]]
            piv[[k, p]] = piv[[p, k]]
        if LU[k, k] == 0.0:
            raise np.linalg.LinAlgError("singular matrix")
        # eliminate below the pivot, store the multipliers in place of the zeros
        LU[k + 1:, k] /= LU[k, k]
        LU[k + 1:, k + 1:] -= np.outer(LU[k + 1:, k], LU[k, k + 1:])
    if LU[n - 1, n - 1] == 0.0:
        raise np.linalg.LinAlgError("singular matrix")
    return LU, piv


def lu_solve(LU, piv, b):
    """Solve A x = b given lu_factor(A): forward substitution with L, back substitution with U."""
    n = len(b)
    x = np.array(b, dtype=float)[piv]
    for i in range(n):
        x[i] -= LU[i, :i] @ x[:i]
    for i in reversed(range(n)):
        x[i] = (x[i] - LU[i, i + 1:] @ x[i + 1:]) / LU[i, i]
    return x
