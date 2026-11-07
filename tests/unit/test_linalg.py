import numpy as np
import pytest

from orbitsim.linalg import lu_factor, lu_solve


def test_lu_matches_numpy():
    rng = np.random.default_rng(0)
    for _ in range(20):
        A = rng.normal(size=(6, 6))
        b = rng.normal(size=6)
        LU, piv = lu_factor(A)
        assert np.allclose(lu_solve(LU, piv, b), np.linalg.solve(A, b))


def test_lu_reconstructs_matrix():
    # P A = L U, with L unit lower triangular and U upper triangular packed in LU
    rng = np.random.default_rng(1)
    A = rng.normal(size=(5, 5))
    LU, piv = lu_factor(A)
    L = np.tril(LU, -1) + np.eye(5)
    U = np.triu(LU)
    assert np.allclose(A[piv], L @ U)


def test_lu_needs_pivoting():
    # zero on the diagonal: fails without row exchanges
    A = np.array([[0.0, 1.0], [1.0, 1.0]])
    LU, piv = lu_factor(A)
    assert np.allclose(lu_solve(LU, piv, [1.0, 2.0]), [1.0, 1.0])


def test_lu_rejects_singular_matrix():
    with pytest.raises(np.linalg.LinAlgError):
        lu_factor(np.array([[1.0, 2.0], [2.0, 4.0]]))


def test_lu_does_not_modify_input():
    A = np.array([[0.0, 1.0], [1.0, 1.0]])
    A_copy = A.copy()
    lu_factor(A)
    assert np.array_equal(A, A_copy)
