import numpy as np
import pytest

from orbitsim.newton import NewtonError, newton_solve


def test_newton_solves_nonlinear_system():
    # x^2 + y^2 = 4, x = y  ->  x = y = sqrt(2)
    G = lambda z: np.array([z[0] ** 2 + z[1] ** 2 - 4, z[0] - z[1]])
    dG = lambda z: np.array([[2 * z[0], 2 * z[1]], [1.0, -1.0]])
    assert np.allclose(newton_solve(G, dG, [1.0, 0.5]), [np.sqrt(2), np.sqrt(2)])


def test_newton_solves_linear_system_in_one_step():
    A = np.array([[3.0, 1.0], [1.0, 2.0]])
    b = np.array([9.0, 8.0])
    calls = []

    def dG(z):
        calls.append(1)
        return A

    z = newton_solve(lambda z: A @ z - b, dG, [0.0, 0.0])
    assert np.allclose(z, np.linalg.solve(A, b))
    assert len(calls) <= 2  # one step to the solution, one to confirm dz ~ 0


def test_newton_reports_no_root():
    # x^2 + 1 = 0 has no real root; the iteration hits x = 0, where dG is singular
    with pytest.raises(NewtonError):
        newton_solve(lambda z: z**2 + 1, lambda z: np.diag(2 * z), [1.0], max_iter=10)


def test_newton_reports_too_many_iterations():
    # converges, but not within 2 iterations
    with pytest.raises(NewtonError):
        newton_solve(lambda z: z**3 - 8, lambda z: np.diag(3 * z**2), [10.0], max_iter=2)
