"""
Newton's method for a nonlinear system G(z) = 0.

Each iteration linearises G around the current guess and solves

    dG(z) dz = -G(z),   z <- z + dz

with the LU solver from linalg.py. Implicit integrators call this once per step.
"""

import numpy as np

from .linalg import lu_factor, lu_solve


class NewtonError(RuntimeError):
    pass


def newton_solve(G, dG, z0, tol=1e-12, max_iter=20):
    """Solve G(z) = 0 starting from z0. dG(z) is the Jacobian of G.

    Stops when the update is small relative to z: |dz| <= tol * (1 + |z|).
    """
    z = np.array(z0, dtype=float)
    for _ in range(max_iter):
        try:
            LU, piv = lu_factor(dG(z))
        except np.linalg.LinAlgError as e:
            raise NewtonError("singular Jacobian in Newton iteration") from e
        dz = lu_solve(LU, piv, -G(z))
        if not np.all(np.isfinite(dz)):
            # nearly singular Jacobian; without this check inf <= tol * inf would count as converged
            raise NewtonError("Newton update is not finite")
        z = z + dz
        if np.linalg.norm(dz) <= tol * (1.0 + np.linalg.norm(z)):
            return z
    raise NewtonError(f"Newton did not converge in {max_iter} iterations")
