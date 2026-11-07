"""
Two-body physics: the ODE, its Jacobian, conserved quantities and the exact
circular-orbit solution used as the reference.

Units: km, s. State vector y = [x, y, z, vx, vy, vz].
"""

import numpy as np

MU_EARTH = 398600.4418  # Earth's gravitational parameter, km^3/s^2
R_EARTH = 6378.137      # Earth's equatorial radius, km


def two_body_rhs(t, y, mu=MU_EARTH):
    """Right-hand side of the two-body ODE: dy/dt = f(t, y).

    dr/dt = v
    dv/dt = -mu * r / |r|^3
    """
    r = y[:3]
    v = y[3:]
    a = -mu * r / np.linalg.norm(r) ** 3
    return np.concatenate([v, a])


def two_body_jacobian(t, y, mu=MU_EARTH):
    """Jacobian J = df/dy of the two-body RHS, shape (6, 6). Needed by implicit methods.

    J = [[0, I],
         [A, 0]]   with   A = da/dr = -mu/|r|^3 * (I - 3 r r^T / |r|^2)
    """
    r = y[:3]
    rn = np.linalg.norm(r)
    A = -mu / rn**3 * (np.eye(3) - 3 * np.outer(r, r) / rn**2)
    J = np.zeros((6, 6))
    J[:3, 3:] = np.eye(3)
    J[3:, :3] = A
    return J


def specific_energy(y, mu=MU_EARTH):
    """Specific orbital energy E = v^2/2 - mu/r. Constant for the exact solution."""
    r = np.linalg.norm(y[:3])
    v = np.linalg.norm(y[3:])
    return 0.5 * v**2 - mu / r


def below_surface(y):
    """True once the satellite is inside the Earth: used to stop a propagation."""
    return np.linalg.norm(y[:3]) < R_EARTH


def circular_orbit_state(altitude_km):
    """Initial state for a circular, equatorial orbit at the given altitude."""
    r0 = R_EARTH + altitude_km
    v0 = np.sqrt(MU_EARTH / r0)
    return np.array([r0, 0.0, 0.0, 0.0, v0, 0.0])


def orbital_period(y0):
    """Period of the circular orbit through y0."""
    return 2 * np.pi * np.sqrt(np.linalg.norm(y0[:3]) ** 3 / MU_EARTH)


def exact_circular_position(t, altitude_km):
    """Exact position on the circular orbit at time t (for error measurement)."""
    r0 = R_EARTH + altitude_km
    omega = np.sqrt(MU_EARTH / r0**3)
    return np.array([r0 * np.cos(omega * t), r0 * np.sin(omega * t), 0.0])
