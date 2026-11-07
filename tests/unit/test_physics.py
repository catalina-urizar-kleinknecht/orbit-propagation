import numpy as np

from orbitsim.physics import (
    MU_EARTH,
    R_EARTH,
    below_surface,
    circular_orbit_state,
    exact_circular_position,
    orbital_period,
    specific_energy,
    two_body_jacobian,
    two_body_rhs,
)


def test_rhs_points_to_the_centre():
    y = circular_orbit_state(400.0)
    dy = two_body_rhs(0.0, y)
    assert np.array_equal(dy[:3], y[3:])  # dr/dt = v
    r = y[:3]
    assert np.allclose(dy[3:], -MU_EARTH / np.linalg.norm(r) ** 3 * r)


def test_jacobian_matches_finite_differences():
    y = circular_orbit_state(400.0) + np.array([10, -20, 30, 0.1, 0.2, -0.3])
    eps = 1e-6
    J_fd = np.column_stack([
        (two_body_rhs(0, y + eps * e) - two_body_rhs(0, y - eps * e)) / (2 * eps) for e in np.eye(6)
    ])
    assert np.allclose(two_body_jacobian(0, y), J_fd, rtol=1e-6, atol=1e-12)


def test_circular_orbit_energy():
    # E = -mu / (2a) with a = r for a circular orbit
    y = circular_orbit_state(400.0)
    assert np.isclose(specific_energy(y), -MU_EARTH / (2 * (R_EARTH + 400.0)))


def test_exact_position_closes_after_one_period():
    y0 = circular_orbit_state(400.0)
    T = orbital_period(y0)
    assert np.allclose(exact_circular_position(0.0, 400.0), y0[:3])
    assert np.allclose(exact_circular_position(T, 400.0), y0[:3], atol=1e-8)
    # 92.6 min at 400 km (README section 1.5)
    assert abs(T / 60 - 92.6) < 0.05


def test_below_surface():
    assert not below_surface(circular_orbit_state(1.0))
    assert below_surface(circular_orbit_state(-1.0))
