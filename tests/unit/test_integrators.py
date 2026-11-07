import numpy as np
import pytest

from orbitsim.integrators import METHODS, ODE, euler_step, implicit_euler_step
from orbitsim.physics import (
    circular_orbit_state,
    exact_circular_position,
    orbital_period,
    two_body_jacobian,
    two_body_rhs,
)
from orbitsim.propagate import propagate


def test_method_names_are_unique():
    names = [m.name for m in METHODS]
    assert len(names) == len(set(names))


@pytest.mark.parametrize("method", METHODS, ids=lambda m: m.name)
def test_observed_order(method):
    """Halving h divides the error after a short arc by about 2^order."""
    y0 = circular_orbit_state(400.0)
    t_end = orbital_period(y0) / 10  # short enough for every method to stay in orbit
    errors = []
    for h in (t_end / 40, t_end / 80):
        traj = propagate(method.step, ODE(two_body_rhs, two_body_jacobian), y0, t_end, h)
        errors.append(np.linalg.norm(traj.ys[-1, :3] - exact_circular_position(traj.t_end, 400.0)))
    assert np.log2(errors[0] / errors[1]) == pytest.approx(method.order, abs=0.15)


def test_euler_on_linear_decay():
    # y' = -y: one Euler step from 1 gives 1 - h, one implicit Euler step gives 1 / (1 + h)
    ode = ODE(lambda t, y: -y, lambda t, y: -np.eye(1))
    h = 0.1
    assert np.allclose(euler_step(ode, 0.0, np.array([1.0]), h), [1 - h])
    assert np.allclose(implicit_euler_step(ode, 0.0, np.array([1.0]), h), [1 / (1 + h)])


def test_implicit_euler_satisfies_its_equation():
    # y_{n+1} = y_n + h f(t_{n+1}, y_{n+1}) must hold to Newton's tolerance
    ode = ODE(two_body_rhs, two_body_jacobian)
    y = circular_orbit_state(400.0)
    h = 30.0
    z = implicit_euler_step(ode, 0.0, y, h)
    assert np.allclose(z, y + h * two_body_rhs(h, z), rtol=1e-12, atol=1e-9)


def test_implicit_euler_needs_a_jacobian():
    ode = ODE(two_body_rhs)
    with pytest.raises(ValueError):
        implicit_euler_step(ode, 0.0, circular_orbit_state(400.0), 10.0)


def test_ode_counts_evaluations():
    ode = ODE(two_body_rhs, two_body_jacobian)
    y = circular_orbit_state(400.0)
    rk4 = next(m for m in METHODS if m.name == "RK4")
    rk4.step(ode, 0.0, y, 10.0)
    assert (ode.n_f, ode.n_jac) == (4, 0)
