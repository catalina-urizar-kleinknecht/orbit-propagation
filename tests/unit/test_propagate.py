import numpy as np

from orbitsim.integrators import ODE, euler_step, implicit_euler_step
from orbitsim.physics import (
    below_surface,
    circular_orbit_state,
    orbital_period,
    two_body_jacobian,
    two_body_rhs,
)
from orbitsim.propagate import propagate


def test_propagate_grid_and_counts():
    ode = ODE(lambda t, y: -y)
    traj = propagate(euler_step, ode, np.array([1.0]), 1.0, 0.1)
    assert len(traj.ts) == len(traj.ys) == 11
    assert np.allclose(traj.ts, np.linspace(0, 1, 11))
    assert np.allclose(traj.ys[:, 0], 0.9 ** np.arange(11))
    assert (traj.n_f, traj.n_jac, traj.stopped) == (10, 0, False)


def test_propagate_counts_only_its_own_run():
    ode = ODE(lambda t, y: -y)
    propagate(euler_step, ode, np.array([1.0]), 1.0, 0.1)
    traj = propagate(euler_step, ode, np.array([1.0]), 0.5, 0.1)
    assert traj.n_f == 5


def test_propagate_stops_at_surface():
    y0 = circular_orbit_state(400.0)
    traj = propagate(implicit_euler_step, ODE(two_body_rhs, two_body_jacobian), y0,
                     orbital_period(y0), 60.0, stop=below_surface)
    assert traj.stopped
    assert below_surface(traj.ys[-1]) and not below_surface(traj.ys[-2])
    assert len(traj.ts) == len(traj.ys)
