"""
Accuracy and convergence order of every method in orbitsim.integrators.METHODS,
measured against the exact circular-orbit solution:

    uv run experiments.py

Runs stop when the satellite hits the surface (implicit Euler does at moderate
step sizes). A stopped run has no meaningful error at t_end, so it is reported
as such and left out of the order estimate.

accuracy_results() and convergence_results() return the numbers; the
experiment_* functions print them. The regression tests compare the numbers
against tests/regression/golden.json.
"""

import numpy as np

from orbitsim.integrators import METHODS, ODE
from orbitsim.physics import (
    below_surface,
    circular_orbit_state,
    exact_circular_position,
    orbital_period,
    specific_energy,
    two_body_jacobian,
    two_body_rhs,
)
from orbitsim.propagate import propagate

ALTITUDE_KM = 400.0
ACCURACY_STEPS = (60.0, 30.0, 10.0)
CONVERGENCE_STEPS = (240.0, 120.0, 60.0, 30.0, 15.0, 7.5, 3.75, 1.875, 0.9375)


def position_error(traj):
    return np.linalg.norm(traj.ys[-1, :3] - exact_circular_position(traj.t_end, ALTITUDE_KM))


def accuracy_results(n_orbits=10, step_sizes=ACCURACY_STEPS):
    """One row per (method, h): cost, position error after n_orbits, energy change.

    The energy change is signed: (E - E0) / |E0| > 0 means the orbit gained
    energy (spirals outwards), < 0 that it lost energy (spirals inwards).
    pos_error is None for a run that hit the surface.
    """
    y0 = circular_orbit_state(ALTITUDE_KM)
    period = orbital_period(y0)
    e0 = specific_energy(y0)
    rows = []
    for m in METHODS:
        for h in step_sizes:
            ode = ODE(two_body_rhs, two_body_jacobian)
            traj = propagate(m.step, ode, y0, n_orbits * period, h, stop=below_surface)
            rows.append({
                "method": m.name,
                "h": h,
                "n_f": traj.n_f,
                "n_jac": traj.n_jac,
                "stopped": traj.stopped,
                "orbits": traj.t_end / period,
                "pos_error": None if traj.stopped else position_error(traj),
                "energy_change": (specific_energy(traj.ys[-1]) - e0) / abs(e0),
            })
    return rows


def convergence_results(step_sizes=CONVERGENCE_STEPS):
    """Error after one orbit for each method and step size, and the observed orders.

    If the error behaves like C * h^p, halving h divides the error by 2^p,
    so p ~ log2(err(h) / err(h/2)). Expected: the order of each method.
    A run that hits the surface has error nan; orders involving it are nan.
    Returns {method name: {"errors": [...], "orders": [...]}}.
    """
    y0 = circular_orbit_state(ALTITUDE_KM)
    period = orbital_period(y0)
    results = {}
    for m in METHODS:
        errors = []
        for h in step_sizes:
            # use a step that divides the period exactly
            n = int(round(period / h))
            traj = propagate(m.step, ODE(two_body_rhs, two_body_jacobian), y0, period, period / n,
                             stop=below_surface)
            errors.append(np.nan if traj.stopped else position_error(traj))
        orders = [np.log2(errors[i] / errors[i + 1]) for i in range(len(errors) - 1)]
        results[m.name] = {"errors": errors, "orders": orders}
    return results


def experiment_accuracy(n_orbits=10):
    """Experiment 1: position error and energy change after several orbits."""
    period = orbital_period(circular_orbit_state(ALTITUDE_KM))
    print(f"\nExperiment 1: {n_orbits} orbits at {ALTITUDE_KM:.0f} km "
          f"(period {period / 60:.1f} min)")
    print(f"{'method':<15} {'h [s]':>6} {'f-evals':>8} {'J-evals':>8} "
          f"{'pos. error [km]':>16} {'rel. energy change':>19}")
    for r in accuracy_results(n_orbits):
        err = f"surface @ {r['orbits']:.2f} orb" if r["stopped"] else f"{r['pos_error']:.3e}"
        print(f"{r['method']:<15} {r['h']:6.0f} {r['n_f']:8d} {r['n_jac']:8d} {err:>16} "
              f"{r['energy_change']:19.3e}")


def experiment_convergence():
    """Experiment 2: observed order of convergence after one orbit."""
    print("\nExperiment 2: observed order of convergence (one orbit), "
          "h = " + ", ".join(f"{h:g}" for h in CONVERGENCE_STEPS) + " s")
    orders_expected = {m.name: m.order for m in METHODS}
    for name, r in convergence_results().items():
        print(f"{name:<15} expected order {orders_expected[name]}")
        print(f"{'':<4} errors [km]:    " + ", ".join("surface" if np.isnan(e) else f"{e:.2e}" for e in r["errors"]))
        print(f"{'':<4} observed order: " + ", ".join("-" if np.isnan(p) else f"{p:.2f}" for p in r["orders"]))


if __name__ == "__main__":
    experiment_accuracy()
    experiment_convergence()
