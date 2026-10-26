"""
Two-body orbit propagation: comparing Euler, RK2 and RK4.

Three fixed-step integrators written from scratch, and two experiments that
measure their error against the exact circular-orbit solution:

    uv run orbit_rk.py

Units: km, s. State vector y = [x, y, z, vx, vy, vz].
"""

import numpy as np

MU_EARTH = 398600.4418  # Earth's gravitational parameter, km^3/s^2
R_EARTH = 6378.137      # Earth's equatorial radius, km


# ---------------------------------------------------------------------------
# Physics
# ---------------------------------------------------------------------------

def two_body_rhs(t, y, mu=MU_EARTH):
    """Right-hand side of the two-body ODE: dy/dt = f(t, y).

    dr/dt = v
    dv/dt = -mu * r / |r|^3
    """
    r = y[:3]
    v = y[3:]
    a = -mu * r / np.linalg.norm(r) ** 3
    return np.concatenate([v, a])


def specific_energy(y, mu=MU_EARTH):
    """Specific orbital energy E = v^2/2 - mu/r. Constant for the exact solution."""
    r = np.linalg.norm(y[:3])
    v = np.linalg.norm(y[3:])
    return 0.5 * v**2 - mu / r


def circular_orbit_state(altitude_km):
    """Initial state for a circular, equatorial orbit at the given altitude."""
    r0 = R_EARTH + altitude_km
    v0 = np.sqrt(MU_EARTH / r0)
    return np.array([r0, 0.0, 0.0, 0.0, v0, 0.0])


def exact_circular_position(t, altitude_km):
    """Exact position on the circular orbit at time t (for error measurement)."""
    r0 = R_EARTH + altitude_km
    omega = np.sqrt(MU_EARTH / r0**3)
    return np.array([r0 * np.cos(omega * t), r0 * np.sin(omega * t), 0.0])


# ---------------------------------------------------------------------------
# Integrators
# Each function takes f(t, y), the current time t, state y and step size h,
# and returns the new state y_{n+1}.
# ---------------------------------------------------------------------------

def euler_step(f, t, y, h):
    """Explicit Euler (order 1): one evaluation of f per step."""
    # follow the tangent at the start of the step
    y_new = y + h * f(t, y)
    return y_new


def rk2_step(f, t, y, h):
    """Midpoint method (order 2): two evaluations of f per step."""
    # use the slope at the midpoint, reached with half an Euler step
    k1 = f(t, y)
    k2 = f(t + h / 2, y + h / 2 * k1)
    y_new = y + h * k2
    return y_new


def rk4_step(f, t, y, h):
    """Classical Runge-Kutta (order 4): four evaluations of f per step."""
    # weighted average of slopes: start, twice at the midpoint, end (1:2:2:1)
    k1 = f(t, y)
    k2 = f(t + h / 2, y + h / 2 * k1)
    k3 = f(t + h / 2, y + h / 2 * k2)
    k4 = f(t + h, y + h * k3)
    y_new = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return y_new


STEPPERS = {"Euler": (euler_step, 1), "RK2": (rk2_step, 2), "RK4": (rk4_step, 4)}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def propagate(stepper, f, y0, t_end, h):
    """Integrate from t=0 to t_end with fixed step h. Returns (times, states)."""
    n_steps = int(round(t_end / h))
    ts = np.linspace(0.0, n_steps * h, n_steps + 1)
    ys = np.empty((n_steps + 1, len(y0)))
    ys[0] = y0
    for i in range(n_steps):
        ys[i + 1] = stepper(f, ts[i], ys[i], h)
    return ts, ys


def experiment_accuracy(altitude_km=400.0, n_orbits=10, step_sizes=(60.0, 30.0, 10.0)):
    """Experiment 1: position error and energy drift after several orbits."""
    y0 = circular_orbit_state(altitude_km)
    period = 2 * np.pi * np.sqrt(np.linalg.norm(y0[:3]) ** 3 / MU_EARTH)
    t_end = n_orbits * period
    e0 = specific_energy(y0)

    print(f"\nExperiment 1: {n_orbits} orbits at {altitude_km:.0f} km "
          f"(period {period / 60:.1f} min)")
    print(f"{'method':<6} {'h [s]':>7} {'f-evals':>9} {'pos. error [km]':>16} {'rel. energy drift':>18}")
    for name, (stepper, evals_per_step) in STEPPERS.items():
        for h in step_sizes:
            ts, ys = propagate(stepper, two_body_rhs, y0, t_end, h)
            pos_err = np.linalg.norm(ys[-1, :3] - exact_circular_position(ts[-1], altitude_km))
            drift = abs((specific_energy(ys[-1]) - e0) / e0)
            print(f"{name:<6} {h:7.0f} {evals_per_step * (len(ts) - 1):9d} {pos_err:16.3e} {drift:18.3e}")


def experiment_convergence(altitude_km=400.0, step_sizes=(240.0, 120.0, 60.0, 30.0, 15.0)):
    """Experiment 2: observed order of convergence after one orbit.

    If the error behaves like C * h^p, halving h divides the error by 2^p,
    so p ~ log2(err(h) / err(h/2)). Expect p ~ 1, 2, 4.
    """
    y0 = circular_orbit_state(altitude_km)
    period = 2 * np.pi * np.sqrt(np.linalg.norm(y0[:3]) ** 3 / MU_EARTH)

    print("\nExperiment 2: observed order of convergence (one orbit)")
    for name, (stepper, _) in STEPPERS.items():
        errors = []
        for h in step_sizes:
            # use a step that divides the period exactly
            n = int(round(period / h))
            ts, ys = propagate(stepper, two_body_rhs, y0, n * (period / n), period / n)
            errors.append(np.linalg.norm(ys[-1, :3] - exact_circular_position(ts[-1], altitude_km)))
        orders = [np.log2(errors[i] / errors[i + 1]) for i in range(len(errors) - 1)]
        print(f"{name:<6} errors: " + ", ".join(f"{e:.2e}" for e in errors))
        print(f"{'':<6} observed order: " + ", ".join(f"{p:.2f}" for p in orders))


if __name__ == "__main__":
    experiment_accuracy()
    experiment_convergence()
