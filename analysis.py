"""
Plots and a SciPy comparison for the two-body integrators in orbit_rk.py.

    uv add matplotlib scipy
    uv run analysis.py

Saves three figures to ./figures/ and prints a comparison table.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

from orbit_rk import (
    MU_EARTH,
    R_EARTH,
    STEPPERS,
    circular_orbit_state,
    exact_circular_position,
    propagate,
    two_body_rhs,
)

ALTITUDE_KM = 400.0
FIG_DIR = Path("figures")


def orbital_period(y0):
    return 2 * np.pi * np.sqrt(np.linalg.norm(y0[:3]) ** 3 / MU_EARTH)


def counted(f):
    """Wrap f so we can count how often it is evaluated."""
    def wrapper(t, y):
        wrapper.calls += 1
        return f(t, y)
    wrapper.calls = 0
    return wrapper


# ---------------------------------------------------------------------------
# 1a. Orbits in the x-y plane
# ---------------------------------------------------------------------------

def plot_orbits(h=60.0, n_orbits=3):
    """Same step size for all methods: shows how Euler spirals outwards."""
    y0 = circular_orbit_state(ALTITUDE_KM)
    t_end = n_orbits * orbital_period(y0)

    fig, ax = plt.subplots(figsize=(7, 7))
    earth = plt.Circle((0, 0), R_EARTH, color="#cfd8e3", zorder=0, label="Earth")
    ax.add_patch(earth)

    for name, (stepper, _) in STEPPERS.items():
        _, ys = propagate(stepper, two_body_rhs, y0, t_end, h)
        ax.plot(ys[:, 0], ys[:, 1], lw=1.2, label=f"{name} (h = {h:.0f} s)")

    theta = np.linspace(0, 2 * np.pi, 400)
    r0 = R_EARTH + ALTITUDE_KM
    ax.plot(r0 * np.cos(theta), r0 * np.sin(theta), "k--", lw=0.8, label="exact orbit")

    ax.set_aspect("equal")
    ax.set_xlabel("x [km]")
    ax.set_ylabel("y [km]")
    ax.set_title(f"{n_orbits} orbits at {ALTITUDE_KM:.0f} km, fixed step h = {h:.0f} s")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "orbits.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 1b. Convergence: error vs step size on a log-log scale
# ---------------------------------------------------------------------------

def plot_convergence(step_sizes=(240, 120, 60, 30, 15, 7.5, 3.75, 1.875)):
    """Error after one orbit vs h. The slope of each line is the order."""
    y0 = circular_orbit_state(ALTITUDE_KM)
    period = orbital_period(y0)

    fig, ax = plt.subplots(figsize=(7, 5))
    for name, (stepper, _) in STEPPERS.items():
        hs, errors = [], []
        for h_target in step_sizes:
            n = int(round(period / h_target))
            h = period / n
            ts, ys = propagate(stepper, two_body_rhs, y0, period, h)
            err = np.linalg.norm(ys[-1, :3] - exact_circular_position(ts[-1], ALTITUDE_KM))
            hs.append(h)
            errors.append(err)
        ax.loglog(hs, errors, "o-", label=name)

    # reference slopes, anchored at the smallest step size
    h_ref = np.array([min(step_sizes), max(step_sizes)])
    for p, style in [(1, ":"), (2, "--"), (4, "-.")]:
        ax.loglog(h_ref, 1e-3 * (h_ref / h_ref[0]) ** p, "k" + style, lw=0.8, label=f"slope {p}")

    ax.set_xlabel("step size h [s]")
    ax.set_ylabel("position error after one orbit [km]")
    ax.set_title("Convergence: the slope of each line is the order of the method")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "convergence.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2. SciPy solve_ivp with adaptive step size vs our fixed-step methods
# ---------------------------------------------------------------------------

def compare_with_scipy(n_orbits=10):
    """Accuracy vs cost (number of f-evaluations) for fixed and adaptive methods."""
    y0 = circular_orbit_state(ALTITUDE_KM)
    t_end = n_orbits * orbital_period(y0)
    exact = exact_circular_position(t_end, ALTITUDE_KM)

    results = {}  # name -> list of (f-evals, error)

    # our fixed-step methods
    for name, (stepper, evals_per_step) in STEPPERS.items():
        points = []
        for h in (240, 120, 60, 30, 15, 7.5):
            n = int(round(t_end / h))
            ts, ys = propagate(stepper, two_body_rhs, y0, t_end, t_end / n)
            err = np.linalg.norm(ys[-1, :3] - exact)
            points.append((evals_per_step * n, err))
        results[name] = points

    # SciPy adaptive methods: we set tolerances, SciPy chooses the steps
    print(f"\nSciPy solve_ivp, {n_orbits} orbits")
    print(f"{'method':<7} {'rtol':>8} {'steps':>7} {'f-evals':>8} {'pos. error [km]':>16}")
    for method in ("RK45", "DOP853"):
        points = []
        for rtol in (1e-4, 1e-6, 1e-8, 1e-10, 1e-12):
            f = counted(two_body_rhs)
            sol = solve_ivp(f, (0.0, t_end), y0, method=method, rtol=rtol, atol=rtol * 1e-3)
            err = np.linalg.norm(sol.y[:3, -1] - exact)
            n_steps = len(sol.t) - 1
            points.append((sol.nfev, err))
            print(f"{method:<7} {rtol:8.0e} {n_steps:7d} {sol.nfev:8d} {err:16.3e}")
        results[f"SciPy {method} (adaptive)"] = points

    fig, ax = plt.subplots(figsize=(7, 5))
    for name, points in results.items():
        evals, errs = zip(*points)
        style = "s--" if "SciPy" in name else "o-"
        ax.loglog(evals, errs, style, label=name)
    ax.set_xlabel("number of f-evaluations (cost)")
    ax.set_ylabel(f"position error after {n_orbits} orbits [km]")
    ax.set_title("Accuracy vs cost: fixed-step methods and SciPy adaptive solvers")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "accuracy_vs_cost.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    FIG_DIR.mkdir(exist_ok=True)
    plot_orbits()
    plot_convergence()
    compare_with_scipy()
    print(f"\nFigures saved to {FIG_DIR.resolve()}")
