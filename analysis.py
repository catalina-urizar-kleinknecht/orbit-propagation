"""
Plots and a SciPy comparison for the integrators in orbitsim.integrators.METHODS.

    uv run analysis.py

Saves four figures to ./figures/ and prints a comparison table.

Runs stop when the satellite hits the surface. Such a run is drawn up to the
impact (marked with an x) in the trajectory plots, and left out of the error
plots, where an error at t_end would mean nothing.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

from orbitsim.integrators import METHODS, ODE
from orbitsim.physics import (
    R_EARTH,
    below_surface,
    circular_orbit_state,
    exact_circular_position,
    orbital_period,
    two_body_jacobian,
    two_body_rhs,
)
from orbitsim.propagate import propagate

ALTITUDE_KM = 400.0
FIG_DIR = Path("figures")


def run(method, y0, t_end, h):
    """Propagate with a fresh evaluation counter, stopping at the surface."""
    return propagate(method.step, ODE(two_body_rhs, two_body_jacobian), y0, t_end, h, stop=below_surface)


def draw_earth_and_exact_orbit(ax):
    ax.add_patch(plt.Circle((0, 0), R_EARTH, color="#cfd8e3", zorder=0, label="Earth"))
    theta = np.linspace(0, 2 * np.pi, 400)
    r0 = R_EARTH + ALTITUDE_KM
    ax.plot(r0 * np.cos(theta), r0 * np.sin(theta), "k--", lw=0.8, label="exact orbit")


# ---------------------------------------------------------------------------
# 1a. Orbits in the x-y plane
# ---------------------------------------------------------------------------

def plot_orbits(h=60.0, n_orbits=3):
    """Same step size for all methods: Euler spirals outwards, implicit Euler falls in."""
    y0 = circular_orbit_state(ALTITUDE_KM)
    t_end = n_orbits * orbital_period(y0)

    fig, ax = plt.subplots(figsize=(7, 7))
    draw_earth_and_exact_orbit(ax)
    for m in METHODS:
        traj = run(m, y0, t_end, h)
        ax.plot(traj.ys[:, 0], traj.ys[:, 1], lw=1.2, color=m.color, label=f"{m.name} (h = {h:.0f} s)")
        if traj.stopped:
            ax.plot(*traj.ys[-1, :2], "x", color=m.color, ms=10, mew=2)

    ax.set_aspect("equal")
    ax.set_xlabel("x [km]")
    ax.set_ylabel("y [km]")
    ax.set_title(f"{n_orbits} orbits at {ALTITUDE_KM:.0f} km, fixed step h = {h:.0f} s")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "orbits.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 1b. Altitude over time: which methods gain or lose energy
# ---------------------------------------------------------------------------

def plot_altitude(h=5.0, n_orbits=3):
    """Altitude vs time. A small h, so that implicit Euler lasts long enough to see the trend."""
    y0 = circular_orbit_state(ALTITUDE_KM)
    period = orbital_period(y0)

    fig, ax = plt.subplots(figsize=(8, 5))
    for m in METHODS:
        traj = run(m, y0, n_orbits * period, h)
        alt = np.linalg.norm(traj.ys[:, :3], axis=1) - R_EARTH
        ax.plot(traj.ts / period, alt, lw=1.4, color=m.color, label=m.name)
        if traj.stopped:
            ax.plot(traj.ts[-1] / period, alt[-1], "x", color=m.color, ms=10, mew=2)

    ax.axhline(ALTITUDE_KM, color="k", ls="--", lw=0.8, label="exact altitude")
    ax.axhline(0.0, color="#8a9bb0", lw=1.0, label="surface")
    ax.set_xlabel("time [orbits]")
    ax.set_ylabel("altitude [km]")
    ax.set_title(f"Altitude with h = {h:.0f} s: explicit Euler gains energy, implicit Euler loses it")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "altitude.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 1c. Convergence: error vs step size on a log-log scale
# ---------------------------------------------------------------------------

def plot_convergence(step_sizes=(240, 120, 60, 30, 15, 7.5, 3.75, 1.875, 0.9375)):
    """Error after one orbit vs h. The slope of each line is the order."""
    y0 = circular_orbit_state(ALTITUDE_KM)
    period = orbital_period(y0)

    fig, ax = plt.subplots(figsize=(7, 5))
    for m in METHODS:
        hs, errors = [], []
        for h_target in step_sizes:
            n = int(round(period / h_target))
            traj = run(m, y0, period, period / n)
            if traj.stopped:
                continue  # crashed before completing the orbit: no error to measure
            hs.append(period / n)
            errors.append(np.linalg.norm(traj.ys[-1, :3] - exact_circular_position(traj.t_end, ALTITUDE_KM)))
        ax.loglog(hs, errors, "o-", color=m.color, label=m.name)

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
    period = orbital_period(y0)
    t_end = n_orbits * period
    exact = exact_circular_position(t_end, ALTITUDE_KM)

    fig, ax = plt.subplots(figsize=(10, 5))

    # our fixed-step methods
    for m in METHODS:
        points = []
        for h in (240, 120, 60, 30, 15, 7.5):
            n = int(round(t_end / h))
            traj = run(m, y0, t_end, t_end / n)
            if not traj.stopped:
                points.append((traj.n_f, np.linalg.norm(traj.ys[-1, :3] - exact)))
        if points:
            evals, errs = zip(*points)
            ax.loglog(evals, errs, "o-", color=m.color, label=m.name)
        else:
            # legend entry only: say why the method is missing
            ax.plot([], [], "x", color=m.color, mew=2,
                    label=f"{m.name}: hits the surface at every h tested")

    # SciPy adaptive methods: we set tolerances, SciPy chooses the steps
    print(f"\nSciPy solve_ivp, {n_orbits} orbits")
    print(f"{'method':<7} {'rtol':>8} {'steps':>7} {'f-evals':>8} {'pos. error [km]':>16}")
    for method, color in (("RK45", "#30638e"), ("DOP853", "#3a3a3a")):
        points = []
        for rtol in (1e-4, 1e-6, 1e-8, 1e-10, 1e-12):
            sol = solve_ivp(two_body_rhs, (0.0, t_end), y0, method=method, rtol=rtol, atol=rtol * 1e-3)
            err = np.linalg.norm(sol.y[:3, -1] - exact)
            n_steps = len(sol.t) - 1
            points.append((sol.nfev, err))
            print(f"{method:<7} {rtol:8.0e} {n_steps:7d} {sol.nfev:8d} {err:16.3e}")
        evals, errs = zip(*points)
        ax.loglog(evals, errs, "s--", color=color, label=f"SciPy {method} (adaptive)")

    ax.set_xlabel("number of f-evaluations (cost)")
    ax.set_ylabel(f"position error after {n_orbits} orbits [km]")
    ax.set_title("Accuracy vs cost: fixed-step methods and SciPy adaptive solvers")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.tight_layout()
    fig.savefig(FIG_DIR / "accuracy_vs_cost.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    FIG_DIR.mkdir(exist_ok=True)
    plot_orbits()
    plot_altitude()
    plot_convergence()
    compare_with_scipy()
    print(f"\nFigures saved to {FIG_DIR.resolve()}")
