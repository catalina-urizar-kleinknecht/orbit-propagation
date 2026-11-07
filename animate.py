"""
Animated comparison of all methods in orbitsim.integrators.METHODS: orbits on
the left, energy drift on the right.

    uv run animate.py

Saves figures/orbits_energy.gif. A run that hits the surface stops there and
is marked with an x; its energy curve ends at the impact.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from orbitsim.integrators import METHODS, ODE
from orbitsim.physics import (
    R_EARTH,
    below_surface,
    circular_orbit_state,
    orbital_period,
    specific_energy,
    two_body_jacobian,
    two_body_rhs,
)
from orbitsim.propagate import propagate

ALTITUDE_KM = 400.0
H = 60.0           # step size [s], the same for all methods
N_ORBITS = 4
FRAME_STRIDE = 2   # draw every 2nd integration step
TRAIL = 120        # how many past points to show as a trail
OUT = Path("figures") / "orbits_energy.gif"


def simulate():
    y0 = circular_orbit_state(ALTITUDE_KM)
    period = orbital_period(y0)
    e0 = specific_energy(y0)
    runs = {}
    for m in METHODS:
        traj = propagate(m.step, ODE(two_body_rhs, two_body_jacobian), y0, N_ORBITS * period, H,
                         stop=below_surface)
        drift = np.abs((np.array([specific_energy(y) for y in traj.ys]) - e0) / e0)
        runs[m] = (traj, np.maximum(drift, 1e-16))  # floor for the log scale
    return runs, period


def main():
    runs, period = simulate()
    n_steps = int(round(N_ORBITS * period / H)) + 1
    frames = range(0, n_steps, FRAME_STRIDE)

    fig, (ax_orbit, ax_energy) = plt.subplots(1, 2, figsize=(11, 5.2))
    fig.suptitle(f"Same step size (h = {H:.0f} s), {N_ORBITS} orbits at {ALTITUDE_KM:.0f} km", fontsize=13)

    # left: orbits
    r_max = max(np.max(np.linalg.norm(traj.ys[:, :3], axis=1)) for traj, _ in runs.values())
    lim = 1.05 * r_max
    ax_orbit.set_xlim(-lim, lim)
    ax_orbit.set_ylim(-lim, lim)
    ax_orbit.set_aspect("equal")
    ax_orbit.add_patch(plt.Circle((0, 0), R_EARTH, color="#cfd8e3", zorder=0))
    theta = np.linspace(0, 2 * np.pi, 400)
    r0 = R_EARTH + ALTITUDE_KM
    ax_orbit.plot(r0 * np.cos(theta), r0 * np.sin(theta), "k--", lw=0.7, label="exact orbit")
    ax_orbit.set_xlabel("x [km]")
    ax_orbit.set_ylabel("y [km]")

    # right: energy drift
    ax_energy.set_yscale("log")
    ax_energy.set_xlim(0, (n_steps - 1) * H / 3600)
    ax_energy.set_ylim(1e-12, 2)
    ax_energy.set_xlabel("time [h]")
    ax_energy.set_ylabel("relative energy drift |E - E0| / |E0|")
    ax_energy.grid(True, which="major", alpha=0.3)

    trails, dots, curves = {}, {}, {}
    for m in runs:
        (trails[m],) = ax_orbit.plot([], [], color=m.color, lw=1.5, label=m.name)
        (dots[m],) = ax_orbit.plot([], [], "o", color=m.color, ms=7)
        (curves[m],) = ax_energy.plot([], [], color=m.color, lw=1.8, label=m.name)
    ax_orbit.legend(loc="upper right", fontsize=9)
    ax_energy.legend(loc="lower right", fontsize=9)
    clock = ax_orbit.text(0.03, 0.03, "", transform=ax_orbit.transAxes, fontsize=10)

    def update(i):
        artists = []
        for m, (traj, drift) in runs.items():
            # a run that stopped early stays frozen at its last state
            j = min(i, len(traj.ts) - 1)
            crashed = traj.stopped and i >= j
            start = max(0, j - TRAIL)
            trails[m].set_data(traj.ys[start:j + 1, 0], traj.ys[start:j + 1, 1])
            dots[m].set_data([traj.ys[j, 0]], [traj.ys[j, 1]])
            dots[m].set_marker("x" if crashed else "o")
            dots[m].set_markeredgewidth(2.5 if crashed else 1.0)
            curves[m].set_data(traj.ts[:j + 1] / 3600, drift[:j + 1])
            artists += [trails[m], dots[m], curves[m]]
        t = i * H
        clock.set_text(f"t = {t / 3600:4.1f} h   (orbit {t / period:3.1f})")
        return artists + [clock]

    anim = FuncAnimation(fig, update, frames=frames, interval=50, blit=True)
    OUT.parent.mkdir(exist_ok=True)
    anim.save(OUT, writer=PillowWriter(fps=20), dpi=80)
    plt.close(fig)
    print(f"Saved {OUT.resolve()}")


if __name__ == "__main__":
    main()
