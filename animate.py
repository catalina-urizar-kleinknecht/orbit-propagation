"""
Animated comparison of Euler, RK2 and RK4: orbits on the left, energy drift on the right.

    uv add matplotlib pillow
    uv run animate.py

Saves figures/orbits_energy.gif
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from orbit_rk import (
    MU_EARTH,
    R_EARTH,
    STEPPERS,
    circular_orbit_state,
    propagate,
    specific_energy,
    two_body_rhs,
)

ALTITUDE_KM = 400.0
H = 60.0           # step size [s], the same for all methods
N_ORBITS = 4
FRAME_STRIDE = 2   # draw every 2nd integration step
TRAIL = 120        # how many past points to show as a trail
COLORS = {"Euler": "#d1495b", "RK2": "#edae49", "RK4": "#00798c"}
OUT = Path("figures") / "orbits_energy.gif"


def simulate():
    y0 = circular_orbit_state(ALTITUDE_KM)
    period = 2 * np.pi * np.sqrt(np.linalg.norm(y0[:3]) ** 3 / MU_EARTH)
    e0 = specific_energy(y0)
    runs = {}
    for name, (stepper, _) in STEPPERS.items():
        ts, ys = propagate(stepper, two_body_rhs, y0, N_ORBITS * period, H)
        drift = np.abs((np.array([specific_energy(y) for y in ys]) - e0) / e0)
        runs[name] = (ts, ys, np.maximum(drift, 1e-16))  # floor for the log scale
    return runs, period


def main():
    runs, period = simulate()
    n_steps = len(next(iter(runs.values()))[0])
    frames = range(0, n_steps, FRAME_STRIDE)

    fig, (ax_orbit, ax_energy) = plt.subplots(1, 2, figsize=(11, 5.2))
    fig.suptitle(f"Same step size (h = {H:.0f} s), {N_ORBITS} orbits at {ALTITUDE_KM:.0f} km", fontsize=13)

    # left: orbits
    r_max = max(np.max(np.linalg.norm(ys[:, :3], axis=1)) for _, ys, _ in runs.values())
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
    ts_hours = next(iter(runs.values()))[0] / 3600
    ax_energy.set_yscale("log")
    ax_energy.set_xlim(0, ts_hours[-1])
    ax_energy.set_ylim(1e-12, 2)
    ax_energy.set_xlabel("time [h]")
    ax_energy.set_ylabel("relative energy drift |E - E0| / |E0|")
    ax_energy.grid(True, which="major", alpha=0.3)

    trails, dots, curves = {}, {}, {}
    for name in runs:
        c = COLORS[name]
        (trails[name],) = ax_orbit.plot([], [], color=c, lw=1.5, label=name)
        (dots[name],) = ax_orbit.plot([], [], "o", color=c, ms=7)
        (curves[name],) = ax_energy.plot([], [], color=c, lw=1.8, label=name)
    ax_orbit.legend(loc="upper right", fontsize=9)
    ax_energy.legend(loc="lower right", fontsize=9)
    clock = ax_orbit.text(0.03, 0.03, "", transform=ax_orbit.transAxes, fontsize=10)

    def update(i):
        artists = []
        for name, (ts, ys, drift) in runs.items():
            start = max(0, i - TRAIL)
            trails[name].set_data(ys[start:i + 1, 0], ys[start:i + 1, 1])
            dots[name].set_data([ys[i, 0]], [ys[i, 1]])
            curves[name].set_data(ts[:i + 1] / 3600, drift[:i + 1])
            artists += [trails[name], dots[name], curves[name]]
        t = next(iter(runs.values()))[0][i]
        clock.set_text(f"t = {t / 3600:4.1f} h   (orbit {t / period:3.1f})")
        return artists + [clock]

    anim = FuncAnimation(fig, update, frames=frames, interval=50, blit=True)
    OUT.parent.mkdir(exist_ok=True)
    anim.save(OUT, writer=PillowWriter(fps=20), dpi=80)
    plt.close(fig)
    print(f"Saved {OUT.resolve()}")


if __name__ == "__main__":
    main()
