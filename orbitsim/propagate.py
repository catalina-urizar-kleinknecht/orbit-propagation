"""
Fixed-step driver: applies a stepper from t = 0 to t_end.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class Trajectory:
    ts: np.ndarray       # times, shape (N,)
    ys: np.ndarray       # states, shape (N, len(y0))
    n_f: int             # evaluations of f
    n_jac: int           # evaluations of the Jacobian (implicit methods only)
    stopped: bool        # True if stop(y) ended the run before t_end

    @property
    def t_end(self):
        return self.ts[-1]


def propagate(step, ode, y0, t_end, h, stop=None):
    """Integrate from t = 0 to t_end with fixed step h.

    stop(y) -> bool ends the run early (e.g. when the satellite hits the
    surface); the trajectory then ends at the first state where it is True.
    """
    n_steps = int(round(t_end / h))
    f0, jac0 = ode.n_f, ode.n_jac
    ts = np.empty(n_steps + 1)
    ys = np.empty((n_steps + 1, len(y0)))
    ts[0], ys[0] = 0.0, y0
    stopped = False
    for i in range(n_steps):
        ys[i + 1] = step(ode, ts[i], ys[i], h)
        ts[i + 1] = (i + 1) * h
        if stop is not None and stop(ys[i + 1]):
            ts, ys, stopped = ts[:i + 2], ys[:i + 2], True
            break
    return Trajectory(ts, ys, ode.n_f - f0, ode.n_jac - jac0, stopped)
