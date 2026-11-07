"""
Fixed-step integrators and the registry of methods used by all scripts.

Every stepper has the signature step(ode, t, y, h) -> y_{n+1}, where ode
provides ode.f(t, y) and, for implicit methods, ode.jac(t, y).

To add a method: write its step function here and append it to METHODS.
The experiments, plots and the animation pick it up from there.
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .newton import newton_solve


class ODE:
    """dy/dt = f(t, y), with an optional Jacobian df/dy.

    Counts how often f and the Jacobian are evaluated, as a measure of cost.
    """

    def __init__(self, rhs, jac=None):
        self.rhs = rhs
        self.jacobian = jac
        self.n_f = 0
        self.n_jac = 0

    def f(self, t, y):
        self.n_f += 1
        return self.rhs(t, y)

    def jac(self, t, y):
        if self.jacobian is None:
            raise ValueError("this method needs the Jacobian df/dy, but the ODE has none")
        self.n_jac += 1
        return self.jacobian(t, y)


# ---------------------------------------------------------------------------
# Explicit methods: y_{n+1} from slopes evaluated at known states
# ---------------------------------------------------------------------------

def euler_step(ode, t, y, h):
    """Explicit Euler (order 1): one evaluation of f per step."""
    # follow the tangent at the start of the step
    return y + h * ode.f(t, y)


def rk2_step(ode, t, y, h):
    """Midpoint method (order 2): two evaluations of f per step."""
    # use the slope at the midpoint, reached with half an Euler step
    k1 = ode.f(t, y)
    k2 = ode.f(t + h / 2, y + h / 2 * k1)
    return y + h * k2


def rk4_step(ode, t, y, h):
    """Classical Runge-Kutta (order 4): four evaluations of f per step."""
    # weighted average of slopes: start, twice at the midpoint, end (1:2:2:1)
    k1 = ode.f(t, y)
    k2 = ode.f(t + h / 2, y + h / 2 * k1)
    k3 = ode.f(t + h / 2, y + h / 2 * k2)
    k4 = ode.f(t + h, y + h * k3)
    return y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


# ---------------------------------------------------------------------------
# Implicit methods: y_{n+1} appears on both sides, solved with Newton (-> LU)
# ---------------------------------------------------------------------------

def implicit_euler_step(ode, t, y, h):
    """Implicit (backward) Euler (order 1): uses the slope at the END of the step.

    y_{n+1} = y_n + h f(t_{n+1}, y_{n+1}) is solved for z = y_{n+1} as
        G(z)  = z - y_n - h f(t_{n+1}, z) = 0
        dG(z) = I - h J(t_{n+1}, z)
    Each Newton iteration costs one f, one Jacobian and one LU solve.
    """
    t_new = t + h
    I = np.eye(len(y))
    z0 = y + h * ode.f(t, y)  # explicit Euler as the initial guess
    return newton_solve(
        G=lambda z: z - y - h * ode.f(t_new, z),
        dG=lambda z: I - h * ode.jac(t_new, z),
        z0=z0,
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Method:
    name: str
    step: Callable
    order: int
    color: str  # the same color in every figure


METHODS = [
    Method("Euler", euler_step, 1, "#d1495b"),
    Method("Implicit Euler", implicit_euler_step, 1, "#6a4c93"),
    Method("RK2", rk2_step, 2, "#edae49"),
    Method("RK4", rk4_step, 4, "#00798c"),
]
