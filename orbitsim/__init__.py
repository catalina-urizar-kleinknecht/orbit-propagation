"""
Orbit propagation with hand-written integrators.

    physics.py      two-body ODE, Jacobian, energy, exact circular orbit
    integrators.py  steppers (explicit and implicit) and the METHODS registry
    newton.py       Newton's method, used by implicit steppers
    linalg.py       LU decomposition, used by Newton
    propagate.py    fixed-step driver
"""
