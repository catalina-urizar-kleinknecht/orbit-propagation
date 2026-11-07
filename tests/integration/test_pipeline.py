"""
End-to-end runs of the scripts: every method through propagate, Newton and LU,
and the figures and animation written to a temporary directory.
"""

import numpy as np
from PIL import Image

import analysis
import animate
import experiments
from orbitsim.integrators import METHODS
from orbitsim.physics import circular_orbit_state, orbital_period


def by_method(rows):
    return {(r["method"], r["h"]): r for r in rows}


def test_experiments_print_every_method(capsys):
    experiments.experiment_accuracy()
    experiments.experiment_convergence()
    out = capsys.readouterr().out
    assert "Experiment 1" in out and "Experiment 2" in out
    for m in METHODS:
        assert m.name in out


def test_explicit_and_implicit_euler_drift_in_opposite_directions():
    rows = by_method(experiments.accuracy_results())
    period = orbital_period(circular_orbit_state(experiments.ALTITUDE_KM))
    for h in experiments.ACCURACY_STEPS:
        explicit, implicit = rows[("Euler", h)], rows[("Implicit Euler", h)]
        assert not explicit["stopped"] and explicit["energy_change"] > 0
        assert implicit["stopped"] and implicit["energy_change"] < 0
        assert implicit["pos_error"] is None
        # per step: one f for the predictor, plus one f and one Jacobian per Newton iteration
        n_steps = round(implicit["orbits"] * period / h)
        assert implicit["n_f"] == n_steps + implicit["n_jac"]


def test_rk4_stays_on_the_orbit():
    rows = by_method(experiments.accuracy_results())
    assert rows[("RK4", 10.0)]["pos_error"] < 1e-3          # km, after 10 orbits
    assert abs(rows[("RK4", 10.0)]["energy_change"]) < 1e-9


def test_convergence_orders_match_theory():
    results = experiments.convergence_results()
    for m in METHODS:
        # last order estimate before round-off: second to last halving
        orders = [p for p in results[m.name]["orders"] if np.isfinite(p)]
        assert abs(orders[-2] - m.order) < 0.1, (m.name, orders)


def test_analysis_writes_all_figures(tmp_path, monkeypatch):
    monkeypatch.setattr(analysis, "FIG_DIR", tmp_path)
    analysis.plot_orbits()
    analysis.plot_altitude()
    analysis.plot_convergence()
    analysis.compare_with_scipy()
    for name in ("orbits", "altitude", "convergence", "accuracy_vs_cost"):
        with Image.open(tmp_path / f"{name}.png") as im:
            assert im.width > 500 and im.height > 300


def test_animation_writes_gif(tmp_path, monkeypatch):
    out = tmp_path / "anim.gif"
    monkeypatch.setattr(animate, "OUT", out)
    monkeypatch.setattr(animate, "N_ORBITS", 1)  # keep it short
    animate.main()
    with Image.open(out) as im:
        assert im.n_frames > 10
