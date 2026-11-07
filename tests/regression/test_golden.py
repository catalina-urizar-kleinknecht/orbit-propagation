"""
Regression tests: the experiment results must match the stored reference values
in golden.json. Catches any change in numerical behaviour, intended or not.

After an intended change (e.g. a new method, a different step list), review
the new numbers and update the reference:

    uv run pytest tests/regression --update-golden
"""

import json
import math
from pathlib import Path

import pytest

import experiments

GOLDEN = Path(__file__).with_name("golden.json")

# Same inputs, same algorithms: results should agree to near machine precision.
# The tolerances leave room for last-bit differences between platforms and
# NumPy builds; ATOL_KM matters for RK4's ~1e-9 km errors, where round-off dominates.
RTOL = 1e-6
ATOL_KM = 1e-9


def current_results():
    return {
        "accuracy": experiments.accuracy_results(),
        "convergence": {
            name: r["errors"] for name, r in experiments.convergence_results().items()
        },
    }


def to_json(results):
    # NaN (run hit the surface) is not valid JSON: store it as null
    clean = {
        "accuracy": results["accuracy"],
        "convergence": {
            name: [None if math.isnan(e) else e for e in errors]
            for name, errors in results["convergence"].items()
        },
    }
    return json.dumps(clean, indent=2) + "\n"


@pytest.fixture(scope="module")
def results():
    return current_results()


@pytest.fixture(scope="module")
def golden():
    if not GOLDEN.exists():
        pytest.fail(f"{GOLDEN.name} missing: run pytest tests/regression --update-golden")
    return json.loads(GOLDEN.read_text())


def close(a, b):
    if a is None or b is None:
        return a is None and b is None
    return math.isclose(a, b, rel_tol=RTOL, abs_tol=ATOL_KM)


def test_update_golden(results, update_golden):
    if not update_golden:
        pytest.skip("only runs with --update-golden")
    GOLDEN.write_text(to_json(results))


def test_same_runs_as_golden(results, golden, update_golden):
    if update_golden:
        pytest.skip("golden being updated")
    have = [(r["method"], r["h"]) for r in results["accuracy"]]
    want = [(r["method"], r["h"]) for r in golden["accuracy"]]
    assert have == want, "methods or step sizes changed: update the golden file"
    assert sorted(results["convergence"]) == sorted(golden["convergence"])


def test_accuracy_matches_golden(results, golden, update_golden):
    if update_golden:
        pytest.skip("golden being updated")
    for r, g in zip(results["accuracy"], golden["accuracy"]):
        label = f"{r['method']}, h = {r['h']}"
        # cost and stopping are exact
        assert (r["n_f"], r["n_jac"], r["stopped"]) == (g["n_f"], g["n_jac"], g["stopped"]), label
        for key in ("orbits", "pos_error", "energy_change"):
            assert close(r[key], g[key]), f"{label}: {key} = {r[key]}, golden {g[key]}"


def test_convergence_matches_golden(results, golden, update_golden):
    if update_golden:
        pytest.skip("golden being updated")
    for name, errors in results["convergence"].items():
        for h, e, g in zip(experiments.CONVERGENCE_STEPS, errors, golden["convergence"][name]):
            e = None if math.isnan(e) else e
            assert close(e, g), f"{name}, h = {h}: error = {e}, golden {g}"
