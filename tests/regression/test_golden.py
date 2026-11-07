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

# Same inputs, same algorithms, but not bit-identical results across machines:
# NumPy's OpenBLAS picks CPU-specific kernels (e.g. AVX-512 on GitHub runners),
# which round differently in the last bit. Over N steps that noise grows to about
# N * eps * |r| ~ 1e4 * 2e-16 * 7e3 km ~ 1e-8 km, so absolute tolerances sit
# above that floor. Check locally with e.g. OPENBLAS_CORETYPE=SkylakeX.
RTOL = 1e-6
ATOL = {
    "orbits": 1e-9,          # t_end / period: exact multiples of h, no round-off growth
    "pos_error": 1e-7,       # km (0.1 mm); RK4's smallest errors (~1e-9 km) are pure round-off
    "energy_change": 1e-12,  # relative; RK4's smallest drift is 3e-10
}


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


def close(a, b, atol):
    if a is None or b is None:
        return a is None and b is None
    return math.isclose(a, b, rel_tol=RTOL, abs_tol=atol)


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


def report(mismatches):
    """Fail with every mismatch at once, so one CI run shows the whole picture."""
    assert not mismatches, f"{len(mismatches)} value(s) differ from golden.json:\n  " + "\n  ".join(mismatches)


def test_accuracy_matches_golden(results, golden, update_golden):
    if update_golden:
        pytest.skip("golden being updated")
    mismatches = []
    for r, g in zip(results["accuracy"], golden["accuracy"]):
        label = f"{r['method']}, h = {r['h']}"
        # cost and stopping are exact
        for key in ("n_f", "n_jac", "stopped"):
            if r[key] != g[key]:
                mismatches.append(f"{label}: {key} = {r[key]}, golden {g[key]}")
        for key in ("orbits", "pos_error", "energy_change"):
            if not close(r[key], g[key], ATOL[key]):
                mismatches.append(f"{label}: {key} = {r[key]}, golden {g[key]}")
    report(mismatches)


def test_convergence_matches_golden(results, golden, update_golden):
    if update_golden:
        pytest.skip("golden being updated")
    mismatches = []
    for name, errors in results["convergence"].items():
        for h, e, g in zip(experiments.CONVERGENCE_STEPS, errors, golden["convergence"][name]):
            e = None if math.isnan(e) else e
            if not close(e, g, ATOL["pos_error"]):
                mismatches.append(f"{name}, h = {h}: error = {e}, golden {g}")
    report(mismatches)
