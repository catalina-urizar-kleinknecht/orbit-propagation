import matplotlib
import pytest

matplotlib.use("Agg")  # no display on CI


def pytest_addoption(parser):
    parser.addoption(
        "--update-golden",
        action="store_true",
        help="rewrite tests/regression/golden.json from the current code instead of comparing",
    )


@pytest.fixture
def update_golden(request):
    return request.config.getoption("--update-golden")
