"""Common test fixtures."""
from pytest import fixture

from diversity.shared import SharedArrayManager

# pytest_plugins = ["pytest_profiling"]


@fixture()
def shared_array_manager():
    with SharedArrayManager() as manager:
        yield manager
