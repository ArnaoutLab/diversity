"""Common test fixtures."""
from pytest import fixture

from diversity.shared import SharedArrayManager


@fixture()
def shared_array_manager():
    with SharedArrayManager() as manager:
        yield manager
