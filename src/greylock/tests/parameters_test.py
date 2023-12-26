"""Tests for greylock.parameters."""
from pytest import mark, warns

from greylock.parameters import configure_arguments
from greylock.exceptions import ArgumentWarning

CONFIGURE_ARGUMENTS_TEST_CASES = [
    {
        "description": "Warn viewpoint > 100",
        "args": ["-v", "1", "2", "0", "inf", "101"],
        "expect_warning": True,
    },
    {
        "description": "Do not warn valid viewpoint",
        "args": ["-v", "0", "100", "inf"],
        "expect_warning": False,
    },
]


class TestConfigureArguments:
    """Tests parameters.configure_arguments."""

    @mark.parametrize("test_case", CONFIGURE_ARGUMENTS_TEST_CASES)
    def test_configure_arguments(self, test_case):
        parser = configure_arguments()
        if test_case["expect_warning"]:
            with warns(ArgumentWarning):
                parser.parse_args(test_case["args"])
        else:
            parser.parse_args(test_case["args"])
