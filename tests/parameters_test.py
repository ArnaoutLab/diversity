"""Tests for diversity.parameters."""
from argparse import ArgumentParser, Namespace
from warnings import filterwarnings, resetwarnings

from pytest import mark, raises

from diversity.parameters import configure_arguments
from diversity.utilities import ArgumentWarning

CONFIGURE_ARGUMENTS_TEST_CASES = [
    {
        "description": "Warn viewpoint > 100",
        "args": ["-v", "1", "2", "0", "inf", "101"],
        "expect_raise": True,
    },
    {
        "description": "Do not warn valid viewpoint",
        "args": ["-v", "0", "100", "inf"],
        "expect_raise": False,
    },
]


class TestConfigureArguments:
    """Tests diversity.parameters.configure_arguments."""

    @mark.parametrize("test_case", CONFIGURE_ARGUMENTS_TEST_CASES)
    def test_configure_arguments(self, test_case):
        parser = configure_arguments()
        if test_case["expect_raise"]:
            filterwarnings("error", category=ArgumentWarning)
            with raises(ArgumentWarning):
                parser.parse_args(test_case["args"])
            resetwarnings()
        else:
            parser.parse_args(test_case["args"])
