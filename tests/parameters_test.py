import sys

sys.path = ['/Users/arnaoutlab/Desktop/diversity_package/src',
 '/Users/arnaoutlab/Desktop/diversity_package',
 '/Users/arnaoutlab/Desktop',
 '/Users/arnaoutlab/Desktop',
 '/Users/arnaoutlab/Desktop/~/Documents/GitHub/vader',
 '/Users/arnaoutlab/.pyenv/versions/3.9.16/lib/python39.zip',
 '/Users/arnaoutlab/.pyenv/versions/3.9.16/lib/python3.9',
 '/Users/arnaoutlab/.pyenv/versions/3.9.16/lib/python3.9/lib-dynload',
 '',
 '/Users/arnaoutlab/.local/lib/python3.9/site-packages',
 '/Users/arnaoutlab/.pyenv/versions/3.9.16/lib/python3.9/site-packages']

"""Tests for diversity.parameters."""
from pytest import mark, warns

from diversity.parameters import configure_arguments
from diversity.exceptions import ArgumentWarning

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
