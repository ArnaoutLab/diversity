"""Tests for diversity.utilities."""

from numpy import allclose, array, dtype, inf, ones
from pandas import DataFrame
from pytest import mark, raises, warns

from diversity.exceptions import ArgumentWarning, InvalidArgumentError
from diversity.utilities import (
    get_file_delimiter,
    power_mean,
)

GET_FILE_DELIMITER_TEST_CASES = [
    {
        "description": ".csv extension; filename only.",
        "filepath": "filename.csv",
        "delimiter": ",",
        "expect_warning": False,
    },
    {
        "description": ".csv extension; full path.",
        "filepath": "/path/to/file.csv",
        "delimiter": ",",
        "expect_warning": False,
    },
    {
        "description": ".tsv extension; filename only.",
        "filepath": "filename.tsv",
        "delimiter": "\t",
        "expect_warning": False,
    },
    {
        "description": ".tsv extension; full path.",
        "filepath": "/path/to/file.tsv",
        "delimiter": "\t",
        "expect_warning": False,
    },
    {
        "description": "No extension; filename only.",
        "filepath": "filename",
        "delimiter": "\t",
        "expect_warning": True,
    },
    {
        "description": "No extension; full path.",
        "filepath": "/path/to/file",
        "delimiter": "\t",
        "expect_warning": True,
    },
    {
        "description": ".tsv.csv extension; filename only.",
        "filepath": "filename.tsv.csv",
        "delimiter": ",",
        "expect_warning": False,
    },
    {
        "description": ".tsv.csv extension; full path.",
        "filepath": "/path/to/file.tsv.csv",
        "delimiter": ",",
        "expect_warning": False,
    },
    {
        "description": ".csv.tsv extension; filename only.",
        "filepath": "filename.csv.tsv",
        "delimiter": "\t",
        "expect_warning": False,
    },
    {
        "description": ".csv.tsv extension; full path.",
        "filepath": "/path/to/file.csv.tsv",
        "delimiter": "\t",
        "expect_warning": False,
    },
]


class TestGetFileDelimiter:
    """Tests utilities.get_file_delimiter."""

    @mark.parametrize("test_case", GET_FILE_DELIMITER_TEST_CASES)
    def test_delimiter(self, test_case):
        """Tests get_file_delimiter."""
        if test_case["expect_warning"]:
            with warns(ArgumentWarning):
                delimiter = get_file_delimiter(test_case["filepath"])
        else:
            delimiter = get_file_delimiter(test_case["filepath"])
        assert delimiter == test_case["delimiter"]


POWER_MEAN_TEST_CASES = [
    {
        "description": "No zero weights; nonzero order; -100 <= order <= 100; 2-d data.",
        "order": 3,
        "weights": array(
            [
                [1.0e-07, 1.0e-07, 1.0e-07],
                [2.0e-08, 2.0e-08, 2.0e-08],
                [1.0e-01, 1.0e-01, 1.0e-01],
            ]
        ),
        "items": array(
            [
                [0.30, 0.69, 0.53],
                [0.74, 0.14, 0.53],
                [0.71, 0.13, 0.86],
            ]
        ),
        "atol": 1e-8,
        "expected_result": array([0.32955284, 0.06034367, 0.39917668]),
        "expect_raise": False,
    },
    {
        "description": "No zero weights; nonzero order; -100 <= order <= 100; 2-d data.",
        "order": 1,
        "weights": array(
            [
                [1.0e-07, 1.0e-07, 1.0e-07],
                [2.0e-08, 2.0e-08, 2.0e-08],
                [1.0e-01, 1.0e-01, 1.0e-01],
            ]
        ),
        "items": array(
            [
                [0.30, 0.69, 0.53],
                [0.74, 0.14, 0.53],
                [0.71, 0.13, 0.86],
            ]
        ),
        "atol": 1e-8,
        "expected_result": array([0.07100004, 0.01300007, 0.08600006]),
        "expect_raise": False,
    },
    {
        "description": "No zero weights; nonzero order; -100 <= order <= 100; 2-d data.",
        "order": -100,
        "weights": array(
            [
                [1.0e-07, 1.0e-07, 1.0e-07],
                [2.0e-08, 2.0e-08, 2.0e-08],
                [1.0e-01, 1.0e-01, 1.0e-01],
            ]
        ),
        "items": array(
            [
                [0.30, 0.69, 0.53],
                [0.74, 0.14, 0.53],
                [0.71, 0.13, 0.86],
            ]
        ),
        "atol": 1e-8,
        "expected_result": array([0.35246927, 0.13302809, 0.62156143]),
        "expect_raise": False,
    },
    {
        "description": "Some zero weights; nonzero order; -100 <= order <= 100; 2-d data.",
        "order": 2,
        "weights": array(
            [
                [1.0e-09, 1.0e-09, 1.0e-09],
                [0.0e-00, 0.0e-00, 0.0e-00],
                [1.0e-07, 1.0e-07, 1.0e-04],
                [0.0e-00, 0.0e-00, 0.0e-00],
                [2.0e-08, 2.0e-08, 4.0e-08],
                [1.0e-01, 1.0e-01, 2.0e-01],
            ]
        ),
        "items": array(
            [
                [0.52, 0.73, 0.85],
                [0.56, 0.28, 0.99],
                [0.30, 0.69, 0.53],
                [0.65, 0.65, 0.65],
                [0.74, 0.14, 0.53],
                [0.71, 0.13, 0.86],
            ]
        ),
        "atol": 1e-8,
        "expected_result": array([0.22452176, 0.04111019, 0.3846402231]),
        "expect_raise": False,
    },
    {
        "description": "Zero weight column; nonzero order; -100 <= order <= 100; 2-d data.",
        "order": 2,
        "weights": array(
            [
                [1.0e-09, 1.0e-09, 1.0e-02],
                [0.0e-00, 0.0e-00, 0.0e-00],
                [1.0e-10, 1.0e-10, 1.0e-02],
                [0.0e-00, 0.0e-00, 0.0e-00],
                [2.0e-11, 2.0e-11, 2.0e-02],
                [0.5e-08, 0.5e-08, 0.5e-01],
            ]
        ),
        "items": array(
            [
                [0.52, 0.73, 0.85],
                [0.56, 0.28, 0.99],
                [0.30, 0.69, 0.53],
                [0.65, 0.65, 0.65],
                [0.74, 0.14, 0.53],
                [0.71, 0.13, 0.86],
            ]
        ),
        "atol": 1e-8,
        "expected_result": None,
        "expect_raise": True,
    },
    {
        "description": "No zero weights; zero order; 2-d data.",
        "order": 0,
        "weights": array(
            [
                [1.0e-07, 1.0e-07, 1.0e-07],
                [2.0e-08, 2.0e-08, 2.0e-08],
                [1.0e-01, 1.0e-01, 1.0e-01],
            ]
        ),
        "items": array(
            [
                [0.30, 0.69, 0.53],
                [0.74, 0.14, 0.53],
                [0.71, 0.13, 0.86],
            ]
        ),
        "atol": 1e-8,
        "expected_result": array([0.96633071, 0.8154443, 0.9850308]),
        "expect_raise": False,
    },
    {
        "description": "Some zero weights; zero order; 2-d data.",
        "order": 0,
        "weights": array(
            [
                [1.0e-09, 1.0e-09, 1.0e-09],
                [0.0e-00, 0.0e-00, 0.0e-00],
                [1.0e-07, 1.0e-07, 1.0e-04],
                [0.0e-00, 0.0e-00, 0.0e-00],
                [2.0e-08, 2.0e-08, 4.0e-08],
                [1.0e-01, 1.0e-01, 2.0e-01],
            ]
        ),
        "items": array(
            [
                [0.52, 0.73, 0.85],
                [0.56, 0.28, 0.99],
                [0.30, 0.69, 0.53],
                [0.65, 0.65, 0.65],
                [0.74, 0.14, 0.53],
                [0.71, 0.13, 0.86],
            ]
        ),
        "atol": 1e-8,
        "expected_result": array([0.96633071, 0.8154443, 0.9702242087]),
        "expect_raise": False,
    },
    {
        "description": "All zero weights; zero order; 2-d data.",
        "order": 0,
        "weights": array(
            [
                [1.0e-09, 1.0e-09, 1.0e-09],
                [0.0e-00, 0.0e-00, 0.0e-00],
                [1.0e-10, 1.0e-10, 1.0e-10],
                [0.0e-00, 0.0e-00, 0.0e-00],
                [2.0e-11, 2.0e-11, 2.0e-11],
                [0.5e-08, 0.5e-08, 0.5e-08],
            ]
        ),
        "items": array(
            [
                [0.52, 0.73, 0.85],
                [0.56, 0.28, 0.99],
                [0.30, 0.69, 0.53],
                [0.65, 0.65, 0.65],
                [0.74, 0.14, 0.53],
                [0.71, 0.13, 0.86],
            ]
        ),
        "atol": 1e-8,
        "expected_result": None,
        "expect_raise": True,
    },
    {
        "description": "No zero weights; order < -100; 2-d data.",
        "order": -101,
        "weights": array(
            [
                [1.0e-07, 1.0e-07, 1.0e-07],
                [2.0e-08, 2.0e-08, 2.0e-08],
                [1.0e-01, 1.0e-01, 1.0e-01],
            ]
        ),
        "items": array(
            [
                [0.30, 0.69, 0.53],
                [0.74, 0.14, 0.53],
                [0.71, 0.13, 0.86],
            ]
        ),
        "atol": 1e-8,
        "expected_result": array([0.30, 0.13, 0.53]),
        "expect_raise": False,
    },
    {
        "description": "Some zero weights; order == -inf; 2-d data.",
        "order": -inf,
        "weights": array(
            [
                [1.0e-09, 1.0e-09, 1.0e-09],
                [0.0e-00, 0.0e-00, 0.0e-00],
                [1.0e-07, 1.0e-07, 1.0e-04],
                [0.0e-00, 0.0e-00, 0.0e-00],
                [2.0e-08, 2.0e-08, 4.0e-08],
                [1.0e-01, 1.0e-01, 2.0e-01],
            ]
        ),
        "items": array(
            [
                [0.28, 0.10, 0.51],
                [0.56, 0.28, 0.99],
                [0.30, 0.69, 0.53],
                [0.28, 0.09, 0.14],
                [0.74, 0.14, 0.53],
                [0.71, 0.13, 0.86],
            ]
        ),
        "atol": 1e-8,
        "expected_result": array([0.30, 0.13, 0.53]),
        "expect_raise": False,
    },
    {
        "description": "Zero weight columns; order < -100; 2-d data.",
        "order": -382,
        "weights": array(
            [
                [1.0e-03, 1.0e-01, 1.0e-09],
                [0.0e-00, 0.0e-00, 0.0e-00],
                [1.0e-00, 1.0e-00, 1.0e-10],
                [0.0e-00, 0.0e-00, 0.0e-00],
                [2.0e-01, 2.0e-01, 2.0e-11],
                [0.5e-00, 0.5e-02, 0.5e-08],
            ]
        ),
        "items": array(
            [
                [0.52, 0.73, 0.85],
                [0.56, 0.28, 0.99],
                [0.30, 0.69, 0.53],
                [0.65, 0.65, 0.65],
                [0.74, 0.14, 0.53],
                [0.71, 0.13, 0.86],
            ]
        ),
        "atol": 1e-8,
        "expected_result": None,
        "expect_raise": True,
    },
    {
        "description": "No zero weights; order > 100; 2-d data.",
        "order": 101,
        "weights": array(
            [
                [1.0e-07, 1.0e-07, 1.0e-07],
                [2.0e-08, 2.0e-08, 2.0e-08],
                [1.0e-01, 1.0e-01, 1.0e-01],
            ]
        ),
        "items": array(
            [
                [0.30, 0.69, 0.53],
                [0.74, 0.14, 0.53],
                [0.71, 0.13, 0.86],
            ]
        ),
        "atol": 1e-8,
        "expected_result": array([0.74, 0.69, 0.86]),
        "expect_raise": False,
    },
    {
        "description": "Some zero weights; order == inf; 2-d data.",
        "order": inf,
        "weights": array(
            [
                [1.0e-09, 1.0e-09, 1.0e-09],
                [0.0e-00, 0.0e-00, 0.0e-00],
                [1.0e-07, 1.0e-07, 1.0e-04],
                [0.0e-00, 0.0e-00, 0.0e-00],
                [2.0e-08, 2.0e-08, 4.0e-08],
                [1.0e-01, 1.0e-01, 2.0e-01],
            ]
        ),
        "items": array(
            [
                [0.99, 0.79, 0.90],
                [0.56, 0.28, 0.99],
                [0.30, 0.69, 0.53],
                [0.89, 0.09, 0.14],
                [0.74, 0.14, 0.53],
                [0.71, 0.13, 0.86],
            ]
        ),
        "atol": 1e-8,
        "expected_result": array([0.74, 0.69, 0.86]),
        "expect_raise": False,
    },
    {
        "description": "All zero weights; order > 100; 2-d data.",
        "order": 364,
        "weights": array(
            [
                [1.0e-09, 1.0e-09, 1.0e-09],
                [0.0e-00, 0.0e-00, 0.0e-00],
                [1.0e-10, 1.0e-10, 1.0e-10],
                [0.0e-00, 0.0e-00, 0.0e-00],
                [2.0e-11, 2.0e-11, 2.0e-11],
                [0.5e-08, 0.5e-08, 0.5e-08],
            ]
        ),
        "items": array(
            [
                [0.52, 0.73, 0.85],
                [0.56, 0.28, 0.99],
                [0.30, 0.69, 0.53],
                [0.65, 0.65, 0.65],
                [0.74, 0.14, 0.53],
                [0.71, 0.13, 0.86],
            ]
        ),
        "atol": 1e-8,
        "expected_result": None,
        "expect_raise": True,
    },
    {
        "description": "Some zero weights; nonzero order; -100 <= order <= 100; 1-d data",
        "order": 2,
        "weights": array(
            [
                1.0e-09,
                0.0e-00,
                1.0e-07,
                0.0e-00,
                2.0e-08,
                1.0e-01,
            ]
        ),
        "items": array(
            [
                0.52,
                0.56,
                0.30,
                0.65,
                0.74,
                0.71,
            ]
        ),
        "atol": 1e-8,
        "expected_result": array([0.22452176]),
        "expect_raise": False,
    },
    {
        "description": "Some zero weights; zero order; 1-d data.",
        "order": 0,
        "weights": array(
            [
                1.0e-09,
                0.0e-00,
                1.0e-07,
                0.0e-00,
                2.0e-08,
                1.0e-01,
            ]
        ),
        "items": array(
            [
                0.52,
                0.56,
                0.30,
                0.65,
                0.74,
                0.71,
            ]
        ),
        "atol": 1e-8,
        "expected_result": array([0.96633071]),
        "expect_raise": False,
    },
    {
        "description": "Some zero weights; order == -inf; 1-d data.",
        "order": -inf,
        "weights": array(
            [
                1.0e-09,
                0.0e-00,
                1.0e-07,
                0.0e-00,
                2.0e-08,
                1.0e-01,
            ]
        ),
        "items": array(
            [
                0.52,
                0.56,
                0.30,
                0.28,
                0.74,
                0.71,
            ]
        ),
        "atol": 1e-8,
        "expected_result": array([0.30]),
        "expect_raise": False,
    },
    {
        "description": "Some zero weights; order == inf; 1-d data.",
        "order": inf,
        "weights": array(
            [
                1.0e-09,
                0.0e-00,
                1.0e-07,
                0.0e-00,
                2.0e-08,
                1.0e-01,
            ]
        ),
        "items": array(
            [
                0.52,
                0.56,
                0.30,
                0.89,
                0.74,
                0.71,
            ]
        ),
        "atol": 1e-8,
        "expected_result": array([0.74]),
        "expect_raise": False,
    },
    {
        "description": "Zero weights; nonzero order; -100 <= order <= 100; 1-d data",
        "order": 2,
        "weights": array(
            [
                1.0e-09,
                0.0e-00,
                1.0e-10,
                0.0e-00,
                0.5e-08,
                1.0e-09,
            ]
        ),
        "items": array(
            [
                0.52,
                0.56,
                0.30,
                0.65,
                0.74,
                0.71,
            ]
        ),
        "atol": 1e-8,
        "expected_result": None,
        "expect_raise": True,
    },
    {
        "description": "Some zero weights; nonzero order; -100 <= order <= 100; 2-d data; atol == 1e-9.",
        "order": 2,
        "weights": array(
            [
                [1.1e-09, 1.1e-09, 1.1e-09],
                [0.0e-00, 0.0e-00, 0.0e-00],
                [1.0e-07, 1.0e-07, 1.0e-07],
                [0.0e-00, 0.0e-00, 0.0e-00],
                [2.0e-08, 2.0e-08, 4.0e-08],
                [1.0e-06, 1.0e-06, 2.0e-06],
            ]
        ),
        "items": array(
            [
                [0.52, 0.73, 0.85],
                [0.56, 0.28, 0.99],
                [0.30, 0.69, 0.53],
                [0.65, 0.65, 0.65],
                [0.74, 0.14, 0.53],
                [0.71, 0.13, 0.86],
            ]
        ),
        "atol": 1e-9,
        "expected_result": array([0.0007241198, 0.0002559066, 0.001232607298]),
        "expect_raise": False,
    },
    {
        "description": "Some zero weights; zero order; 2-d data; atol == 1e-2.",
        "order": 0,
        "weights": array(
            [
                [1.1e-02, 1.1e-02, 1.1e-02],
                [0.0e-00, 0.0e-00, 1.0e-3],
                [1.0e-01, 1.0e-01, 1.0e-01],
                [0.0e-00, 0.0e-00, 0.0e-00],
                [4.0e-02, 4.0e-01, 4.0e-01],
                [2.0e-01, 2.0e-02, 2.0e-02],
            ]
        ),
        "items": array(
            [
                [0.52, 0.73, 0.85],
                [0.56, 0.28, 0.99],
                [0.30, 0.69, 0.53],
                [0.65, 0.65, 0.65],
                [0.74, 0.14, 0.53],
                [0.71, 0.13, 0.86],
            ]
        ),
        "atol": 1e-2,
        "expected_result": array([0.8120992339, 0.4198668065, 0.7245218911]),
        "expect_raise": False,
    },
    {
        "description": "Some zero weights; order == -inf; 2-d data; atol == 1e-9.",
        "order": -inf,
        "weights": array(
            [
                [1.1e-09, 1.1e-09, 1.1e-09],
                [0.0e-00, 0.0e-00, 0.0e-00],
                [1.0e-07, 1.0e-07, 1.0e-07],
                [0.0e-00, 0.0e-00, 0.0e-00],
                [2.0e-08, 2.0e-08, 4.0e-08],
                [1.0e-01, 1.0e-01, 2.0e-06],
            ]
        ),
        "items": array(
            [
                [0.28, 0.10, 0.51],
                [0.56, 0.28, 0.99],
                [0.30, 0.69, 0.53],
                [0.28, 0.09, 0.14],
                [0.74, 0.14, 0.53],
                [0.71, 0.13, 0.86],
            ]
        ),
        "atol": 1e-9,
        "expected_result": array([0.28, 0.10, 0.51]),
        "expect_raise": False,
    },
    {
        "description": "Some zero weights; order == inf; 2-d data; atol == 1e-9.",
        "order": inf,
        "weights": array(
            [
                [1.1e-09, 1.1e-09, 1.1e-09],
                [0.0e-00, 0.0e-00, 0.0e-00],
                [1.0e-07, 1.0e-07, 1.0e-07],
                [0.0e-00, 0.0e-00, 0.0e-00],
                [2.0e-08, 2.0e-08, 4.0e-08],
                [1.0e-01, 1.0e-01, 2.0e-06],
            ]
        ),
        "items": array(
            [
                [0.99, 0.79, 0.90],
                [0.56, 0.28, 0.99],
                [0.30, 0.69, 0.53],
                [0.89, 0.09, 0.14],
                [0.74, 0.14, 0.53],
                [0.71, 0.13, 0.86],
            ]
        ),
        "atol": 1e-9,
        "expected_result": array([0.99, 0.79, 0.90]),
        "expect_raise": False,
    },
    {
        "description": "Some zero weights; nonzero order; -100 <= order <= 100; 2-d data; mismatching number of rows.",
        "order": 2,
        "weights": array(
            [
                [1.1e-09, 1.1e-09, 1.1e-09],
                [0.0e-00, 0.0e-00, 0.0e-00],
                [1.0e-07, 1.0e-07, 1.0e-07],
                [0.0e-00, 0.0e-00, 0.0e-00],
                [2.0e-08, 2.0e-08, 4.0e-08],
                [1.0e-06, 1.0e-06, 2.0e-06],
                [2.0e-05, 1.2e-02, 4.5e-07],
            ]
        ),
        "items": array(
            [
                [0.52, 0.73, 0.85],
                [0.56, 0.28, 0.99],
                [0.30, 0.69, 0.53],
                [0.65, 0.65, 0.65],
                [0.74, 0.14, 0.53],
                [0.71, 0.13, 0.86],
            ]
        ),
        "atol": 1e-8,
        "expected_result": None,
        "expect_raise": True,
    },
    {
        "description": "Some zero weights; nonzero order; -100 <= order <= 100; 2-d data; mismatching number of columns.",
        "order": 2,
        "weights": array(
            [
                [1.1e-09, 1.1e-09, 1.1e-09],
                [0.0e-00, 0.0e-00, 0.0e-00],
                [1.0e-07, 1.0e-07, 1.0e-07],
                [0.0e-00, 0.0e-00, 0.0e-00],
                [2.0e-08, 2.0e-08, 4.0e-08],
                [1.0e-06, 1.0e-06, 2.0e-06],
            ]
        ),
        "items": array(
            [
                [0.52, 0.73, 0.85, 1.23],
                [0.56, 0.28, 0.99, 4.56],
                [0.30, 0.69, 0.53, 7.89],
                [0.65, 0.65, 0.65, 0.12],
                [0.74, 0.14, 0.53, 3.45],
                [0.71, 0.13, 0.86, 6.78],
            ]
        ),
        "atol": 1e-8,
        "expected_result": None,
        "expect_raise": True,
    },
    {
        "description": "Some zero weights; nonzero order; -100 <= order <= 100; 1/2-d data; mismatching dimensions.",
        "order": 2,
        "weights": array(
            [
                1.1e-09,
                0.0e-00,
                1.0e-07,
                0.0e-00,
                2.0e-08,
                1.0e-06,
            ]
        ),
        "items": array(
            [
                [0.52, 0.73, 0.85],
                [0.56, 0.28, 0.99],
                [0.30, 0.69, 0.53],
                [0.65, 0.65, 0.65],
                [0.74, 0.14, 0.53],
                [0.71, 0.13, 0.86],
            ]
        ),
        "atol": 1e-8,
        "expected_result": None,
        "expect_raise": True,
    },
    {
        "description": "Some zero weights; nonzero order; -100 <= order <= 100; 1-d data; mismatching number of rows.",
        "order": 2,
        "weights": array(
            [
                1.1e-09,
                0.0e-00,
                1.0e-07,
                0.0e-00,
                2.0e-08,
                1.0e-06,
                2.0e-05,
            ]
        ),
        "items": array(
            [
                0.52,
                0.56,
                0.30,
                0.65,
                0.74,
                0.71,
            ]
        ),
        "atol": 1e-8,
        "expected_result": None,
        "expect_raise": True,
    },
    {
        "description": "Some zero weights; nonzero order; -100 <= order <= 100; >2-d.",
        "order": 2,
        "weights": array(
            [
                [[1.1e-09, 2.2e-08], [3.3e-07, 4.4e-06]],
                [[5.5e-06, 6.6e-05], [7.7e-04, 8.8e-03]],
            ]
        ),
        "items": array(
            [
                [[1.1e-09, 2.2e-08], [3.3e-07, 4.4e-06]],
                [[5.5e-06, 6.6e-05], [7.7e-04, 8.8e-03]],
            ]
        ),
        "atol": 1e-8,
        "expected_result": None,
        "expect_raise": True,
    },
]


class TestPowerMean:
    """Tests utilities.power_mean."""

    @mark.parametrize("test_case", POWER_MEAN_TEST_CASES)
    def test_power_mean(self, test_case):
        """Tests power_mean test cases."""
        if test_case["expect_raise"]:
            with raises(InvalidArgumentError):
                power_mean(
                    order=test_case["order"],
                    weights=test_case["weights"],
                    items=test_case["items"],
                    atol=test_case["atol"],
                )
        else:
            actual_result = power_mean(
                order=test_case["order"],
                weights=test_case["weights"],
                items=test_case["items"],
                atol=test_case["atol"],
            )
            assert allclose(actual_result, test_case["expected_result"])
            assert allclose(test_case["expected_result"], actual_result)
