"""Tests for diversity.utilities."""
from copy import deepcopy

from numpy import all as numpy_all, allclose, array, dtype, inf, ndarray
from pytest import mark, raises, warns

from diversity.utilities import (
    ArgumentWarning,
    get_file_delimiter,
    InvalidArgumentError,
    partition_range,
    power_mean,
    SharedArray,
    SharedArraySpec,
    SharedArrayView,
    unique_correspondence,
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


PARTITION_RANGE_TEST_CASES = [
    {
        "description": "Evenly divisible range.",
        "range_": range(10),
        "num_chunks": 5,
        "expected_ranges": [
            range(0, 2),
            range(2, 4),
            range(4, 6),
            range(6, 8),
            range(8, 10),
        ],
        "expect_raise": False,
    },
    {
        "description": "Unevenly divisible range.",
        "range_": range(10),
        "num_chunks": 3,
        "expected_ranges": [
            range(0, 3),
            range(3, 6),
            range(6, 10),
        ],
        "expect_raise": False,
    },
    {
        "description": "More chunks than range.",
        "range_": range(5),
        "num_chunks": 6,
        "expected_ranges": [
            range(0, 0),
            range(0, 1),
            range(1, 2),
            range(2, 3),
            range(3, 4),
            range(4, 5),
        ],
        "expect_raise": False,
    },
    {
        "description": "0 chunks, non-empty range.",
        "range_": range(10),
        "num_chunks": 0,
        "expected_ranges": None,
        "expect_raise": True,
    },
    {
        "description": "0 chunks, empty range.",
        "range_": range(0, 0),
        "num_chunks": 0,
        "expected_ranges": None,
        "expect_raise": True,
    },
    {
        "description": "Negative chunks, non-empty range.",
        "range_": range(10),
        "num_chunks": -1,
        "expected_ranges": None,
        "expect_raise": True,
    },
    {
        "description": "Negative chunks, empty range.",
        "range_": range(5, 5),
        "num_chunks": -13,
        "expected_ranges": None,
        "expect_raise": True,
    },
    {
        "description": "Non-zero chunks, empty range.",
        "range_": range(10, 10),
        "num_chunks": 3,
        "expected_ranges": [range(10, 10), range(10, 10), range(10, 10)],
        "expect_raise": False,
    },
]


class TestPartitionRange:
    """Tests utilities.partition_range."""

    @mark.parametrize("test_case", PARTITION_RANGE_TEST_CASES)
    def test_partition_range(self, test_case):
        """Tests partition_range."""
        if test_case["expect_raise"]:
            with raises(InvalidArgumentError):
                partition_range(test_case["range_"], test_case["num_chunks"])
        else:
            ranges = partition_range(test_case["range_"], test_case["num_chunks"])
            assert ranges == test_case["expected_ranges"]


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


UNIQUE_CORRESPONDENCE_TEST_CASES = [
    {
        "description": "No ordering imposed; non-empty non-unique items",
        "items": array([1, 2, 1, 5, 3, 2, 4, 100, 3, 4]),
        "ordered_unique_items": None,
        "n_unique": 6,
    },
    {
        "description": "No ordering imposed; non-empty unique items",
        "items": array(["foo", "bar", "zip", "zap", "zongo", "wakka"]),
        "ordered_unique_items": None,
        "n_unique": 6,
    },
    {
        "description": "No ordering imposed; empty items",
        "items": array([]),
        "ordered_unique_items": None,
        "n_unique": 0,
    },
    {
        "description": "Ordering consists of uniques only; non-empty non-unique items",
        "items": array(["foo", "bar", "fooo", "zip", "bar", "foo", "foo"]),
        "ordered_unique_items": array(["zip", "fooo", "bar", "foo"]),
        "n_unique": 4,
    },
    {
        "description": "Ordering consists of uniques only; non-empty unique items",
        "items": array([124, 64, 12, 32]),
        "ordered_unique_items": array([64, 12, 32, 124]),
        "n_unique": 4,
    },
    {
        "description": "Ordering consists of uniques only; empty items",
        "items": array([]),
        "ordered_unique_items": array([]),
        "n_unique": 0,
    },
    {
        "description": "Ordering properly conains uniques; non-empty non-unique items",
        "items": array(["foo", "bar", "fooo", "zip", "bar", "foo", "foo"]),
        "ordered_unique_items": array(
            ["zip", "shazam", "foo", "bar", "bazinga", "fooo"]
        ),
        "n_unique": 4,
    },
    {
        "description": "Ordering properly conains uniques; non-empty unique items",
        "items": array(["foo", "bar", "fooo", "zip"]),
        "ordered_unique_items": array(
            ["zip", "shazam", "foo", "bar", "bazinga", "fooo"]
        ),
        "n_unique": 4,
    },
    {
        "description": "Ordering properly conains uniques; empty items",
        "items": array([]),
        "ordered_unique_items": array(
            ["zip", "shazam", "foo", "bar", "bazinga", "fooo"]
        ),
        "n_unique": 0,
    },
]


class TestUniqueCorrespondence:
    """Tests utilities.unique_correspondence."""

    @mark.parametrize("test_case", UNIQUE_CORRESPONDENCE_TEST_CASES)
    def test_unique_correspondence(self, test_case):
        """Tests unique_correspondence test cases."""
        original_items = deepcopy(test_case["items"])
        original_ordered_unique_items = deepcopy(test_case["ordered_unique_items"])
        result_unique_items, result_item_positions = unique_correspondence(
            items=test_case["items"],
            ordered_unique_items=test_case["ordered_unique_items"],
        )
        assert len(set(result_unique_items)) == len(result_unique_items)
        assert set(original_items).issubset(set(result_unique_items))
        for unique_pos, item in zip(result_item_positions, original_items):
            assert item == result_unique_items[unique_pos]
        if original_ordered_unique_items is not None:
            assert all(result_unique_items == original_ordered_unique_items)


SHARED_ARRAY_TEST_CASES = [
    {
        "description": "1-d character array.",
        "shape": (10,),
        "dtype": dtype("<U1"),
        "data": array(["f", "r", "u", "i", "t", "l", "o", "o", "p", "s"]),
        "modify_coordinates": (4,),
        "original_value": "t",
        "modified_value": "x",
    },
    {
        "description": "3-d integer array.",
        "shape": (2, 3, 2),
        "dtype": dtype("i8"),
        "data": array([[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]]),
        "modify_coordinates": (1, 1, 1),
        "original_value": 10,
        "modified_value": -100,
    },
]


class TestSharedArray:
    """Tests utilities.SharedArray."""

    @mark.parametrize("test_case", SHARED_ARRAY_TEST_CASES)
    def test_from_array(self, test_case):
        """Tests .from_array factory."""
        shared_array = SharedArray.from_array(test_case["data"])
        assert numpy_all(shared_array.data == test_case["data"])
        del shared_array

    @mark.parametrize("test_case", SHARED_ARRAY_TEST_CASES)
    def test_init(self, test_case):
        """Tests .__init__."""
        shared_array = SharedArray(test_case["shape"], test_case["dtype"])
        shared_array.data[:] = test_case["data"]
        assert numpy_all(shared_array.data == test_case["data"])
        del shared_array

    @mark.parametrize("test_case", SHARED_ARRAY_TEST_CASES)
    def test_modify(self, test_case):
        """Tests that the shared memory points to exact same data."""
        shared_array = SharedArray(test_case["shape"], test_case["dtype"])
        shared_array.data[:] = test_case["data"]
        data = ndarray(
            shape=test_case["shape"],
            dtype=test_case["dtype"],
            buffer=shared_array.shared_memory.buf,
        )
        assert data[test_case["modify_coordinates"]] == test_case["original_value"]
        assert (
            shared_array.data[test_case["modify_coordinates"]]
            == test_case["original_value"]
        )
        data[test_case["modify_coordinates"]] = test_case["modified_value"]
        assert data[test_case["modify_coordinates"]] == test_case["modified_value"]
        assert (
            shared_array.data[test_case["modify_coordinates"]]
            == test_case["modified_value"]
        )


SHARED_ARRAY_VIEW_TEST_CASES = [
    {
        "description": "1-d character array.",
        "shape": (10,),
        "dtype": dtype("<U1"),
        "shared_array": SharedArray.from_array(
            array(["f", "r", "u", "i", "t", "l", "o", "o", "p", "s"])
        ),
        "modify_coordinates": (4,),
        "original_value": "t",
        "modified_value": "x",
    },
    {
        "description": "3-d integer array.",
        "shape": (2, 3, 2),
        "dtype": dtype("i8"),
        "shared_array": SharedArray.from_array(
            array([[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]])
        ),
        "modify_coordinates": (1, 1, 1),
        "original_value": 10,
        "modified_value": -100,
    },
]


class TestSharedArrayView:
    """Tests utilities.SharedArrayView."""

    @mark.parametrize("test_case", SHARED_ARRAY_VIEW_TEST_CASES)
    def test_init(self, test_case):
        """Tests .__init__."""
        shared_array_view = SharedArrayView(
            SharedArraySpec.from_shared_array(test_case["shared_array"])
        )
        assert shared_array_view.name == test_case["shared_array"].name
        assert shared_array_view.shape == test_case["shared_array"].shape
        assert shared_array_view.dtype == test_case["shared_array"].dtype
        assert numpy_all(shared_array_view.data == test_case["shared_array"].data)

    @mark.parametrize("test_case", SHARED_ARRAY_VIEW_TEST_CASES)
    def test_modify(self, test_case):
        """Tests that the shared memory points to exact same data."""
        shared_array_view = SharedArrayView(
            SharedArraySpec.from_shared_array(test_case["shared_array"])
        )
        data = ndarray(
            shape=test_case["shape"],
            dtype=test_case["dtype"],
            buffer=test_case["shared_array"].shared_memory.buf,
        )
        assert data[test_case["modify_coordinates"]] == test_case["original_value"]
        assert (
            shared_array_view.data[test_case["modify_coordinates"]]
            == test_case["original_value"]
        )
        assert (
            test_case["shared_array"].data[test_case["modify_coordinates"]]
            == test_case["original_value"]
        )
        data[test_case["modify_coordinates"]] = test_case["modified_value"]
        assert data[test_case["modify_coordinates"]] == test_case["modified_value"]
        assert (
            shared_array_view.data[test_case["modify_coordinates"]]
            == test_case["modified_value"]
        )
        assert (
            test_case["shared_array"].data[test_case["modify_coordinates"]]
            == test_case["modified_value"]
        )
