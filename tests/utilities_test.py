"""Tests for diversity.utilities."""
from copy import deepcopy

from numpy import allclose, array, dtype, empty, inf, ones
from pandas import DataFrame, Index
from pandas.testing import assert_frame_equal
from pytest import mark, raises, warns

from diversity.exceptions import ArgumentWarning, InvalidArgumentError
from diversity.log import LOGGER
from diversity.shared import SharedArraySpec, SharedArrayView
from diversity.utilities import (
    get_file_delimiter,
    partition_range,
    pivot_table,
    power_mean,
    subset_by_column,
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


PIVOT_TABLE_TEST_CASES = [
    {
        "description": "All combinations present",
        "data_frame": DataFrame(
            {
                "pivot_col": ["a", "a", "b", "b", "c", "c"],
                "index_col": [1, 2, 1, 2, 1, 2],
                "value_col1": [1010, 2010, 1020, 2020, 1030, 2030],
                "value_col2": [11, 21, 12, 22, 13, 23],
            }
        ),
        "pivot_column": "pivot_col",
        "index_column": "index_col",
        "value_columns": ["value_col1", "value_col2"],
        "pivot_ordering": None,
        "index_ordering": None,
        "out": None,
        "pivotted_table": array(
            [[1010, 11, 1020, 12, 1030, 13], [2010, 21, 2020, 22, 2030, 23]],
            dtype=dtype("f8"),
        ),
    },
    {
        "description": "Some combinations missing",
        "data_frame": DataFrame(
            {
                "pivot_col": ["a", "b", "c", "c"],
                "index_col": [2, 1, 1, 2],
                "value_col1": [2010, 1020, 1030, 2030],
                "value_col2": [21, 12, 13, 23],
            }
        ),
        "pivot_column": "pivot_col",
        "index_column": "index_col",
        "value_columns": ["value_col1", "value_col2"],
        "pivot_ordering": None,
        "index_ordering": None,
        "out": None,
        "pivotted_table": array(
            [[0, 0, 1020, 12, 1030, 13], [2010, 21, 0, 0, 2030, 23]],
            dtype=dtype("f8"),
        ),
    },
    {
        "description": "Noise column",
        "data_frame": DataFrame(
            {
                "pivot_col": ["a", "b", "c", "c"],
                "index_col": [2, 1, 1, 2],
                "value_col1": [2010, 1020, 1030, 2030],
                "value_col2": [21, 12, 13, 23],
                "noise_col": ["zip", "zap", "zippidi", "zap"],
            }
        ),
        "pivot_column": "pivot_col",
        "index_column": "index_col",
        "value_columns": ["value_col1", "value_col2"],
        "pivot_ordering": None,
        "index_ordering": None,
        "out": None,
        "pivotted_table": array(
            [[0, 0, 1020, 12, 1030, 13], [2010, 21, 0, 0, 2030, 23]],
            dtype=dtype("f8"),
        ),
    },
    {
        "description": "Single value column",
        "data_frame": DataFrame(
            {
                "pivot_col": ["a", "b", "b", "c", "c"],
                "index_col": [1, 1, 2, 1, 2],
                "value_col": [1010, 1020, 2020, 1030, 2030],
            }
        ),
        "pivot_column": "pivot_col",
        "index_column": "index_col",
        "value_columns": ["value_col"],
        "pivot_ordering": None,
        "index_ordering": None,
        "out": None,
        "pivotted_table": array(
            [[1010, 1020, 1030], [0, 2020, 2030]],
            dtype=dtype("f8"),
        ),
    },
    {
        "description": "Non-default pivot_ordering",
        "data_frame": DataFrame(
            {
                "pivot_col": ["a", "b", "c", "c"],
                "index_col": [2, 1, 1, 2],
                "value_col1": [2010, 1020, 1030, 2030],
                "value_col2": [21, 12, 13, 23],
            }
        ),
        "pivot_column": "pivot_col",
        "index_column": "index_col",
        "value_columns": ["value_col1", "value_col2"],
        "pivot_ordering": array(["b", "a", "c"]),
        "index_ordering": None,
        "out": None,
        "pivotted_table": array(
            [[1020, 12, 0, 0, 1030, 13], [0, 0, 2010, 21, 2030, 23]],
            dtype=dtype("f8"),
        ),
    },
    {
        "description": "Non-default index_ordering",
        "data_frame": DataFrame(
            {
                "pivot_col": ["a", "b", "c", "c"],
                "index_col": [2, 1, 1, 2],
                "value_col1": [2010, 1020, 1030, 2030],
                "value_col2": [21, 12, 13, 23],
            }
        ),
        "pivot_column": "pivot_col",
        "index_column": "index_col",
        "value_columns": ["value_col1", "value_col2"],
        "pivot_ordering": None,
        "index_ordering": array([2, 1]),
        "out": None,
        "pivotted_table": array(
            [[2010, 21, 0, 0, 2030, 23], [0, 0, 1020, 12, 1030, 13]],
            dtype=dtype("f8"),
        ),
    },
    {
        "description": "Non-default pivot_ordering and index_ordering",
        "data_frame": DataFrame(
            {
                "pivot_col": ["a", "b", "c", "c"],
                "index_col": [2, 1, 1, 2],
                "value_col1": [2010, 1020, 1030, 2030],
                "value_col2": [21, 12, 13, 23],
            }
        ),
        "pivot_column": "pivot_col",
        "index_column": "index_col",
        "value_columns": ["value_col1", "value_col2"],
        "pivot_ordering": array(["c", "a", "b"]),
        "index_ordering": array([2, 1]),
        "out": None,
        "pivotted_table": array(
            [[2030, 23, 2010, 21, 0, 0], [1030, 13, 0, 0, 1020, 12]],
            dtype=dtype("f8"),
        ),
    },
    {
        "description": "Some combinations missing; numpy array out argument",
        "data_frame": DataFrame(
            {
                "pivot_col": ["a", "b", "c", "c"],
                "index_col": [2, 1, 1, 2],
                "value_col1": [2010, 1020, 1030, 2030],
                "value_col2": [21, 12, 13, 23],
            }
        ),
        "pivot_column": "pivot_col",
        "index_column": "index_col",
        "value_columns": ["value_col1", "value_col2"],
        "pivot_ordering": None,
        "index_ordering": None,
        "out": ones(shape=(2, 6), dtype=int),
        "pivotted_table": array(
            [[0, 0, 1020, 12, 1030, 13], [2010, 21, 0, 0, 2030, 23]],
            dtype=dtype("f8"),
        ),
    },
    {
        "description": "Some combinations missing; shared array out argument",
        "data_frame": DataFrame(
            {
                "pivot_col": ["a", "b", "c", "c"],
                "index_col": [2, 1, 1, 2],
                "value_col1": [2010, 1020, 1030, 2030],
                "value_col2": [21, 12, 13, 23],
            }
        ),
        "pivot_column": "pivot_col",
        "index_column": "index_col",
        "value_columns": ["value_col1", "value_col2"],
        "pivot_ordering": None,
        "index_ordering": None,
        "out": SharedArrayView(
            spec=SharedArraySpec(
                name="fake_name",
                shape=(2, 6),
                dtype=dtype("f8"),
            ),
            memory_view=ones(shape=(2, 6), dtype=dtype("i8")).data,
        ),
        "pivotted_table": array(
            [[0, 0, 1020, 12, 1030, 13], [2010, 21, 0, 0, 2030, 23]],
            dtype=dtype("f8"),
        ),
    },
]


class TestPivotTable:
    """Tests utilities.pivot_table."""

    @mark.parametrize("test_case", PIVOT_TABLE_TEST_CASES)
    def test_pivot_table(self, test_case):
        """Tests pivot_table."""
        pivotted_table = pivot_table(
            data_frame=test_case["data_frame"],
            pivot_column=test_case["pivot_column"],
            index_column=test_case["index_column"],
            value_columns=test_case["value_columns"],
            pivot_ordering=test_case["pivot_ordering"],
            index_ordering=test_case["index_ordering"],
            out=test_case["out"],
        )
        assert pivotted_table.shape == test_case["pivotted_table"].shape
        assert allclose(pivotted_table, test_case["pivotted_table"])
        if test_case["out"] is not None:
            if isinstance(test_case["out"], SharedArrayView):
                assert pivotted_table is test_case["out"].data
            else:
                assert pivotted_table is test_case["out"]


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


SUBSET_BY_COLUMN_TEST_CASES = [
    {
        "description": "proper nonempty subset",
        "data_frame": DataFrame(
            {"a": [1, 2, 3, 2, 4, 3, 1], "b": [3, 2, 7, 1, 4, 5, 6]},
            index=list(range(7)),
        ),
        "column": "a",
        "subset": [2, 4],
        "subsetted_data_frame": DataFrame(
            {"a": [2, 2, 4], "b": [2, 1, 4]}, index=[1, 3, 4]
        ),
    },
    {
        "description": "improper subset",
        "data_frame": DataFrame(
            {"a": [1, 2, 3, 2, 4, 3, 1], "b": [3, 2, 7, 1, 4, 5, 6]},
            index=list(range(7)),
        ),
        "column": "a",
        "subset": {1, 2, 3, 4},
        "subsetted_data_frame": DataFrame(
            {"a": [1, 2, 3, 2, 4, 3, 1], "b": [3, 2, 7, 1, 4, 5, 6]},
            index=list(range(7)),
        ),
    },
    {
        "description": "empty subset",
        "data_frame": DataFrame(
            {"a": [1, 2, 3, 2, 4, 3, 1], "b": [3, 2, 7, 1, 4, 5, 6]},
            index=list(range(7)),
        ),
        "column": "a",
        "subset": set(),
        "subsetted_data_frame": DataFrame(
            {"a": [], "b": []},
            index=Index([], dtype=dtype("i8")),
            dtype=dtype("i8"),
        ),
    },
    {
        "description": "subset=None",
        "data_frame": DataFrame(
            {"a": [1, 2, 3, 2, 4, 3, 1], "b": [3, 2, 7, 1, 4, 5, 6]},
            index=list(range(7)),
        ),
        "column": "a",
        "subset": None,
        "subsetted_data_frame": DataFrame(
            {"a": [1, 2, 3, 2, 4, 3, 1], "b": [3, 2, 7, 1, 4, 5, 6]},
            index=list(range(7)),
        ),
    },
    {
        "description": "proper nonempty subset for string column",
        "data_frame": DataFrame(
            {"a": ["a", "b", "c", "b", "d", "c", "a"], "b": [3, 2, 7, 1, 4, 5, 6]},
            index=list(range(7)),
        ),
        "column": "a",
        "subset": ["b", "d"],
        "subsetted_data_frame": DataFrame(
            {"a": ["b", "b", "d"], "b": [2, 1, 4]}, index=[1, 3, 4]
        ),
    },
]


class TestSubsetByColumn:
    """Tests utilities.subset_by_column."""

    @mark.parametrize("test_case", SUBSET_BY_COLUMN_TEST_CASES)
    def test_subset_by_column(self, test_case):
        """Tests subset_by_column test cases."""
        subsetted_data_frame = subset_by_column(
            data_frame=test_case["data_frame"],
            column=test_case["column"],
            subset=test_case["subset"],
        )
        assert_frame_equal(subsetted_data_frame, test_case["subsetted_data_frame"])


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
