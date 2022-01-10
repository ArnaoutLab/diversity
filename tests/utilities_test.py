"""Tests for diversity.utilities."""
from copy import deepcopy

from numpy import allclose, array, inf
from pytest import mark

from diversity.utilities import power_mean, unique_correspondence


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
        "expected_result": array([0.32955284, 0.06034367, 0.39917668]),
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
        "expected_result": array([0.07100004, 0.01300007, 0.08600006]),
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
        "expected_result": array([0.35246927, 0.13302809, 0.62156143]),
    },
    {
        "description": "Some zero weights; nonzero order; -100 <= order <= 100; 2-d data.",
        "order": 2,
        "weights": array(
            [
                [1.0e-09, 1.0e-09, 1.0e-09],
                [0.0e00, 0.0e00, 0.0e00],
                [1.0e-07, 1.0e-07, 1.0e-07],
                [0.0e00, 0.0e00, 0.0e00],
                [2.0e-08, 2.0e-08, 2.0e-08],
                [1.0e-01, 1.0e-01, 1.0e-01],
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
        "expected_result": array([0.22452176, 0.04111019, 0.27195594]),
    },
    {
        "description": "All zero weights; nonzero order; -100 <= order <= 100; 2-d data.",
        "order": 2,
        "weights": array(
            [
                [1.0e-09, 1.0e-09, 1.0e-09],
                [0.0e00, 0.0e00, 0.0e00],
                [1.0e-10, 1.0e-10, 1.0e-10],
                [0.0e00, 0.0e00, 0.0e00],
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
        "expected_result": array([0.0, 0.0, 0.0]),
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
        "expected_result": array([0.96633071, 0.8154443, 0.9850308]),
    },
    {
        "description": "Some zero weights; zero order; 2-d data.",
        "order": 0,
        "weights": array(
            [
                [1.0e-09, 1.0e-09, 1.0e-09],
                [0.0e00, 0.0e00, 0.0e00],
                [1.0e-07, 1.0e-07, 1.0e-07],
                [0.0e00, 0.0e00, 0.0e00],
                [2.0e-08, 2.0e-08, 2.0e-08],
                [1.0e-01, 1.0e-01, 1.0e-01],
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
        "expected_result": array([0.96633071, 0.8154443, 0.9850308]),
    },
    {
        "description": "All zero weights; zero order; 2-d data.",
        "order": 0,
        "weights": array(
            [
                [1.0e-09, 1.0e-09, 1.0e-09],
                [0.0e00, 0.0e00, 0.0e00],
                [1.0e-10, 1.0e-10, 1.0e-10],
                [0.0e00, 0.0e00, 0.0e00],
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
        "expected_result": array([0.0, 0.0, 0.0]),
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
        "expected_result": array([0.30, 0.13, 0.53]),
    },
    {
        "description": "Some zero weights; order == -inf; 2-d data.",
        "order": -inf,
        "weights": array(
            [
                [1.0e-09, 1.0e-09, 1.0e-09],
                [0.0e00, 0.0e00, 0.0e00],
                [1.0e-07, 1.0e-07, 1.0e-07],
                [0.0e00, 0.0e00, 0.0e00],
                [2.0e-08, 2.0e-08, 2.0e-08],
                [1.0e-01, 1.0e-01, 1.0e-01],
            ]
        ),
        "items": array(
            [
                [0.52, 0.73, 0.85],
                [0.56, 0.28, 0.99],
                [0.30, 0.69, 0.53],
                [0.28, 0.09, 0.14],
                [0.74, 0.14, 0.53],
                [0.71, 0.13, 0.86],
            ]
        ),
        "expected_result": array([0.30, 0.13, 0.53]),
    },
    {
        "description": "All zero weights; order < -100; 2-d data.",
        "order": -382,
        "weights": array(
            [
                [1.0e-09, 1.0e-09, 1.0e-09],
                [0.0e00, 0.0e00, 0.0e00],
                [1.0e-10, 1.0e-10, 1.0e-10],
                [0.0e00, 0.0e00, 0.0e00],
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
        "expected_result": array([0.0, 0.0, 0.0]),
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
        "expected_result": array([0.74, 0.69, 0.86]),
    },
    {
        "description": "Some zero weights; order == inf; 2-d data.",
        "order": inf,
        "weights": array(
            [
                [1.0e-09, 1.0e-09, 1.0e-09],
                [0.0e00, 0.0e00, 0.0e00],
                [1.0e-07, 1.0e-07, 1.0e-07],
                [0.0e00, 0.0e00, 0.0e00],
                [2.0e-08, 2.0e-08, 2.0e-08],
                [1.0e-01, 1.0e-01, 1.0e-01],
            ]
        ),
        "items": array(
            [
                [0.52, 0.73, 0.85],
                [0.56, 0.28, 0.99],
                [0.30, 0.69, 0.53],
                [0.89, 0.09, 0.14],
                [0.74, 0.14, 0.53],
                [0.71, 0.13, 0.86],
            ]
        ),
        "expected_result": array([0.74, 0.69, 0.86]),
    },
    {
        "description": "All zero weights; order > 100; 2-d data.",
        "order": 364,
        "weights": array(
            [
                [1.0e-09, 1.0e-09, 1.0e-09],
                [0.0e00, 0.0e00, 0.0e00],
                [1.0e-10, 1.0e-10, 1.0e-10],
                [0.0e00, 0.0e00, 0.0e00],
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
        "expected_result": array([0.0, 0.0, 0.0]),
    },
    {
        "description": "Some zero weights; nonzero order; -100 <= order <= 100; 1-d data",
        "order": 2,
        "weights": array(
            [
                1.0e-09,
                0.0e00,
                1.0e-07,
                0.0e00,
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
        "expected_result": array([0.22452176]),
    },
    {
        "description": "Some zero weights; zero order; 1-d data.",
        "order": 0,
        "weights": array(
            [
                1.0e-09,
                0.0e00,
                1.0e-07,
                0.0e00,
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
        "expected_result": array([0.96633071]),
    },
    {
        "description": "Some zero weights; order == -inf; 1-d data.",
        "order": -inf,
        "weights": array(
            [
                1.0e-09,
                0.0e00,
                1.0e-07,
                0.0e00,
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
        "expected_result": array([0.30]),
    },
    {
        "description": "Some zero weights; order == inf; 1-d data.",
        "order": inf,
        "weights": array(
            [
                1.0e-09,
                0.0e00,
                1.0e-07,
                0.0e00,
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
        "expected_result": array([0.74]),
    },
]


class TestPowerMean:
    """Tests metacommunity.utilities.power_mean."""

    @mark.parametrize("test_case", POWER_MEAN_TEST_CASES)
    def test_power_mean(self, test_case):
        """Tests power_mean test cases."""
        actual_result = power_mean(
            order=test_case["order"],
            weights=test_case["weights"],
            items=test_case["items"],
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
    """Tests metacommunity.utilities.unique_correspondence."""

    @mark.parametrize("test_case", UNIQUE_CORRESPONDENCE_TEST_CASES)
    def test_unique_correspondence(self, test_case):
        """Tests unique_correspondence test cases."""
        original_items = deepcopy(test_case["items"])
        original_ordered_unique_items = deepcopy(
            test_case["ordered_unique_items"])
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
