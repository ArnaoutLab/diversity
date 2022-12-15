"""Tests for diversity.abundance."""
from numpy import allclose, array, array_equal, dtype
from pandas import DataFrame
from pytest import mark, raises

from diversity.abundance import AbundanceFromArray, make_abundance


class MockClass:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class MockAbundanceFromArray(MockClass):
    pass


counts_array_3by2 = array([[2, 4], [3, 0], [0, 1]])
counts_dataframe_3by2 = DataFrame(counts_array_3by2)


@mark.parametrize(
    "counts, expected",
    [
        (counts_array_3by2, DataFrame),
        (counts_dataframe_3by2, DataFrame),
    ],
)
def test_make_abundance(counts, expected):
    abundance = AbundanceFromArray(counts=counts)
    assert isinstance(abundance.counts, expected)


def test_make_abundance_not_implemented():
    with raises(NotImplementedError):
        make_abundance(counts=1)


ABUNDANCE_TEST_CASES = [
    {
        "description": "default parameters; 2 communities; both contain exclusive species",
        "counts": array([[2, 4], [3, 0], [0, 1]], dtype=dtype("f8")),
        "subcommunity_abundance": array(
            [[2 / 10, 4 / 10], [3 / 10, 0 / 10], [0 / 10, 1 / 10]]
        ),
        "metacommunity_abundance": array([[6 / 10], [3 / 10], [1 / 10]]),
        "subcommunity_normalizing_constants": array([5 / 10, 5 / 10]),
        "normalized_subcommunity_abundance": array(
            [[2 / 5, 4 / 5], [3 / 5, 0 / 5], [0 / 5, 1 / 5]]
        ),
    },
    {
        "description": "default parameters; 2 communities; one contains exclusive species",
        "counts": array([[2, 4], [3, 0], [5, 1]], dtype=dtype("f8")),
        "subcommunity_abundance": array(
            [[2 / 15, 4 / 15], [3 / 15, 0 / 15], [5 / 15, 1 / 15]]
        ),
        "metacommunity_abundance": array(
            [
                [6 / 15],
                [3 / 15],
                [6 / 15],
            ]
        ),
        "subcommunity_normalizing_constants": array([10 / 15, 5 / 15]),
        "normalized_subcommunity_abundance": array(
            [[2 / 10, 4 / 5], [3 / 10, 0 / 5], [5 / 10, 1 / 5]]
        ),
    },
    {
        "description": "default parameters; 2 communities; neither contain exclusive species",
        "counts": array(
            [
                [2, 4],
                [3, 1],
                [1, 5],
            ],
            dtype=dtype("f8"),
        ),
        "subcommunity_abundance": array(
            [
                [2 / 16, 4 / 16],
                [3 / 16, 1 / 16],
                [1 / 16, 5 / 16],
            ]
        ),
        "metacommunity_abundance": array([[6 / 16], [4 / 16], [6 / 16]]),
        "subcommunity_normalizing_constants": array([6 / 16, 10 / 16]),
        "normalized_subcommunity_abundance": array(
            [
                [2 / 6, 4 / 10],
                [3 / 6, 1 / 10],
                [1 / 6, 5 / 10],
            ]
        ),
    },
    {
        "description": "default parameters; 2 mutually exclusive communities",
        "counts": array(
            [
                [2, 0],
                [3, 0],
                [0, 1],
                [0, 4],
            ],
            dtype=dtype("f8"),
        ),
        "subcommunity_abundance": array(
            [
                [2 / 10, 0 / 10],
                [3 / 10, 0 / 10],
                [0 / 10, 1 / 10],
                [0 / 10, 4 / 10],
            ]
        ),
        "metacommunity_abundance": array([[2 / 10], [3 / 10], [1 / 10], [4 / 10]]),
        "subcommunity_normalizing_constants": array([5 / 10, 5 / 10]),
        "normalized_subcommunity_abundance": array(
            [
                [2 / 5, 0 / 5],
                [3 / 5, 0 / 5],
                [0 / 5, 1 / 5],
                [0 / 5, 4 / 5],
            ]
        ),
    },
    {
        "description": "default parameters; 1 community",
        "counts": array(
            [
                [2],
                [5],
                [3],
            ],
            dtype=dtype("f8"),
        ),
        "subcommunity_abundance": array(
            [
                [2 / 10],
                [5 / 10],
                [3 / 10],
            ]
        ),
        "metacommunity_abundance": array(
            [
                [2 / 10],
                [5 / 10],
                [3 / 10],
            ]
        ),
        "subcommunity_normalizing_constants": array([10 / 10]),
        "normalized_subcommunity_abundance": array(
            [
                [2 / 10],
                [5 / 10],
                [3 / 10],
            ]
        ),
    },
]


class TestAbundanceFromArray:
    """Tests metacommunity.AbundanceFromArray."""

    @mark.parametrize("test_case", ABUNDANCE_TEST_CASES)
    def test_init(self, test_case):
        """Tests initializer."""
        abundance = AbundanceFromArray(counts=test_case["counts"])
        array_equal(abundance.counts, test_case["counts"])

    @mark.parametrize("test_case", ABUNDANCE_TEST_CASES)
    def test_subcommunity_abundance(self, test_case):
        """Tests .subcommunity_abundance."""
        abundance = AbundanceFromArray(counts=test_case["counts"])
        subcommunity_abundance = abundance.subcommunity_abundance()
        assert allclose(subcommunity_abundance, test_case["subcommunity_abundance"])

    @mark.parametrize("test_case", ABUNDANCE_TEST_CASES)
    def test_metacommunity_abundance(self, test_case):
        """Tests .metacommunity_abundance."""
        abundance = AbundanceFromArray(counts=test_case["counts"])
        metacommunity_abundance = abundance.metacommunity_abundance()
        assert allclose(metacommunity_abundance, test_case["metacommunity_abundance"])

    @mark.parametrize("test_case", ABUNDANCE_TEST_CASES)
    def test_subcommunity_normalizing_constants(self, test_case):
        """Tests .subcommunity_normalizing_constants."""
        abundance = AbundanceFromArray(counts=test_case["counts"])
        subcommunity_normalizing_constants = (
            abundance.subcommunity_normalizing_constants()
        )
        assert allclose(
            subcommunity_normalizing_constants,
            test_case["subcommunity_normalizing_constants"],
        )

    @mark.parametrize("test_case", ABUNDANCE_TEST_CASES)
    def test_normalized_subcommunity_abundance(self, test_case):
        """Tests .normalized_subcommunity_abundance."""
        abundance = AbundanceFromArray(counts=test_case["counts"])
        normalized_subcommunity_abundance = (
            abundance.normalized_subcommunity_abundance()
        )
        assert allclose(
            normalized_subcommunity_abundance,
            test_case["normalized_subcommunity_abundance"],
        )
