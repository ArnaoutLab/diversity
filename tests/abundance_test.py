"""Tests for diversity.abundance."""
from dataclasses import dataclass
from numpy import allclose, array, ndarray
from pandas import DataFrame
from pytest import mark, raises

from diversity.abundance import Abundance, AbundanceFromArray, make_abundance


counts_array_3by2 = array([[2, 4], [3, 0], [0, 1]])
counts_dataframe_3by2 = DataFrame(counts_array_3by2)


@dataclass
class AbundanceExclusiveSpecies:
    description: str = "2 subcommunities; both contain exclusive species"
    counts: ndarray = counts_array_3by2
    subcommunity_abundance: ndarray = (
        array([[2 / 10, 4 / 10], [3 / 10, 0 / 10], [0 / 10, 1 / 10]]),
    )
    metacommunity_abundance: ndarray = array([[6 / 10], [3 / 10], [1 / 10]])
    subcommunity_normalizing_constants: ndarray = array([5 / 10, 5 / 10])
    normalized_subcommunity_abundance: ndarray = array(
        [[2 / 5, 4 / 5], [3 / 5, 0 / 5], [0 / 5, 1 / 5]]
    )


@dataclass
class AbundanceSingleExclusiveSpecies:
    description: str = "2 subcommunities; one contains exclusive species"
    counts: ndarray = array([[2, 4], [3, 0], [5, 1]])
    subcommunity_abundance: ndarray = (
        array([[2 / 15, 4 / 15], [3 / 15, 0 / 15], [5 / 15, 1 / 15]]),
    )
    metacommunity_abundance: ndarray = (
        array(
            [
                [6 / 15],
                [3 / 15],
                [6 / 15],
            ]
        ),
    )
    subcommunity_normalizing_constants: ndarray = array([10 / 15, 5 / 15])
    normalized_subcommunity_abundance: ndarray = array(
        [[2 / 10, 4 / 5], [3 / 10, 0 / 5], [5 / 10, 1 / 5]]
    )


@dataclass
class AbundanceNoExclusiveSpecies:
    description: str = "2 communities; neither contain exclusive species"
    counts: ndarray = array(
        [
            [2, 4],
            [3, 1],
            [1, 5],
        ],
    )
    subcommunity_abundance: ndarray = (
        array(
            [
                [2 / 16, 4 / 16],
                [3 / 16, 1 / 16],
                [1 / 16, 5 / 16],
            ]
        ),
    )
    metacommunity_abundance: ndarray = array([[6 / 16], [4 / 16], [6 / 16]])
    subcommunity_normalizing_constants: ndarray = array([6 / 16, 10 / 16])
    normalized_subcommunity_abundance: ndarray = array(
        [
            [2 / 6, 4 / 10],
            [3 / 6, 1 / 10],
            [1 / 6, 5 / 10],
        ]
    )


@dataclass
class AbundanceMutuallyExclusive:
    description: str = "2 mutually exclusive communities"
    counts: ndarray = array(
        [
            [2, 0],
            [3, 0],
            [0, 1],
            [0, 4],
        ],
    )
    subcommunity_abundance: ndarray = array(
        [
            [2 / 10, 0 / 10],
            [3 / 10, 0 / 10],
            [0 / 10, 1 / 10],
            [0 / 10, 4 / 10],
        ]
    )
    metacommunity_abundance: ndarray = array([[2 / 10], [3 / 10], [1 / 10], [4 / 10]])
    subcommunity_normalizing_constants: ndarray = array([5 / 10, 5 / 10])
    normalized_subcommunity_abundance: ndarray = array(
        [
            [2 / 5, 0 / 5],
            [3 / 5, 0 / 5],
            [0 / 5, 1 / 5],
            [0 / 5, 4 / 5],
        ]
    )


@dataclass
class AbundanceOneSubcommunity:
    description: str = "one community"
    counts = array(
        [
            [2],
            [5],
            [3],
        ],
    )
    subcommunity_abundance: ndarray = array(
        [
            [2 / 10],
            [5 / 10],
            [3 / 10],
        ]
    )
    metacommunity_abundance: ndarray = array(
        [
            [2 / 10],
            [5 / 10],
            [3 / 10],
        ]
    )
    subcommunity_normalizing_constants: ndarray = (array([10 / 10]),)
    normalized_subcommunity_abundance: ndarray = array(
        [
            [2 / 10],
            [5 / 10],
            [3 / 10],
        ]
    )


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


@mark.parametrize(
    "test_case",
    [
        AbundanceExclusiveSpecies(),
        AbundanceSingleExclusiveSpecies(),
        AbundanceNoExclusiveSpecies(),
        AbundanceMutuallyExclusive(),
        AbundanceOneSubcommunity(),
    ],
)
class TestAbundanceFromArray:
    def test_init(self, test_case):
        assert isinstance(AbundanceFromArray(counts=test_case.counts), Abundance)

    def test_subcommunity_abundance(self, test_case):
        abundance = AbundanceFromArray(counts=test_case.counts)
        assert allclose(
            abundance.subcommunity_abundance, test_case.subcommunity_abundance
        )

    def test_metacommunity_abundance(self, test_case):
        abundance = AbundanceFromArray(counts=test_case.counts)
        print(test_case.metacommunity_abundance)
        assert allclose(
            abundance.metacommunity_abundance, test_case.metacommunity_abundance
        )

    def test_subcommunity_normalizing_constants(self, test_case):
        abundance = AbundanceFromArray(counts=test_case.counts)
        assert allclose(
            abundance.subcommunity_normalizing_constants,
            test_case.subcommunity_normalizing_constants,
        )

    def test_normalized_subcommunity_abundance(self, test_case):
        abundance = AbundanceFromArray(counts=test_case.counts)
        assert allclose(
            abundance.normalized_subcommunity_abundance,
            test_case.normalized_subcommunity_abundance,
        )
