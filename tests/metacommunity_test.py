"""Tests for diversity.metacommunity."""
from numpy import array, empty, float64
from pytest import mark

from diversity.metacommunity import Abundance
from diversity.utilities import unique_correspondence

ABUNDANCE_TEST_CASES = [
    {
        "description": "default parameters; 2 communities; both contain exclusive species",
        "counts": array(
            [
                ["community_1", "species_1", "2"],
                ["community_1", "species_2", "3"],
                ["community_2", "species_1", "4"],
                ["community_2", "species_3", "1"],
            ]
        ),
        "species_order": None,
        "subcommunity_order": None,
        "subcommunity_column": 0,
        "species_column": 1,
        "count_column": 2,
        "expected_species_order": unique_correspondence(
            array(["species_1", "species_2", "species_1", "species_3"])
        )[0],
        "expected_subcommunity_order": unique_correspondence(
            array(["community_1", "community_1", "community_2", "community_2"])
        )[0],
        "subcommunity_species_to_abundance": {
            ("community_1", "species_1"): 2 / 10,
            ("community_1", "species_2"): 3 / 10,
            ("community_1", "species_3"): 0 / 10,
            ("community_2", "species_1"): 4 / 10,
            ("community_2", "species_2"): 0 / 10,
            ("community_2", "species_3"): 1 / 10,
        },
    }
]


class TestAbundance:
    """Tests diversity.metacommunity.Abundance."""

    def arrange_matrix(
        self, subcommunity_order, species_order, subcommunity_species_to_abundance
    ):
        abundances = empty(
            shape=(len(species_order), len(subcommunity_order)), dtype=float64
        )
        for (
            subcommunity,
            species,
        ), abundance in subcommunity_species_to_abundance.items():
            i = species_order.index(species)
            j = subcommunity_order.index(subcommunity)
            abundances[i, j] = abundance
        return abundances

    @mark.parametrize("test_case", ABUNDANCE_TEST_CASES)
    def test_init(self, test_case):
        abundance = Abundance(
            counts=test_case["counts"],
            species_order=test_case["species_order"],
            subcommunity_order=test_case["subcommunity_order"],
            subcommunity_column=test_case["subcommunity_column"],
            species_column=test_case["species_column"],
            count_column=test_case["count_column"],
        )
        assert (abundance.species_order == test_case["expected_species_order"]).all()
        assert (
            abundance.subcommunity_order == test_case["expected_subcommunity_order"]
        ).all()
