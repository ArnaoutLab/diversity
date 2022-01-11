"""Tests for diversity.metacommunity."""
from copy import deepcopy
from itertools import product

from numpy import array, empty, float64, isclose
from pytest import mark

from diversity.metacommunity import Abundance
from diversity.utilities import unique_correspondence

ABUNDANCE_TEST_CASES = [
    ####################################################################
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
        "subcommunity_species_to_subcommunity_abundance": {
            ("community_1", "species_1"): 2 / 10,
            ("community_1", "species_2"): 3 / 10,
            ("community_1", "species_3"): 0 / 10,
            ("community_2", "species_1"): 4 / 10,
            ("community_2", "species_2"): 0 / 10,
            ("community_2", "species_3"): 1 / 10,
        },
        "species_to_metacommunity_abundance": {
            "species_1": 6 / 10,
            "species_2": 3 / 10,
            "species_3": 1 / 10,
        },
        "subcommunity_to_normalizing_constants": {
            "community_1": 5 / 10,
            "community_2": 5 / 10,
        },
        "subcommunity_species_to_normalized_subcommunity_abundance": {
            ("community_1", "species_1"): 2 / 5,
            ("community_1", "species_2"): 3 / 5,
            ("community_1", "species_3"): 0 / 5,
            ("community_2", "species_1"): 4 / 5,
            ("community_2", "species_2"): 0 / 5,
            ("community_2", "species_3"): 1 / 5,
        },
    },
    ####################################################################
    {
        "description": "default parameters; 2 communities; one contains exclusive species",
        "counts": array(
            [
                ["community_1", "species_1", "2"],
                ["community_1", "species_2", "3"],
                ["community_2", "species_1", "4"],
                ["community_2", "species_3", "1"],
                ["community_1", "species_3", "5"],
            ]
        ),
        "species_order": None,
        "subcommunity_order": None,
        "subcommunity_column": 0,
        "species_column": 1,
        "count_column": 2,
        "expected_species_order": unique_correspondence(
            array(["species_1", "species_2", "species_1", "species_3", "species_3"])
        )[0],
        "expected_subcommunity_order": unique_correspondence(
            array(
                [
                    "community_1",
                    "community_1",
                    "community_2",
                    "community_2",
                    "community_1",
                ]
            )
        )[0],
        "subcommunity_species_to_subcommunity_abundance": {
            ("community_1", "species_1"): 2 / 15,
            ("community_1", "species_2"): 3 / 15,
            ("community_1", "species_3"): 5 / 15,
            ("community_2", "species_1"): 4 / 15,
            ("community_2", "species_2"): 0 / 15,
            ("community_2", "species_3"): 1 / 15,
        },
        "species_to_metacommunity_abundance": {
            "species_1": 6 / 15,
            "species_2": 3 / 15,
            "species_3": 6 / 15,
        },
        "subcommunity_to_normalizing_constants": {
            "community_1": 10 / 15,
            "community_2": 5 / 15,
        },
        "subcommunity_species_to_normalized_subcommunity_abundance": {
            ("community_1", "species_1"): 2 / 10,
            ("community_1", "species_2"): 3 / 10,
            ("community_1", "species_3"): 5 / 10,
            ("community_2", "species_1"): 4 / 5,
            ("community_2", "species_2"): 0 / 5,
            ("community_2", "species_3"): 1 / 5,
        },
    },
    ####################################################################
    {
        "description": "default parameters; 2 communities; neither contain exclusive species",
        "counts": array(
            [
                ["community_1", "species_1", "2"],
                ["community_1", "species_2", "3"],
                ["community_2", "species_1", "4"],
                ["community_2", "species_3", "1"],
                ["community_1", "species_3", "5"],
                ["community_2", "species_2", "1"],
            ]
        ),
        "species_order": None,
        "subcommunity_order": None,
        "subcommunity_column": 0,
        "species_column": 1,
        "count_column": 2,
        "expected_species_order": unique_correspondence(
            array(
                [
                    "species_1",
                    "species_2",
                    "species_1",
                    "species_3",
                    "species_3",
                    "species_2",
                ]
            )
        )[0],
        "expected_subcommunity_order": unique_correspondence(
            array(
                [
                    "community_1",
                    "community_1",
                    "community_2",
                    "community_2",
                    "community_1",
                    "community_2",
                ]
            )
        )[0],
        "subcommunity_species_to_subcommunity_abundance": {
            ("community_1", "species_1"): 2 / 16,
            ("community_1", "species_2"): 3 / 16,
            ("community_1", "species_3"): 5 / 16,
            ("community_2", "species_1"): 4 / 16,
            ("community_2", "species_2"): 1 / 16,
            ("community_2", "species_3"): 1 / 16,
        },
        "species_to_metacommunity_abundance": {
            "species_1": 6 / 16,
            "species_2": 4 / 16,
            "species_3": 6 / 16,
        },
        "subcommunity_to_normalizing_constants": {
            "community_1": 10 / 16,
            "community_2": 6 / 16,
        },
        "subcommunity_species_to_normalized_subcommunity_abundance": {
            ("community_1", "species_1"): 2 / 10,
            ("community_1", "species_2"): 3 / 10,
            ("community_1", "species_3"): 5 / 10,
            ("community_2", "species_1"): 4 / 6,
            ("community_2", "species_2"): 1 / 6,
            ("community_2", "species_3"): 1 / 6,
        },
    },
    ####################################################################
    {
        "description": "default parameters; 2 mutually exclusive communities",
        "counts": array(
            [
                ["community_1", "species_1", "2"],
                ["community_1", "species_2", "3"],
                ["community_2", "species_4", "4"],
                ["community_2", "species_3", "1"],
            ]
        ),
        "species_order": None,
        "subcommunity_order": None,
        "subcommunity_column": 0,
        "species_column": 1,
        "count_column": 2,
        "expected_species_order": unique_correspondence(
            array(["species_1", "species_2", "species_4", "species_3"])
        )[0],
        "expected_subcommunity_order": unique_correspondence(
            array(
                [
                    "community_1",
                    "community_1",
                    "community_2",
                    "community_2",
                ]
            )
        )[0],
        "subcommunity_species_to_subcommunity_abundance": {
            ("community_1", "species_1"): 2 / 10,
            ("community_1", "species_2"): 3 / 10,
            ("community_1", "species_3"): 0 / 10,
            ("community_1", "species_4"): 0 / 10,
            ("community_2", "species_1"): 0 / 10,
            ("community_2", "species_2"): 0 / 10,
            ("community_2", "species_3"): 1 / 10,
            ("community_2", "species_4"): 4 / 10,
        },
        "species_to_metacommunity_abundance": {
            "species_1": 2 / 10,
            "species_2": 3 / 10,
            "species_3": 1 / 10,
            "species_4": 4 / 10,
        },
        "subcommunity_to_normalizing_constants": {
            "community_1": 5 / 10,
            "community_2": 5 / 10,
        },
        "subcommunity_species_to_normalized_subcommunity_abundance": {
            ("community_1", "species_1"): 2 / 5,
            ("community_1", "species_2"): 3 / 5,
            ("community_1", "species_3"): 0 / 5,
            ("community_1", "species_4"): 0 / 5,
            ("community_2", "species_1"): 0 / 5,
            ("community_2", "species_2"): 0 / 5,
            ("community_2", "species_3"): 1 / 5,
            ("community_2", "species_4"): 4 / 5,
        },
    },
    ####################################################################
    {
        "description": "default parameters; 1 community",
        "counts": array(
            [
                ["community_1", "species_1", "2"],
                ["community_1", "species_3", "3"],
                ["community_1", "species_2", "5"],
            ]
        ),
        "species_order": None,
        "subcommunity_order": None,
        "subcommunity_column": 0,
        "species_column": 1,
        "count_column": 2,
        "expected_species_order": unique_correspondence(
            array(["species_1", "species_3", "species_2"])
        )[0],
        "expected_subcommunity_order": unique_correspondence(
            array(
                [
                    "community_1",
                    "community_1",
                    "community_1",
                ]
            )
        )[0],
        "subcommunity_species_to_subcommunity_abundance": {
            ("community_1", "species_1"): 2 / 10,
            ("community_1", "species_2"): 5 / 10,
            ("community_1", "species_3"): 3 / 10,
        },
        "species_to_metacommunity_abundance": {
            "species_1": 2 / 10,
            "species_2": 5 / 10,
            "species_3": 3 / 10,
        },
        "subcommunity_to_normalizing_constants": {
            "community_1": 10 / 10,
        },
        "subcommunity_species_to_normalized_subcommunity_abundance": {
            ("community_1", "species_1"): 2 / 10,
            ("community_1", "species_2"): 5 / 10,
            ("community_1", "species_3"): 3 / 10,
        },
    },
    ####################################################################
    {
        "description": "nondefault species_order; 2 communities; both contain exclusive species",
        "counts": array(
            [
                ["community_1", "species_1", "2"],
                ["community_1", "species_2", "3"],
                ["community_2", "species_1", "4"],
                ["community_2", "species_3", "1"],
            ]
        ),
        "species_order": array(["species_2", "species_1", "species_3"]),
        "subcommunity_order": None,
        "subcommunity_column": 0,
        "species_column": 1,
        "count_column": 2,
        "expected_species_order": array(["species_2", "species_1", "species_3"]),
        "expected_subcommunity_order": unique_correspondence(
            array(["community_1", "community_1", "community_2", "community_2"])
        )[0],
        "subcommunity_species_to_subcommunity_abundance": {
            ("community_1", "species_1"): 2 / 10,
            ("community_1", "species_2"): 3 / 10,
            ("community_1", "species_3"): 0 / 10,
            ("community_2", "species_1"): 4 / 10,
            ("community_2", "species_2"): 0 / 10,
            ("community_2", "species_3"): 1 / 10,
        },
        "species_to_metacommunity_abundance": {
            "species_1": 6 / 10,
            "species_2": 3 / 10,
            "species_3": 1 / 10,
        },
        "subcommunity_to_normalizing_constants": {
            "community_1": 5 / 10,
            "community_2": 5 / 10,
        },
        "subcommunity_species_to_normalized_subcommunity_abundance": {
            ("community_1", "species_1"): 2 / 5,
            ("community_1", "species_2"): 3 / 5,
            ("community_1", "species_3"): 0 / 5,
            ("community_2", "species_1"): 4 / 5,
            ("community_2", "species_2"): 0 / 5,
            ("community_2", "species_3"): 1 / 5,
        },
    },
    ####################################################################
    {
        "description": "nondefault subcommunity_order; 3 communities; all contain exclusive species",
        "counts": array(
            [
                ["community_1", "species_2", "3"],
                ["community_2", "species_1", "4"],
                ["community_1", "species_1", "2"],
                ["community_2", "species_3", "1"],
                ["community_3", "species_1", "1"],
                ["community_3", "species_4", "1"],
            ]
        ),
        "species_order": None,
        "subcommunity_order": array(["community_2", "community_1", "community_3"]),
        "subcommunity_column": 0,
        "species_column": 1,
        "count_column": 2,
        "expected_species_order": unique_correspondence(
            array(
                [
                    "species_1",
                    "species_2",
                    "species_1",
                    "species_3",
                    "species_1",
                    "species_4",
                ]
            )
        )[0],
        "expected_subcommunity_order": array(
            ["community_2", "community_1", "community_3"]
        ),
        "subcommunity_species_to_subcommunity_abundance": {
            ("community_1", "species_1"): 2 / 12,
            ("community_1", "species_2"): 3 / 12,
            ("community_1", "species_3"): 0 / 12,
            ("community_1", "species_4"): 0 / 12,
            ("community_2", "species_1"): 4 / 12,
            ("community_2", "species_2"): 0 / 12,
            ("community_2", "species_3"): 1 / 12,
            ("community_2", "species_4"): 0 / 12,
            ("community_3", "species_1"): 1 / 12,
            ("community_3", "species_2"): 0 / 12,
            ("community_3", "species_3"): 0 / 12,
            ("community_3", "species_4"): 1 / 12,
        },
        "species_to_metacommunity_abundance": {
            "species_1": 7 / 12,
            "species_2": 3 / 12,
            "species_3": 1 / 12,
            "species_4": 1 / 12,
        },
        "subcommunity_to_normalizing_constants": {
            "community_1": 5 / 12,
            "community_2": 5 / 12,
            "community_3": 2 / 12,
        },
        "subcommunity_species_to_normalized_subcommunity_abundance": {
            ("community_1", "species_1"): 2 / 5,
            ("community_1", "species_2"): 3 / 5,
            ("community_1", "species_3"): 0 / 5,
            ("community_1", "species_4"): 0 / 5,
            ("community_2", "species_1"): 4 / 5,
            ("community_2", "species_2"): 0 / 5,
            ("community_2", "species_3"): 1 / 5,
            ("community_2", "species_4"): 0 / 5,
            ("community_3", "species_1"): 1 / 2,
            ("community_3", "species_2"): 0 / 2,
            ("community_3", "species_3"): 0 / 2,
            ("community_3", "species_4"): 1 / 2,
        },
    },
    ####################################################################
    {
        "description": "permuted columns; 2 communities; both contain exclusive species",
        "counts": array(
            [
                ["2", "community_1", "species_1"],
                ["3", "community_1", "species_2"],
                ["4", "community_2", "species_1"],
                ["1", "community_2", "species_3"],
            ]
        ),
        "species_order": None,
        "subcommunity_order": None,
        "subcommunity_column": 1,
        "species_column": 2,
        "count_column": 0,
        "expected_species_order": unique_correspondence(
            array(["species_1", "species_2", "species_1", "species_3"])
        )[0],
        "expected_subcommunity_order": unique_correspondence(
            array(["community_1", "community_1", "community_2", "community_2"])
        )[0],
        "subcommunity_species_to_subcommunity_abundance": {
            ("community_1", "species_1"): 2 / 10,
            ("community_1", "species_2"): 3 / 10,
            ("community_1", "species_3"): 0 / 10,
            ("community_2", "species_1"): 4 / 10,
            ("community_2", "species_2"): 0 / 10,
            ("community_2", "species_3"): 1 / 10,
        },
        "species_to_metacommunity_abundance": {
            "species_1": 6 / 10,
            "species_2": 3 / 10,
            "species_3": 1 / 10,
        },
        "subcommunity_to_normalizing_constants": {
            "community_1": 5 / 10,
            "community_2": 5 / 10,
        },
        "subcommunity_species_to_normalized_subcommunity_abundance": {
            ("community_1", "species_1"): 2 / 5,
            ("community_1", "species_2"): 3 / 5,
            ("community_1", "species_3"): 0 / 5,
            ("community_2", "species_1"): 4 / 5,
            ("community_2", "species_2"): 0 / 5,
            ("community_2", "species_3"): 1 / 5,
        },
    },
]


class TestAbundance:
    """Tests diversity.metacommunity.Abundance."""

    def make_abundance(self, test_case):
        """Simple initializer without modifying test_case."""
        test_case_ = deepcopy(test_case)
        return Abundance(
            counts=test_case_["counts"],
            species_order=test_case_["species_order"],
            subcommunity_order=test_case_["subcommunity_order"],
            subcommunity_column=test_case_["subcommunity_column"],
            species_column=test_case_["species_column"],
            count_column=test_case_["count_column"],
        )

    @mark.parametrize("test_case", ABUNDANCE_TEST_CASES)
    def test_init(self, test_case):
        """Tests initializer."""
        abundance = self.make_abundance(test_case)
        assert (abundance.species_order == test_case["expected_species_order"]).all()
        assert (
            abundance.subcommunity_order == test_case["expected_subcommunity_order"]
        ).all()

    @mark.parametrize("test_case", ABUNDANCE_TEST_CASES)
    def test_subcommunity_abundance(self, test_case):
        abundance = self.make_abundance(test_case)
        subcommunity_abundance = abundance.subcommunity_abundance
        for (i, species), (j, subcommunity) in product(
            enumerate(test_case["expected_species_order"]),
            enumerate(test_case["expected_subcommunity_order"]),
        ):
            assert isclose(
                subcommunity_abundance[i, j],
                test_case["subcommunity_species_to_subcommunity_abundance"][
                    (subcommunity, species)
                ],
            ), (
                f"\n(i,j): {(i,j)};"
                f"\n(species,subcommunity): {(species,subcommunity)}."
            )

    @mark.parametrize("test_case", ABUNDANCE_TEST_CASES)
    def test_metacommunity_abundance(self, test_case):
        abundance = self.make_abundance(test_case)
        metacommunity_abundance = abundance.metacommunity_abundance
        for i, species in enumerate(test_case["expected_species_order"]):
            assert isclose(
                metacommunity_abundance[i, 0],
                test_case["species_to_metacommunity_abundance"][species],
            ), f"\ni:{i};\nspecies: {species}."

    @mark.parametrize("test_case", ABUNDANCE_TEST_CASES)
    def test_subcommunity_normalizing_constants(self, test_case):
        abundance = self.make_abundance(test_case)  ### HERE
        subcommunity_normalizing_constants = (
            abundance.subcommunity_normalizing_constants
        )
        for j, subcommunity in enumerate(test_case["expected_subcommunity_order"]):
            assert isclose(
                subcommunity_normalizing_constants[j],
                test_case["subcommunity_to_normalizing_constants"][subcommunity],
            )

    @mark.parametrize("test_case", ABUNDANCE_TEST_CASES)
    def test_normalized_subcommunity_abundance(self, test_case):
        abundance = self.make_abundance(test_case)
        normalized_subcommunity_abundance = abundance.normalized_subcommunity_abundance
        for (i, species), (j, subcommunity) in product(
            enumerate(test_case["expected_species_order"]),
            enumerate(test_case["expected_subcommunity_order"]),
        ):
            assert isclose(
                normalized_subcommunity_abundance[i, j],
                test_case["subcommunity_species_to_normalized_subcommunity_abundance"][
                    (subcommunity, species)
                ],
            ), (
                f"\n(i,j): {(i,j)};"
                f"\n(species,subcommunity): {(species,subcommunity)};"
            )


SIMILARITY_TEST_CASES = [
    {
        "description": "similarities in memory; 2 communities; default species order",
        "counts": array(
            [
                ["community_1", "species_1", "2"],
                ["community_1", "species_2", "3"],
                ["community_2", "species_1", "4"],
                ["community_2", "species_3", "1"],
            ]
        ),
        "similarity_matrix": array(
            [
                [1, 0.5, 0.1],
                [0.5, 1, 0.2],
                [0.1, 0.2, 1],
            ]
        ),
        "similarities_filepath": None,
        "similarity_function": None,
        "features": None,
        "species_order": None,
        "expected_species_order": unique_correspondence(
            array(["species_1", "species_2", "species_1", "species_3"])
        )[0],
        "species_to_metacommunity_similarity": array([[0.76], [0.62], [0.22]]),
    }
]


class SimilarityTest:
    """Tests diversity.metacommunity.Similarity."""

    @mark.parametrize("test_case", SIMILARITY_TEST_CASES)
    def test_init(self, test_case):
        
