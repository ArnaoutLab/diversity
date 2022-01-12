"""Tests for diversity.metacommunity."""
from copy import deepcopy
from itertools import product

from numpy import allclose, array, empty, float64, isclose
from pytest import mark

from diversity.metacommunity import Abundance, Similarity
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
        """Tests .subcommunity_abundance."""
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
        """Tests .metacommunity_abundance."""
        abundance = self.make_abundance(test_case)
        metacommunity_abundance = abundance.metacommunity_abundance
        for i, species in enumerate(test_case["expected_species_order"]):
            assert isclose(
                metacommunity_abundance[i, 0],
                test_case["species_to_metacommunity_abundance"][species],
            ), f"\ni:{i};\nspecies: {species}."

    @mark.parametrize("test_case", ABUNDANCE_TEST_CASES)
    def test_subcommunity_normalizing_constants(self, test_case):
        """Tests .subcommunity_normalizing_constants."""
        abundance = self.make_abundance(test_case)
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
        """Tests .normalized_subcommunity_abundance."""
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
        "description": "similarities in memory; 2 communities",
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
        "species_order": array(["species_1", "species_2", "species_3"]),
        "expected_species_order": array(["species_1", "species_2", "species_3"]),
        "species_to_relative_abundances": {
            "species_1": array([1 / 1000, 1 / 100]),
            "species_2": array([1 / 10, 1 / 1]),
            "species_3": array([10, 100]),
        },
        "species_to_weighted_similarities": {
            "species_1": array([1.051, 10.51]),
            "species_2": array([2.1005, 21.005]),
            "species_3": array([10.0201, 100.201]),
        },
        "similarities_filecontents": None,
    },
    {
        "description": "similarities in memory; 1 community",
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
        "species_order": array(["species_1", "species_2", "species_3"]),
        "expected_species_order": array(["species_1", "species_2", "species_3"]),
        "species_to_relative_abundances": {
            "species_1": array([1 / 1000]),
            "species_2": array([1 / 10]),
            "species_3": array([10]),
        },
        "species_to_weighted_similarities": {
            "species_1": array([1.051]),
            "species_2": array([2.1005]),
            "species_3": array([10.0201]),
        },
        "similarities_filecontents": None,
    },
    {
        "description": "similarities in file; 2 communities",
        "similarity_matrix": None,
        "similarities_filepath": "similarities_file.tsv",
        "similarity_function": None,
        "features": None,
        "species_order": None,
        "expected_species_order": array(["species_3", "species_1", "species_2"]),
        "species_to_relative_abundances": {
            "species_1": array([1 / 1000, 1 / 100]),
            "species_2": array([1 / 10, 1 / 1]),
            "species_3": array([10, 100]),
        },
        "species_to_weighted_similarities": {
            "species_1": array([1.051, 10.51]),
            "species_2": array([2.1005, 21.005]),
            "species_3": array([10.0201, 100.201]),
        },
        "similarities_filecontents": (
            "species_3\tspecies_1\tspecies_2\n"
            "1\t0.1\t0.2\n"
            "0.1\t1\t0.5\n"
            "0.2\t0.5\t1\n"
        ),
    },
    {
        "description": "similarities in file; 1 community",
        "similarity_matrix": None,
        "similarities_filepath": "similarities_file.tsv",
        "similarity_function": None,
        "features": None,
        "species_order": None,
        "expected_species_order": array(["species_3", "species_1", "species_2"]),
        "species_to_relative_abundances": {
            "species_1": array([1 / 1000]),
            "species_2": array([1 / 10]),
            "species_3": array([10]),
        },
        "species_to_weighted_similarities": {
            "species_1": array([1.051]),
            "species_2": array([2.1005]),
            "species_3": array([10.0201]),
        },
        "similarities_filecontents": (
            "species_3\tspecies_1\tspecies_2\n"
            "1\t0.1\t0.2\n"
            "0.1\t1\t0.5\n"
            "0.2\t0.5\t1\n"
        ),
    },
    {
        "description": "similarity function; 2 communities",
        "similarity_matrix": None,
        "similarities_filepath": "similarities_file.tsv",
        "similarity_function": lambda a, b: 1 / sum(a * b),
        "features": array([[1, 2], [3, 5], [7, 11]]),
        "species_order": array(["species_3", "species_1", "species_2"]),
        "expected_species_order": array(["species_3", "species_1", "species_2"]),
        "species_to_relative_abundances": {
            "species_3": array([1 / 1000, 1 / 100]),
            "species_1": array([1 / 10, 1 / 1]),
            "species_2": array([10, 100]),
        },
        "species_to_weighted_similarities": {
            "species_3": array([0.35271989, 3.52719894]),
            "species_1": array([0.13459705, 1.34597047]),
            "species_2": array([0.0601738, 0.60173802]),
        },
        "similarities_filecontents": (
            "species_3\tspecies_1\tspecies_2\n"
            "2.00e-01\t7.69e-02\t3.45e-02\n"
            "7.69e-02\t2.94e-02\t1.32e-02\n"
            "3.45e-02\t1.32e-02\t5.88e-03\n"
        ),
    },
]


class TestSimilarity:
    """Tests diversity.metacommunity.Similarity."""

    def make_similarity(self, test_case, tmp_path):
        """Initializes test object creating similarities file, if needed."""
        if test_case["similarities_filepath"] is not None:
            absolute_path = tmp_path / test_case["similarities_filepath"]
            test_case["similarities_filepath"] = absolute_path
            if test_case["similarity_function"] is None:
                with open(absolute_path, "w") as file:
                    file.write(test_case["similarities_filecontents"])
        return Similarity(
            similarity_matrix=test_case["similarity_matrix"],
            similarities_filepath=test_case["similarities_filepath"],
            similarity_function=test_case["similarity_function"],
            features=test_case["features"],
            species_order=test_case["species_order"],
        )

    def arrange_values(self, ordered_names, name_to_row):
        """Arranges rows according to ordered_names ordering."""
        matrix = empty(
            shape=(len(ordered_names), len(name_to_row[ordered_names[0]])),
            dtype=float64,
        )
        for i, name in enumerate(ordered_names):
            matrix[i] = name_to_row[name]
        return matrix

    @mark.parametrize("test_case", SIMILARITY_TEST_CASES)
    def test_init(self, test_case, tmp_path):
        """Tests initializer."""
        similarity = self.make_similarity(test_case, tmp_path)
        assert (similarity.species_order == test_case["expected_species_order"]).all()

    @mark.parametrize("test_case", SIMILARITY_TEST_CASES)
    def test_calculate_weighted_similarities(self, test_case, tmp_path):
        """Tests .calculate_weighted_similarities."""
        similarity = self.make_similarity(test_case, tmp_path)
        relative_abundances = self.arrange_values(
            test_case["expected_species_order"],
            test_case["species_to_relative_abundances"],
        )
        weighted_similarities = similarity.calculate_weighted_similarities(
            relative_abundances
        )
        expected_weighted_similarities = self.arrange_values(
            test_case["expected_species_order"],
            test_case["species_to_weighted_similarities"],
        )
        assert weighted_similarities.shape == relative_abundances.shape
        assert allclose(weighted_similarities, expected_weighted_similarities)
        if test_case["similarities_filepath"] is not None:
            with open(test_case["similarities_filepath"], "r") as file:
                similarities_filecontents = file.read()
            assert similarities_filecontents == test_case["similarities_filecontents"]
