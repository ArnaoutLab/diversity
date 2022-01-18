"""Tests for diversity.metacommunity."""
from copy import deepcopy
from itertools import product
from warnings import filterwarnings, resetwarnings

from numpy import allclose, array, empty, float64, isclose, ndarray, unique
from pandas import DataFrame
from pytest import mark

from diversity.metacommunity import (
    Abundance,
    make_similarity,
    make_metacommunity,
    Metacommunity,
    SimilarityFromFile,
    SimilarityFromFunction,
    SimilarityFromMemory,
)
from diversity.utilities import ArgumentWarning, unique_correspondence

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


def arrange_values(ordered_names, name_to_row):
    """Arranges matrix rows according to ordered_names ordering."""
    matrix = empty(
        shape=(len(ordered_names), name_to_row[ordered_names[0]].shape[0]),
        dtype=float64,
    )
    for i, name in enumerate(ordered_names):
        matrix[i][:] = name_to_row[name]
    return matrix


SIMILARITY_FROM_FILE_TEST_CASES = [
    {
        "description": "tsv file; 2 communities",
        "similarity_matrix_filepath": "similarities_file.tsv",
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
        "description": "csv file; 2 communities",
        "similarity_matrix_filepath": "similarities_file.csv",
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
            "species_3,species_1,species_2\n" "1,0.1,0.2\n" "0.1,1,0.5\n" "0.2,0.5,1\n"
        ),
    },
    {
        "description": "no file extension; 1 community",
        "similarity_matrix_filepath": "similarities_file",
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
]


class TestSimilarityFromFile:
    """Tests diversity.metacommunity.Similarity."""

    @mark.parametrize("test_case", SIMILARITY_FROM_FILE_TEST_CASES)
    def test_init(self, test_case, tmp_path):
        """Tests initializer."""
        test_case["similarity_matrix_filepath"] = (
            tmp_path / test_case["similarity_matrix_filepath"]
        )
        with open(test_case["similarity_matrix_filepath"], "w") as file:
            file.write(test_case["similarities_filecontents"])
        if test_case["similarity_matrix_filepath"].suffix == "":
            filterwarnings("ignore", category=ArgumentWarning)
        similarity = SimilarityFromFile(
            similarity_matrix_filepath=test_case["similarity_matrix_filepath"],
        )
        resetwarnings()
        assert (similarity.species_order == test_case["expected_species_order"]).all()

    @mark.parametrize("test_case", SIMILARITY_FROM_FILE_TEST_CASES)
    def test_calculate_weighted_similarities(self, test_case, tmp_path):
        """Tests .calculate_weighted_similarities."""
        test_case["similarity_matrix_filepath"] = (
            tmp_path / test_case["similarity_matrix_filepath"]
        )
        with open(test_case["similarity_matrix_filepath"], "w") as file:
            file.write(test_case["similarities_filecontents"])
        if test_case["similarity_matrix_filepath"].suffix == "":
            filterwarnings("ignore", category=ArgumentWarning)
        similarity = SimilarityFromFile(
            similarity_matrix_filepath=test_case["similarity_matrix_filepath"],
        )
        resetwarnings()
        relative_abundances = arrange_values(
            test_case["expected_species_order"],
            test_case["species_to_relative_abundances"],
        )
        weighted_similarities = similarity.calculate_weighted_similarities(
            relative_abundances
        )
        expected_weighted_similarities = arrange_values(
            test_case["expected_species_order"],
            test_case["species_to_weighted_similarities"],
        )
        assert weighted_similarities.shape == relative_abundances.shape
        assert allclose(weighted_similarities, expected_weighted_similarities)
        with open(test_case["similarity_matrix_filepath"], "r") as file:
            similarities_filecontents = file.read()
        assert similarities_filecontents == test_case["similarities_filecontents"]


SIMILARITY_FROM_FUNCTION_TEST_CASES = [
    {
        "description": "similarity function; 2 communities; tsv similarities file",
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
        # similarity matrix
        # [0.2          , 0.07692307692, 0.03448275862]
        # [0.07692307692, 0.02941176471, 0.01315789474]
        # [0.03448275862, 0.01315789474, 0.005882352941]
    },
    {
        "description": "similarity function; 2 communities; csv similarities file",
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
        # similarity matrix
        # [0.2          , 0.07692307692, 0.03448275862]
        # [0.07692307692, 0.02941176471, 0.01315789474]
        # [0.03448275862, 0.01315789474, 0.005882352941]
    },
    {
        "description": "similarity function; 1 community; similarities file without extension",
        "similarity_function": lambda a, b: 1 / sum(a * b),
        "features": array([[1, 2], [3, 5], [7, 11]]),
        "species_order": array(["species_3", "species_1", "species_2"]),
        "expected_species_order": array(["species_3", "species_1", "species_2"]),
        "species_to_relative_abundances": {
            "species_3": array([1 / 1000]),
            "species_1": array([1 / 10]),
            "species_2": array([10]),
        },
        "species_to_weighted_similarities": {
            "species_3": array([0.35271989]),
            "species_1": array([0.13459705]),
            "species_2": array([0.0601738]),
        },
        # similarity matrix
        # [0.2          , 0.07692307692, 0.03448275862]
        # [0.07692307692, 0.02941176471, 0.01315789474]
        # [0.03448275862, 0.01315789474, 0.005882352941]
    },
]


class TestSimilarityFromFunction:
    """Tests diversity.metacommunity.Similarity."""

    @mark.parametrize("test_case", SIMILARITY_FROM_FUNCTION_TEST_CASES)
    def test_init(self, test_case, tmp_path):
        """Tests initializer."""
        similarity = SimilarityFromFunction(
            similarity_function=test_case["similarity_function"],
            features=test_case["features"],
            species_order=test_case["species_order"],
        )
        assert (similarity.species_order == test_case["expected_species_order"]).all()

    @mark.parametrize("test_case", SIMILARITY_FROM_FUNCTION_TEST_CASES)
    def test_calculate_weighted_similarities(self, test_case, tmp_path):
        """Tests .calculate_weighted_similarities."""
        similarity = SimilarityFromFunction(
            similarity_function=test_case["similarity_function"],
            features=test_case["features"],
            species_order=test_case["species_order"],
        )
        relative_abundances = arrange_values(
            test_case["expected_species_order"],
            test_case["species_to_relative_abundances"],
        )
        weighted_similarities = similarity.calculate_weighted_similarities(
            relative_abundances
        )
        expected_weighted_similarities = arrange_values(
            test_case["expected_species_order"],
            test_case["species_to_weighted_similarities"],
        )
        assert weighted_similarities.shape == relative_abundances.shape
        assert allclose(weighted_similarities, expected_weighted_similarities)


SIMILARITY_FROM_MEMORY_TEST_CASES = [
    {
        "description": "similarities in memory; 2 communities",
        "similarity_matrix": array(
            [
                [1, 0.5, 0.1],
                [0.5, 1, 0.2],
                [0.1, 0.2, 1],
            ]
        ),
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
]


class TestSimilarityFromMemory:
    """Tests diversity.metacommunity.Similarity."""

    def arrange_values(self, ordered_names, name_to_row):
        """Arranges rows according to ordered_names ordering."""
        matrix = empty(
            shape=(len(ordered_names), len(name_to_row[ordered_names[0]])),
            dtype=float64,
        )
        for i, name in enumerate(ordered_names):
            matrix[i] = name_to_row[name]
        return matrix

    @mark.parametrize("test_case", SIMILARITY_FROM_MEMORY_TEST_CASES)
    def test_init(self, test_case):
        """Tests initializer."""
        similarity = SimilarityFromMemory(
            similarity_matrix=test_case["similarity_matrix"],
            species_order=test_case["species_order"],
        )
        assert (similarity.species_order == test_case["expected_species_order"]).all()

    @mark.parametrize("test_case", SIMILARITY_FROM_MEMORY_TEST_CASES)
    def test_calculate_weighted_similarities(self, test_case, tmp_path):
        """Tests .calculate_weighted_similarities."""
        similarity = SimilarityFromMemory(
            similarity_matrix=test_case["similarity_matrix"],
            species_order=test_case["species_order"],
        )
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


CREATE_SIMILARITY_TEST_CASES = [
    {
        "description": "from memory; no additional parameters",
        "similarity_matrix": array([[1.0, 0.5, 0.1], [0.5, 1.0, 0.2], [0.1, 0.2, 1.0]]),
        "species_order": array(["species_1", "species_2", "species_3"]),
        "similarity_matrix_filepath": None,
        "similarity_function": None,
        "features": None,
        "expected_type": SimilarityFromMemory,
    },
    {
        "description": "from memory; with additional ignored parameters",
        "similarity_matrix": array([[1.0, 0.5, 0.1], [0.5, 1.0, 0.2], [0.1, 0.2, 1.0]]),
        "species_order": array(["species_1", "species_2", "species_3"]),
        "similarity_matrix_filepath": None,
        "similarity_function": None,
        "features": array(
            [[3.2, 5.3, 1.2, 9.4], [4.2, 7.4, 9.5, 7.3], [4.2, 6.2, 6.4, 7.3]]
        ),
        "expected_type": SimilarityFromMemory,
    },
    {
        "description": "from function",
        "similarity_matrix": None,
        "species_order": array(["species_1", "species_2", "species_3"]),
        "similarity_matrix_filepath": "foo_similarities.tsv",
        "similarity_function": lambda x, y: 1 / sum(x * y),
        "features": array(
            [[3.2, 5.3, 1.2, 9.4], [4.2, 7.4, 9.5, 7.3], [4.2, 6.2, 6.4, 7.3]]
        ),
        "expected_type": SimilarityFromFunction,
    },
    {
        "description": "from file; no additional parameters",
        "similarity_matrix": None,
        "species_order": None,
        "similarity_matrix_filepath": "foo_similarities.tsv",
        "similarity_function": None,
        "features": None,
        "expected_type": SimilarityFromFile,
    },
    {
        "description": "from memory; with additional ignored parameters",
        "similarity_matrix": None,
        "species_order": array(["species_1", "species_2", "species_3"]),
        "similarity_matrix_filepath": "foo_similarities.tsv",
        "similarity_function": None,
        "features": None,
        "expected_type": SimilarityFromFile,
    },
]


class TestCreateSimilarity:
    """Tests metacommunity.create_similarity."""

    @mark.parametrize("test_case", CREATE_SIMILARITY_TEST_CASES)
    def test_create_similarity(self, test_case):
        """Tests create_similarity test cases."""
        similarity_object = make_similarity(
            similarity_matrix=test_case["similarity_matrix"],
            species_order=test_case["species_order"],
            similarity_matrix_filepath=test_case["similarity_matrix_filepath"],
            similarity_function=test_case["similarity_function"],
            features=test_case["features"],
        )
        assert isinstance(similarity_object, test_case["expected_type"])


METACOMMUNITY_TEST_CASES = [
    {
        "description": "disjoint communities; uniform counts; uniform inter-community similarities; viewpoint 0.",
        "similarity": SimilarityFromMemory(
            similarity_matrix=array(
                [
                    [1.0, 0.5, 0.5, 0.7, 0.7, 0.7],
                    [0.5, 1.0, 0.5, 0.7, 0.7, 0.7],
                    [0.5, 0.5, 1.0, 0.7, 0.7, 0.7],
                    [0.7, 0.7, 0.7, 1.0, 0.5, 0.5],
                    [0.7, 0.7, 0.7, 0.5, 1.0, 0.5],
                    [0.7, 0.7, 0.7, 0.5, 0.5, 1.0],
                ]
            ),
            species_order=array(
                [
                    "species_1",
                    "species_2",
                    "species_3",
                    "species_4",
                    "species_5",
                    "species_6",
                ]
            ),
        ),
        "abundance": Abundance(
            counts=array(
                [
                    ["subcommunity_1", "species_1", "1"],
                    ["subcommunity_1", "species_2", "1"],
                    ["subcommunity_1", "species_3", "1"],
                    ["subcommunity_2", "species_4", "1"],
                    ["subcommunity_2", "species_5", "1"],
                    ["subcommunity_2", "species_6", "1"],
                ]
            ),
            species_order=array(
                [
                    "species_1",
                    "species_2",
                    "species_3",
                    "species_4",
                    "species_5",
                    "species_6",
                ]
            ),
        ),
        "viewpoint": 0,
        "subcommunity_to_alpha": {
            "subcommunity_1": 3.0,
            "subcommunity_2": 3.0,
        },
        "subcommunity_to_rho": {
            "subcommunity_1": 2.05,
            "subcommunity_2": 2.05,
        },
        "subcommunity_to_beta": {
            "subcommunity_1": 0.487805,
            "subcommunity_2": 0.487805,
        },
        "subcommunity_to_gamma": {
            "subcommunity_1": 1.463415,
            "subcommunity_2": 1.463415,
        },
        "subcommunity_to_normalized_alpha": {
            "subcommunity_1": 1.5,
            "subcommunity_2": 1.5,
        },
        "subcommunity_to_normalized_rho": {
            "subcommunity_1": 1.025,
            "subcommunity_2": 1.025,
        },
        "subcommunity_to_normalized_beta": {
            "subcommunity_1": 0.97561,
            "subcommunity_2": 0.97561,
        },
        "metacommunity_alpha": 3.0,
        "metacommunity_rho": 2.05,
        "metacommunity_beta": 0.487805,
        "metacommunity_gamma": 1.463415,
        "metacommunity_normalized_alpha": 1.5,
        "metacommunity_normalized_rho": 1.025,
        "metacommunity_normalized_beta": 0.97561,
    },
    {
        "description": "overlapping communities; non-uniform counts; non-uniform inter-community similarities; viewpoint 2.",
        "similarity": SimilarityFromMemory(
            similarity_matrix=array(
                [[1.0, 0.5, 0.1], [0.5, 1.0, 0.2], [0.1, 0.2, 1.0]]
            ),
            species_order=array(
                [
                    "species_1",
                    "species_2",
                    "species_3",
                ]
            ),
        ),
        "abundance": Abundance(
            counts=array(
                [
                    ["subcommunity_1", "species_1", "2"],
                    ["subcommunity_1", "species_2", "3"],
                    ["subcommunity_2", "species_1", "5"],
                    ["subcommunity_2", "species_3", "1"],
                ]
            ),
            species_order=array(
                [
                    "species_1",
                    "species_2",
                    "species_3",
                ]
            ),
        ),
        "viewpoint": 2,
        "subcommunity_to_alpha": {
            "subcommunity_1": 2.89473684,
            "subcommunity_2": 2.44444444,
        },
        "subcommunity_to_rho": {
            "subcommunity_1": 1.91938708,
            "subcommunity_2": 1.65870021,
        },
        "subcommunity_to_beta": {
            "subcommunity_1": 0.52099965,
            "subcommunity_2": 0.6028817,
        },
        "subcommunity_to_gamma": {
            "subcommunity_1": 1.47453083,
            "subcommunity_2": 1.45695364,
        },
        "subcommunity_to_normalized_alpha": {
            "subcommunity_1": 1.31578947,
            "subcommunity_2": 1.33333333,
        },
        "subcommunity_to_normalized_rho": {
            "subcommunity_1": 0.87244867,
            "subcommunity_2": 0.90474557,
        },
        "subcommunity_to_normalized_beta": {
            "subcommunity_1": 1.14619924,
            "subcommunity_2": 1.10528311,
        },
        "metacommunity_alpha": 2.630434782608696,
        "metacommunity_rho": 1.7678383245514573,
        "metacommunity_beta": 0.5626846957892072,
        "metacommunity_gamma": 1.4648910411622276,
        "metacommunity_normalized_alpha": 1.3253012048192772,
        "metacommunity_normalized_rho": 0.8897736389973379,
        "metacommunity_normalized_beta": 1.1235132485772303,
    },
]


class TestMetacommunity:
    """Tests metacommunity.Metacommunity."""

    def make_metacommunity(self, test_case):
        """Simple initializer that doesn't modify test_case."""
        test_case_ = deepcopy(test_case)
        return Metacommunity(
            similarity=test_case_["similarity"], abundance=test_case_["abundance"]
        )

    @mark.parametrize("test_case", METACOMMUNITY_TEST_CASES)
    def test_subcommunity_alpha(self, test_case):
        metacommunity = self.make_metacommunity(test_case)
        subcommunity_order = test_case["abundance"].subcommunity_order
        subcommunity_alpha = metacommunity.subcommunity_alpha(test_case["viewpoint"])
        for i, subcommunity in enumerate(subcommunity_order):
            assert isclose(
                subcommunity_alpha[i], test_case["subcommunity_to_alpha"][subcommunity]
            ), f"\n(i, subcommunity): {(i,subcommunity)}"

    @mark.parametrize("test_case", METACOMMUNITY_TEST_CASES)
    def test_subcommunity_rho(self, test_case):
        metacommunity = self.make_metacommunity(test_case)
        subcommunity_order = test_case["abundance"].subcommunity_order
        subcommunity_rho = metacommunity.subcommunity_rho(test_case["viewpoint"])
        for i, subcommunity in enumerate(subcommunity_order):
            assert isclose(
                subcommunity_rho[i], test_case["subcommunity_to_rho"][subcommunity]
            ), f"\n(i, subcommunity): {(i,subcommunity)}"

    @mark.parametrize("test_case", METACOMMUNITY_TEST_CASES)
    def test_subcommunity_beta(self, test_case):
        metacommunity = self.make_metacommunity(test_case)
        subcommunity_order = test_case["abundance"].subcommunity_order
        subcommunity_beta = metacommunity.subcommunity_beta(test_case["viewpoint"])
        for i, subcommunity in enumerate(subcommunity_order):
            assert isclose(
                subcommunity_beta[i], test_case["subcommunity_to_beta"][subcommunity]
            ), f"\n(i, subcommunity): {(i,subcommunity)}"

    @mark.parametrize("test_case", METACOMMUNITY_TEST_CASES)
    def test_subcommunity_gamma(self, test_case):
        metacommunity = self.make_metacommunity(test_case)
        subcommunity_order = test_case["abundance"].subcommunity_order
        subcommunity_gamma = metacommunity.subcommunity_gamma(test_case["viewpoint"])
        for i, subcommunity in enumerate(subcommunity_order):
            assert isclose(
                subcommunity_gamma[i], test_case["subcommunity_to_gamma"][subcommunity]
            ), f"\n(i, subcommunity): {(i,subcommunity)}"

    @mark.parametrize("test_case", METACOMMUNITY_TEST_CASES)
    def test_normalized_subcommunity_alpha(self, test_case):
        metacommunity = self.make_metacommunity(test_case)
        subcommunity_order = test_case["abundance"].subcommunity_order
        normalized_subcommunity_alpha = metacommunity.normalized_subcommunity_alpha(
            test_case["viewpoint"]
        )
        for i, subcommunity in enumerate(subcommunity_order):
            assert isclose(
                normalized_subcommunity_alpha[i],
                test_case["subcommunity_to_normalized_alpha"][subcommunity],
            ), f"\n(i, subcommunity): {(i,subcommunity)}"

    @mark.parametrize("test_case", METACOMMUNITY_TEST_CASES)
    def test_normalized_subcommunity_rho(self, test_case):
        metacommunity = self.make_metacommunity(test_case)
        subcommunity_order = test_case["abundance"].subcommunity_order
        normalized_subcommunity_rho = metacommunity.normalized_subcommunity_rho(
            test_case["viewpoint"]
        )
        for i, subcommunity in enumerate(subcommunity_order):
            assert isclose(
                normalized_subcommunity_rho[i],
                test_case["subcommunity_to_normalized_rho"][subcommunity],
            ), f"\n(i, subcommunity): {(i,subcommunity)}"

    @mark.parametrize("test_case", METACOMMUNITY_TEST_CASES)
    def test_normalized_subcommunity_beta(self, test_case):
        metacommunity = self.make_metacommunity(test_case)
        subcommunity_order = test_case["abundance"].subcommunity_order
        normalized_subcommunity_beta = metacommunity.normalized_subcommunity_beta(
            test_case["viewpoint"]
        )
        for i, subcommunity in enumerate(subcommunity_order):
            assert isclose(
                normalized_subcommunity_beta[i],
                test_case["subcommunity_to_normalized_beta"][subcommunity],
            ), f"\n(i, subcommunity): {(i,subcommunity)}"

    @mark.parametrize("test_case", METACOMMUNITY_TEST_CASES)
    def test_metacommunity_alpha(self, test_case):
        metacommunity = self.make_metacommunity(test_case)
        metacommunity_alpha = metacommunity.metacommunity_alpha(test_case["viewpoint"])
        assert isclose(metacommunity_alpha, test_case["metacommunity_alpha"])

    @mark.parametrize("test_case", METACOMMUNITY_TEST_CASES)
    def test_metacommunity_rho(self, test_case):
        metacommunity = self.make_metacommunity(test_case)
        metacommunity_rho = metacommunity.metacommunity_rho(test_case["viewpoint"])
        assert isclose(metacommunity_rho, test_case["metacommunity_rho"])

    @mark.parametrize("test_case", METACOMMUNITY_TEST_CASES)
    def test_metacommunity_beta(self, test_case):
        metacommunity = self.make_metacommunity(test_case)
        metacommunity_beta = metacommunity.metacommunity_beta(test_case["viewpoint"])
        assert isclose(metacommunity_beta, test_case["metacommunity_beta"])

    @mark.parametrize("test_case", METACOMMUNITY_TEST_CASES)
    def test_metacommunity_gamma(self, test_case):
        metacommunity = self.make_metacommunity(test_case)
        metacommunity_gamma = metacommunity.metacommunity_gamma(test_case["viewpoint"])
        assert isclose(metacommunity_gamma, test_case["metacommunity_gamma"])

    @mark.parametrize("test_case", METACOMMUNITY_TEST_CASES)
    def test_metacommunity_normalized_alpha(self, test_case):
        metacommunity = self.make_metacommunity(test_case)
        metacommunity_normalized_alpha = metacommunity.normalized_metacommunity_alpha(
            test_case["viewpoint"]
        )
        assert isclose(
            metacommunity_normalized_alpha, test_case["metacommunity_normalized_alpha"]
        )

    @mark.parametrize("test_case", METACOMMUNITY_TEST_CASES)
    def test_metacommunity_normalized_rho(self, test_case):
        metacommunity = self.make_metacommunity(test_case)
        metacommunity_normalized_rho = metacommunity.normalized_metacommunity_rho(
            test_case["viewpoint"]
        )
        assert isclose(
            metacommunity_normalized_rho, test_case["metacommunity_normalized_rho"]
        )

    @mark.parametrize("test_case", METACOMMUNITY_TEST_CASES)
    def test_metacommunity_normalized_beta(self, test_case):
        metacommunity = self.make_metacommunity(test_case)
        metacommunity_normalized_beta = metacommunity.normalized_metacommunity_beta(
            test_case["viewpoint"]
        )
        assert isclose(
            metacommunity_normalized_beta, test_case["metacommunity_normalized_beta"]
        )


MAKE_METACOMMUNITY_TEST_CASES = [
    {
        "description": "numpy.ndarray counts; similarities from memory; no additional parameters",
        "counts": array(
            [
                ["community_1", "species_1", "2"],
                ["community_1", "species_2", "3"],
                ["community_2", "species_1", "4"],
                ["community_2", "species_3", "1"],
            ]
        ),
        "similarity_matrix": array([[1.0, 0.5, 0.1], [0.5, 1.0, 0.2], [0.1, 0.2, 1.0]]),
        "species_order": array(["species_1", "species_2", "species_3"]),
        "similarity_matrix_filepath": None,
        "similarity_function": None,
        "features": None,
        "expected_similarity_type": SimilarityFromMemory,
    },
    {
        "description": "numpy.ndarray counts; similarities from memory; with additional ignored parameters",
        "counts": array(
            [
                ["community_1", "species_1", "2"],
                ["community_1", "species_2", "3"],
                ["community_2", "species_1", "4"],
                ["community_2", "species_3", "1"],
            ]
        ),
        "similarity_matrix": array([[1.0, 0.5, 0.1], [0.5, 1.0, 0.2], [0.1, 0.2, 1.0]]),
        "species_order": array(["species_1", "species_2", "species_3"]),
        "similarity_matrix_filepath": None,
        "similarity_function": None,
        "features": array(
            [[3.2, 5.3, 1.2, 9.4], [4.2, 7.4, 9.5, 7.3], [4.2, 6.2, 6.4, 7.3]]
        ),
        "expected_similarity_type": SimilarityFromMemory,
    },
    {
        "description": "numpy.ndarray counts; similarities from function",
        "counts": array(
            [
                ["community_1", "species_1", "2"],
                ["community_1", "species_2", "3"],
                ["community_2", "species_1", "4"],
                ["community_2", "species_3", "1"],
            ]
        ),
        "similarity_matrix": None,
        "species_order": array(["species_1", "species_2", "species_3"]),
        "similarity_matrix_filepath": "foo_similarities.tsv",
        "similarity_function": lambda x, y: 1 / sum(x * y),
        "features": array(
            [[3.2, 5.3, 1.2, 9.4], [4.2, 7.4, 9.5, 7.3], [4.2, 6.2, 6.4, 7.3]]
        ),
        "expected_similarity_type": SimilarityFromFunction,
    },
    {
        "description": "numpy.ndarray counts; similarities from file; no additional parameters",
        "counts": array(
            [
                ["community_1", "species_1", "2"],
                ["community_1", "species_2", "3"],
                ["community_2", "species_1", "4"],
                ["community_2", "species_3", "1"],
            ]
        ),
        "similarity_matrix": None,
        "species_order": None,
        "similarity_matrix_filepath": "foo_similarities.tsv",
        "similarity_function": None,
        "features": None,
        "expected_similarity_type": SimilarityFromFile,
    },
    {
        "description": "numpy.ndarray counts; similarities from memory; with additional ignored parameters",
        "counts": array(
            [
                ["community_1", "species_1", "2"],
                ["community_1", "species_2", "3"],
                ["community_2", "species_1", "4"],
                ["community_2", "species_3", "1"],
            ]
        ),
        "similarity_matrix": None,
        "species_order": array(["species_1", "species_2", "species_3"]),
        "similarity_matrix_filepath": "foo_similarities.tsv",
        "similarity_function": None,
        "features": None,
        "expected_similarity_type": SimilarityFromFile,
    },
    {
        "description": "pandas.DataFrame counts; similarities from memory; no additional parameters",
        "counts": DataFrame(
            {
                "community": [
                    "community_1",
                    "community_1",
                    "community_2",
                    "community_2",
                ],
                "species": ["species_1", "species_2", "species_1", "species_3"],
                "count": [2, 3, 4, 1],
            }
        ),
        "similarity_matrix": array([[1.0, 0.5, 0.1], [0.5, 1.0, 0.2], [0.1, 0.2, 1.0]]),
        "species_order": array(["species_1", "species_2", "species_3"]),
        "similarity_matrix_filepath": None,
        "similarity_function": None,
        "features": None,
        "expected_similarity_type": SimilarityFromMemory,
    },
    {
        "description": "pandas.DataFrame counts; similarities from memory; with additional ignored parameters",
        "counts": DataFrame(
            {
                "community": [
                    "community_1",
                    "community_1",
                    "community_2",
                    "community_2",
                ],
                "species": ["species_1", "species_2", "species_1", "species_3"],
                "count": [2, 3, 4, 1],
            }
        ),
        "similarity_matrix": array([[1.0, 0.5, 0.1], [0.5, 1.0, 0.2], [0.1, 0.2, 1.0]]),
        "species_order": array(["species_1", "species_2", "species_3"]),
        "similarity_matrix_filepath": None,
        "similarity_function": None,
        "features": array(
            [[3.2, 5.3, 1.2, 9.4], [4.2, 7.4, 9.5, 7.3], [4.2, 6.2, 6.4, 7.3]]
        ),
        "expected_similarity_type": SimilarityFromMemory,
    },
    {
        "description": "pandas.DataFrame counts; similarities from function",
        "counts": DataFrame(
            {
                "community": [
                    "community_1",
                    "community_1",
                    "community_2",
                    "community_2",
                ],
                "species": ["species_1", "species_2", "species_1", "species_3"],
                "count": [2, 3, 4, 1],
            }
        ),
        "similarity_matrix": None,
        "species_order": array(["species_1", "species_2", "species_3"]),
        "similarity_matrix_filepath": "foo_similarities.tsv",
        "similarity_function": lambda x, y: 1 / sum(x * y),
        "features": array(
            [[3.2, 5.3, 1.2, 9.4], [4.2, 7.4, 9.5, 7.3], [4.2, 6.2, 6.4, 7.3]]
        ),
        "expected_similarity_type": SimilarityFromFunction,
    },
    {
        "description": "pandas.DataFrame counts; similarities from file; no additional parameters",
        "counts": DataFrame(
            {
                "community": [
                    "community_1",
                    "community_1",
                    "community_2",
                    "community_2",
                ],
                "species": ["species_1", "species_2", "species_1", "species_3"],
                "count": [2, 3, 4, 1],
            }
        ),
        "similarity_matrix": None,
        "species_order": None,
        "similarity_matrix_filepath": "foo_similarities.tsv",
        "similarity_function": None,
        "features": None,
        "expected_similarity_type": SimilarityFromFile,
    },
    {
        "description": "pandas.DataFrame counts; similarities from memory; with additional ignored parameters",
        "counts": DataFrame(
            {
                "community": [
                    "community_1",
                    "community_1",
                    "community_2",
                    "community_2",
                ],
                "species": ["species_1", "species_2", "species_1", "species_3"],
                "count": [2, 3, 4, 1],
            }
        ),
        "similarity_matrix": None,
        "species_order": array(["species_1", "species_2", "species_3"]),
        "similarity_matrix_filepath": "foo_similarities.tsv",
        "similarity_function": None,
        "features": None,
        "expected_similarity_type": SimilarityFromFile,
    },
]


class TestMakeMetacommunity:
    """Tests metacommunity.make_metacommunity."""

    @mark.parametrize("test_case", MAKE_METACOMMUNITY_TEST_CASES)
    def test_create_similarity(self, test_case, tmp_path):
        """Tests make_metacommunity test cases."""
        if test_case["similarity_matrix_filepath"] is not None:
            test_case["similarity_matrix_filepath"] = (
                tmp_path / test_case["similarity_matrix_filepath"]
            )
        if test_case["expected_similarity_type"] == SimilarityFromFile:
            with open(test_case["similarity_matrix_filepath"], "w") as file:
                if isinstance(test_case["counts"], ndarray):
                    unique_species = unique(test_case["counts"][:, 1])
                else:
                    unique_species = unique(test_case["counts"].iloc[:, 1].to_numpy())

                file.write("\t".join(species for species in unique_species) + "\n")
        metacommunity = make_metacommunity(
            counts=test_case["counts"],
            similarity_matrix=test_case["similarity_matrix"],
            similarity_matrix_filepath=test_case["similarity_matrix_filepath"],
            similarity_function=test_case["similarity_function"],
            features=test_case["features"],
            species_order=test_case["species_order"],
        )
        assert isinstance(
            metacommunity.similarity, test_case["expected_similarity_type"]
        )
