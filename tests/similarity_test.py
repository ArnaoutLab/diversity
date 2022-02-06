"""Tests for diversity.similarity."""
from multiprocessing import cpu_count

from numpy import allclose, array, array_equal, dtype, empty, zeros
from pandas import DataFrame
from pandas.testing import assert_frame_equal
from pytest import fixture, mark, warns

from diversity.exceptions import ArgumentWarning
from diversity.similarity import (
    SimilarityFromFile,
    SimilarityFromFunction,
    SimilarityFromMemory,
)


def arrange_values(ordered_names, name_to_row):
    """Arranges matrix rows according to ordered_names ordering."""
    matrix = empty(
        shape=(len(ordered_names), name_to_row[ordered_names[0]].shape[0]),
        dtype=dtype("f8"),
    )
    for i, name in enumerate(ordered_names):
        matrix[i][:] = name_to_row[name]
    return matrix


SIMILARITY_FROM_FILE_TEST_CASES = [
    {
        "description": "tsv file; 2 communities",
        "similarity_matrix_filepath": "similarities_file.tsv",
        "species_subset": ["species_3", "species_1", "species_2"],
        "chunk_size": 1,
        "expected_species_ordering": array(["species_3", "species_1", "species_2"]),
        "relative_abundances": array([[1 / 1000, 1 / 100], [1 / 10, 1 / 1], [10, 100]]),
        "weighted_similarities": array(
            [[2.011, 20.11], [5.1001, 51.001], [10.0502, 100.502]]
        ),
        "similarities_filecontents": (
            "species_3\tspecies_1\tspecies_2\n"
            "1\t0.1\t0.2\n"
            "0.1\t1\t0.5\n"
            "0.2\t0.5\t1\n"
        ),
        "expect_warning": False,
    },
    {
        "description": "csv file; 2 communities",
        "similarity_matrix_filepath": "similarities_file.csv",
        "species_subset": ["species_3", "species_1", "species_2"],
        "chunk_size": 1,
        "expected_species_ordering": array(["species_3", "species_1", "species_2"]),
        "relative_abundances": array([[1 / 1000, 1 / 100], [1 / 10, 1 / 1], [10, 100]]),
        "weighted_similarities": array(
            [[2.011, 20.11], [5.1001, 51.001], [10.0502, 100.502]]
        ),
        "similarities_filecontents": (
            "species_3,species_1,species_2\n" "1,0.1,0.2\n" "0.1,1,0.5\n" "0.2,0.5,1\n"
        ),
        "expect_warning": False,
    },
    {
        "description": "no file extension; 1 community",
        "similarity_matrix_filepath": "similarities_file",
        "species_subset": ["species_3", "species_1", "species_2"],
        "chunk_size": 1,
        "expected_species_ordering": array(["species_3", "species_1", "species_2"]),
        "relative_abundances": array([[1 / 1000], [1 / 10], [10]]),
        "weighted_similarities": array([[2.011], [5.1001], [10.0502]]),
        "similarities_filecontents": (
            "species_3\tspecies_1\tspecies_2\n"
            "1\t0.1\t0.2\n"
            "0.1\t1\t0.5\n"
            "0.2\t0.5\t1\n"
        ),
        "expect_warning": True,
    },
    {
        "description": "species subset",
        "similarity_matrix_filepath": "similarities_file.tsv",
        "species_subset": ["species_1", "species_3"],
        "chunk_size": 1,
        "expected_species_ordering": array(["species_3", "species_1"]),
        "relative_abundances": array([[1 / 1000, 1 / 100], [10, 100]]),
        "weighted_similarities": array([[1.001, 10.01], [10.0001, 100.001]]),
        "similarities_filecontents": (
            "species_3\tspecies_1\tspecies_2\n"
            "1\t0.1\t0.2\n"
            "0.1\t1\t0.5\n"
            "0.2\t0.5\t1\n"
        ),
        "expect_warning": False,
    },
    {
        "description": "non-default chunk_size",
        "similarity_matrix_filepath": "similarities_file.tsv",
        "species_subset": ["species_3", "species_1", "species_2"],
        "chunk_size": 12,
        "expected_species_ordering": array(["species_3", "species_1", "species_2"]),
        "relative_abundances": array([[1 / 1000, 1 / 100], [1 / 10, 1 / 1], [10, 100]]),
        "weighted_similarities": array(
            [[2.011, 20.11], [5.1001, 51.001], [10.0502, 100.502]]
        ),
        "similarities_filecontents": (
            "species_3\tspecies_1\tspecies_2\n"
            "1\t0.1\t0.2\n"
            "0.1\t1\t0.5\n"
            "0.2\t0.5\t1\n"
        ),
        "expect_warning": False,
    },
]


class TestSimilarityFromFile:
    """Tests metacommunity.Similarity."""

    @fixture(params=SIMILARITY_FROM_FILE_TEST_CASES)
    def test_case(self, request, tmp_path):
        filepath = f"{tmp_path}/{request.param['similarity_matrix_filepath']}"
        with open(filepath, "w") as file:
            file.write(request.param["similarities_filecontents"])
        test_case_ = {
            key: request.param[key]
            for key in [
                "description",
                "species_subset",
                "chunk_size",
                "expected_species_ordering",
                "relative_abundances",
                "weighted_similarities",
                "similarities_filecontents",
                "expect_warning",
            ]
        }
        test_case_["similarity_matrix"] = filepath
        return test_case_

    def test_init(self, test_case):
        """Tests initializer."""
        if test_case["expect_warning"]:
            with warns(ArgumentWarning):
                similarity = SimilarityFromFile(
                    similarity_matrix=test_case["similarity_matrix"],
                    species_subset=test_case["species_subset"],
                    chunk_size=test_case["chunk_size"],
                )
        else:
            similarity = SimilarityFromFile(
                similarity_matrix=test_case["similarity_matrix"],
                species_subset=test_case["species_subset"],
                chunk_size=test_case["chunk_size"],
            )
        assert array_equal(
            similarity.species_ordering, test_case["expected_species_ordering"]
        )
        assert similarity.similarity_matrix == test_case["similarity_matrix"]
        assert similarity.chunk_size == test_case["chunk_size"]

    def test_calculate_weighted_similarities(self, test_case):
        """Tests .calculate_weighted_similarities."""
        if test_case["expect_warning"]:
            with warns(ArgumentWarning):
                similarity = SimilarityFromFile(
                    similarity_matrix=test_case["similarity_matrix"],
                    species_subset=test_case["species_subset"],
                    chunk_size=test_case["chunk_size"],
                )
        else:
            similarity = SimilarityFromFile(
                similarity_matrix=test_case["similarity_matrix"],
                species_subset=test_case["species_subset"],
                chunk_size=test_case["chunk_size"],
            )
        weighted_similarities = similarity.calculate_weighted_similarities(
            test_case["relative_abundances"]
        )
        assert weighted_similarities.shape == test_case["weighted_similarities"].shape
        assert allclose(weighted_similarities, test_case["weighted_similarities"])
        with open(test_case["similarity_matrix"], "r") as file:
            similarities_filecontents = file.read()
        assert similarities_filecontents == test_case["similarities_filecontents"]


def sim_func(a, b):
    return 1 / sum(a * b)


SIMILARITY_FROM_FUNCTION_TEST_CASES = [
    {
        "description": "2 communities; 2 features; default num_processors",
        "similarity_function": sim_func,  # lambda a, b: 1 / sum(a * b)
        "features": array([[1, 2], [3, 5], [7, 11]]),
        "species_ordering": array(["species_1", "species_2", "species_3"]),
        "num_processors": None,
        "expected_species_ordering": array(["species_1", "species_2", "species_3"]),
        "relative_abundances": array([[1 / 1000, 1 / 100], [1 / 10, 1 / 1], [10, 100]]),
        "expected_weighted_similarities": array(
            [
                [0.35271989, 3.52719894],
                [0.13459705, 1.34597047],
                [0.0601738, 0.60173802],
            ]
        ),
        # similarity matrix
        # [0.2          , 0.07692307692, 0.03448275862]
        # [0.07692307692, 0.02941176471, 0.01315789474]
        # [0.03448275862, 0.01315789474, 0.005882352941]
    },
    {
        "description": "2 communities; 2 features; num_processors=1",
        "similarity_function": sim_func,  # lambda a, b: 1 / sum(a * b)
        "features": array([[1, 2], [3, 5], [7, 11]]),
        "species_ordering": array(["species_1", "species_2", "species_3"]),
        "num_processors": 1,
        "expected_species_ordering": array(["species_1", "species_2", "species_3"]),
        "relative_abundances": array([[1 / 1000, 1 / 100], [1 / 10, 1 / 1], [10, 100]]),
        "expected_weighted_similarities": array(
            [
                [0.35271989, 3.52719894],
                [0.13459705, 1.34597047],
                [0.0601738, 0.60173802],
            ]
        ),
        # similarity matrix
        # [0.2          , 0.07692307692, 0.03448275862]
        # [0.07692307692, 0.02941176471, 0.01315789474]
        # [0.03448275862, 0.01315789474, 0.005882352941]
    },
    {
        "description": "2 communities; 2 features; num_processors=2",
        "similarity_function": sim_func,  # lambda a, b: 1 / sum(a * b)
        "features": array([[1, 2], [3, 5], [7, 11]]),
        "species_ordering": array(["species_1", "species_2", "species_3"]),
        "num_processors": 2,
        "expected_species_ordering": array(["species_1", "species_2", "species_3"]),
        "relative_abundances": array([[1 / 1000, 1 / 100], [1 / 10, 1 / 1], [10, 100]]),
        "expected_weighted_similarities": array(
            [
                [0.35271989, 3.52719894],
                [0.13459705, 1.34597047],
                [0.0601738, 0.60173802],
            ]
        ),
        # similarity matrix
        # [0.2          , 0.07692307692, 0.03448275862]
        # [0.07692307692, 0.02941176471, 0.01315789474]
        # [0.03448275862, 0.01315789474, 0.005882352941]
    },
    {
        "description": "2 communities; 2 features; num_processors>num_species",
        "similarity_function": sim_func,  # lambda a, b: 1 / sum(a * b)
        "features": array([[1, 2], [3, 5], [7, 11]]),
        "species_ordering": array(["species_1", "species_2", "species_3"]),
        "num_processors": 4,
        "expected_species_ordering": array(["species_1", "species_2", "species_3"]),
        "relative_abundances": array([[1 / 1000, 1 / 100], [1 / 10, 1 / 1], [10, 100]]),
        "expected_weighted_similarities": array(
            [
                [0.35271989, 3.52719894],
                [0.13459705, 1.34597047],
                [0.0601738, 0.60173802],
            ]
        ),
        # similarity matrix
        # [0.2          , 0.07692307692, 0.03448275862]
        # [0.07692307692, 0.02941176471, 0.01315789474]
        # [0.03448275862, 0.01315789474, 0.005882352941]
    },
    {
        "description": "2 communities; 2 features; default num_processors>cpu_count",
        "similarity_function": sim_func,  # lambda a, b: 1 / sum(a * b)
        "features": array([[1, 2], [3, 5], [7, 11]]),
        "species_ordering": array(["species_1", "species_2", "species_3"]),
        "num_processors": cpu_count() + 1,
        "expected_species_ordering": array(["species_1", "species_2", "species_3"]),
        "relative_abundances": array([[1 / 1000, 1 / 100], [1 / 10, 1 / 1], [10, 100]]),
        "expected_weighted_similarities": array(
            [
                [0.35271989, 3.52719894],
                [0.13459705, 1.34597047],
                [0.0601738, 0.60173802],
            ]
        ),
        # similarity matrix
        # [0.2          , 0.07692307692, 0.03448275862]
        # [0.07692307692, 0.02941176471, 0.01315789474]
        # [0.03448275862, 0.01315789474, 0.005882352941]
    },
    {
        "description": "2 communities; 1 feature; default num_processors",
        "similarity_function": sim_func,  # lambda a, b: 1 / sum(a * b)
        "features": array([[1], [3], [7]]),
        "species_ordering": array(["species_1", "species_2", "species_3"]),
        "num_processors": None,
        "expected_species_ordering": array(["species_1", "species_2", "species_3"]),
        "relative_abundances": array([[1 / 1000, 1 / 100], [1 / 10, 1 / 1], [10, 100]]),
        "expected_weighted_similarities": array(
            [
                [1.46290476, 14.62904762],
                [0.48763492, 4.87634921],
                [0.20898639, 2.08986395],
            ]
        ),
        # similarity matrix
        # [1,   1/3,  1/7]
        # [1/3, 1/9,  1/21]
        # [1/7, 1/21, 1/49]
    },
    {
        "description": "1 community; 2 features; default num_processors",
        "similarity_function": sim_func,  # lambda a, b: 1 / sum(a * b)
        "features": array([[1, 2], [3, 5], [7, 11]]),
        "species_ordering": array(["species_1", "species_2", "species_3"]),
        "num_processors": None,
        "expected_species_ordering": array(["species_1", "species_2", "species_3"]),
        "relative_abundances": array([[1 / 1000], [1 / 10], [10]]),
        "expected_weighted_similarities": array(
            [[0.35271989], [0.13459705], [0.0601738]]
        ),
        # similarity matrix
        # [0.2          , 0.07692307692, 0.03448275862]
        # [0.07692307692, 0.02941176471, 0.01315789474]
        # [0.03448275862, 0.01315789474, 0.005882352941]
    },
    {
        "description": "1 community; 1 feature; default num_processors",
        "similarity_function": sim_func,  # lambda a, b: 1 / sum(a * b)
        "features": array([[1], [3], [7]]),
        "species_ordering": array(["species_1", "species_2", "species_3"]),
        "num_processors": None,
        "expected_species_ordering": array(["species_1", "species_2", "species_3"]),
        "relative_abundances": array([[1 / 1000], [1 / 10], [10]]),
        "expected_weighted_similarities": array(
            [
                [1.46290476],
                [0.48763492],
                [0.20898639],
            ]
        ),
        # similarity matrix
        # [1,   1/3,  1/7]
        # [1/3, 1/9,  1/21]
        # [1/7, 1/21, 1/49]
    },
]


class TestSimilarityFromFunction:
    """Tests metacommunity.Similarity."""

    @fixture(params=SIMILARITY_FROM_FUNCTION_TEST_CASES)
    def test_case(self, request, shared_array_manager):
        shared_features = shared_array_manager.from_array(request.param["features"])
        shared_relative_abundances = shared_array_manager.from_array(
            request.param["relative_abundances"]
        )
        test_case_ = {
            key: request.param[key]
            for key in [
                "description",
                "similarity_function",
                "species_ordering",
                "num_processors",
                "expected_species_ordering",
                "expected_weighted_similarities",
            ]
        }
        test_case_.update(
            {
                "shared_features": shared_features,
                "shared_relative_abundances": shared_relative_abundances,
                "shared_array_manager": shared_array_manager,
            }
        )
        return test_case_

    def test_init(self, test_case, tmp_path):
        """Tests initializer."""
        similarity = SimilarityFromFunction(
            similarity_function=test_case["similarity_function"],
            features_spec=test_case["shared_features"].spec,
            species_ordering=test_case["species_ordering"],
            shared_array_manager=test_case["shared_array_manager"],
            num_processors=test_case["num_processors"],
        )
        assert (
            similarity.species_ordering == test_case["expected_species_ordering"]
        ).all()

    def test_calculate_weighted_similarities(self, test_case, tmp_path):
        """Tests .calculate_weighted_similarities."""
        similarity = SimilarityFromFunction(
            similarity_function=test_case["similarity_function"],
            features_spec=test_case["shared_features"].spec,
            species_ordering=test_case["species_ordering"],
            shared_array_manager=test_case["shared_array_manager"],
            num_processors=test_case["num_processors"],
        )
        weighted_similarities = similarity.calculate_weighted_similarities(
            test_case["shared_relative_abundances"]
        )
        assert (
            weighted_similarities.data.shape
            == test_case["expected_weighted_similarities"].shape
        )
        assert allclose(
            weighted_similarities.data, test_case["expected_weighted_similarities"]
        )


SIMILARITY_FROM_FUNCTION_APPLY_SIMILARITY_FUNCTION_TEST_CASES = [
    {
        "description": "All rows; 2 columns",
        "func": sim_func,  # lambda a, b: 1 / sum(a * b)
        "row_start": 0,
        "row_stop": 3,
        "weighted_similarities": zeros(shape=(3, 2), dtype=dtype("f8")),
        "features": array([[1, 2], [3, 5], [7, 11]]),
        "relative_abundances": array([[1 / 1000, 1 / 100], [1 / 10, 1 / 1], [10, 100]]),
        "expected_weighted_similarities": array(
            [
                [0.35271989, 3.52719894],
                [0.13459705, 1.34597047],
                [0.0601738, 0.60173802],
            ]
        ),
    },
    {
        "description": "Single row; 2 columns",
        "func": sim_func,  # lambda a, b: 1 / sum(a * b)
        "row_start": 1,
        "row_stop": 2,
        "weighted_similarities": zeros(shape=(3, 2), dtype=dtype("f8")),
        "features": array([[1, 2], [3, 5], [7, 11]]),
        "relative_abundances": array([[1 / 1000, 1 / 100], [1 / 10, 1 / 1], [10, 100]]),
        "expected_weighted_similarities": array(
            [
                [0.0, 0.0],
                [0.13459705, 1.34597047],
                [0.0, 0.0],
            ]
        ),
    },
    {
        "description": "Some rows; 2 columns",
        "func": sim_func,  # lambda a, b: 1 / sum(a * b)
        "row_start": 1,
        "row_stop": 3,
        "weighted_similarities": zeros(shape=(3, 2), dtype=dtype("f8")),
        "features": array([[1, 2], [3, 5], [7, 11]]),
        "relative_abundances": array([[1 / 1000, 1 / 100], [1 / 10, 1 / 1], [10, 100]]),
        "expected_weighted_similarities": array(
            [
                [0.0, 0.0],
                [0.13459705, 1.34597047],
                [0.0601738, 0.60173802],
            ]
        ),
    },
    {
        "description": "No rows; 2 columns",
        "func": sim_func,  # lambda a, b: 1 / sum(a * b)
        "row_start": 0,
        "row_stop": 0,
        "weighted_similarities": zeros(shape=(3, 2), dtype=dtype("f8")),
        "features": array([[1, 2], [3, 5], [7, 11]]),
        "relative_abundances": array([[1 / 1000, 1 / 100], [1 / 10, 1 / 1], [10, 100]]),
        "expected_weighted_similarities": array(
            [
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ]
        ),
    },
    {
        "description": "All rows; 1 column",
        "func": sim_func,  # lambda a, b: 1 / sum(a * b)
        "row_start": 0,
        "row_stop": 3,
        "weighted_similarities": zeros(shape=(3, 1), dtype=dtype("f8")),
        "features": array([[1, 2], [3, 5], [7, 11]]),
        "relative_abundances": array(
            [
                [1 / 1000],
                [1 / 10],
                [10],
            ]
        ),
        "expected_weighted_similarities": array(
            [
                [0.35271989],
                [0.13459705],
                [0.0601738],
            ]
        ),
    },
    {
        "description": "Single row; 1 column",
        "func": sim_func,  # lambda a, b: 1 / sum(a * b)
        "row_start": 2,
        "row_stop": 3,
        "weighted_similarities": zeros(shape=(3, 1), dtype=dtype("f8")),
        "features": array([[1, 2], [3, 5], [7, 11]]),
        "relative_abundances": array(
            [
                [1 / 1000],
                [1 / 10],
                [10],
            ]
        ),
        "expected_weighted_similarities": array(
            [
                [0.0],
                [0.0],
                [0.0601738],
            ]
        ),
    },
    {
        "description": "Some rows; 1 column",
        "func": sim_func,  # lambda a, b: 1 / sum(a * b)
        "row_start": 0,
        "row_stop": 2,
        "weighted_similarities": zeros(shape=(3, 1), dtype=dtype("f8")),
        "features": array([[1, 2], [3, 5], [7, 11]]),
        "relative_abundances": array(
            [
                [1 / 1000],
                [1 / 10],
                [10],
            ]
        ),
        "expected_weighted_similarities": array(
            [
                [0.35271989],
                [0.13459705],
                [0.0],
            ]
        ),
    },
    {
        "description": "No rows; 1 column",
        "func": sim_func,  # lambda a, b: 1 / sum(a * b)
        "row_start": 1,
        "row_stop": 1,
        "weighted_similarities": zeros(shape=(3, 1), dtype=dtype("f8")),
        "features": array([[1, 2], [3, 5], [7, 11]]),
        "relative_abundances": array(
            [
                [1 / 1000],
                [1 / 10],
                [10],
            ]
        ),
        "expected_weighted_similarities": array(
            [
                [0.0],
                [0.0],
                [0.0],
            ]
        ),
    },
]


class TestSimilarityFromFunctionApplysimilarityFunction:
    """Tests metacommunity.SimilarityFromFunction.ApplySimilarityFunction."""

    @fixture(params=SIMILARITY_FROM_FUNCTION_APPLY_SIMILARITY_FUNCTION_TEST_CASES)
    def test_case(
        self,
        request,
        shared_array_manager,
    ):
        shared_weighted_similarities = shared_array_manager.from_array(
            request.param["weighted_similarities"]
        )
        shared_features = shared_array_manager.from_array(request.param["features"])
        shared_relative_abundances = shared_array_manager.from_array(
            request.param["relative_abundances"]
        )
        test_case_ = {
            key: request.param[key]
            for key in [
                "description",
                "func",
                "row_start",
                "row_stop",
                "expected_weighted_similarities",
            ]
        }
        test_case_.update(
            {
                "shared_weighted_similarities": shared_weighted_similarities,
                "shared_features": shared_features,
                "shared_relative_abundances": shared_relative_abundances,
            }
        )
        return test_case_

    def test_call(self, test_case):
        """Tests .__call__."""
        apply_similarity_function = SimilarityFromFunction.ApplySimilarityFunction(
            test_case["func"]
        )
        apply_similarity_function(
            test_case["row_start"],
            test_case["row_stop"],
            test_case["shared_weighted_similarities"].spec,
            test_case["shared_features"].spec,
            test_case["shared_relative_abundances"].spec,
        )
        assert allclose(
            test_case["shared_weighted_similarities"].data,
            test_case["expected_weighted_similarities"],
        )


SIMILARITY_FROM_MEMORY_TEST_CASES = [
    {
        "description": "2 communities",
        "similarity_matrix": DataFrame(
            data=array(
                [
                    [1, 0.5, 0.1],
                    [0.5, 1, 0.2],
                    [0.1, 0.2, 1],
                ]
            ),
            columns=["species_1", "species_2", "species_3"],
            index=["species_1", "species_2", "species_3"],
        ),
        "species_subset": ["species_1", "species_2", "species_3"],
        "expected_species_ordering": array(["species_1", "species_2", "species_3"]),
        "expected_similarity_matrix": DataFrame(
            data=array(
                [
                    [1, 0.5, 0.1],
                    [0.5, 1, 0.2],
                    [0.1, 0.2, 1],
                ]
            ),
            columns=["species_1", "species_2", "species_3"],
            index=["species_1", "species_2", "species_3"],
        ),
        "relative_abundances": array([[1 / 1000, 1 / 100], [1 / 10, 1 / 1], [10, 100]]),
        "weighted_similarities": array(
            [[1.051, 10.51], [2.1005, 21.005], [10.0201, 100.201]]
        ),
    },
    {
        "description": "1 community",
        "similarity_matrix": DataFrame(
            data=array(
                [
                    [1, 0.5, 0.1],
                    [0.5, 1, 0.2],
                    [0.1, 0.2, 1],
                ]
            ),
            columns=["species_1", "species_2", "species_3"],
            index=["species_1", "species_2", "species_3"],
        ),
        "expected_similarity_matrix": DataFrame(
            data=array(
                [
                    [1, 0.5, 0.1],
                    [0.5, 1, 0.2],
                    [0.1, 0.2, 1],
                ]
            ),
            columns=["species_1", "species_2", "species_3"],
            index=["species_1", "species_2", "species_3"],
        ),
        "species_subset": ["species_1", "species_2", "species_3"],
        "expected_species_ordering": array(["species_1", "species_2", "species_3"]),
        "relative_abundances": array([[1 / 1000], [1 / 10], [10]]),
        "weighted_similarities": array([[1.051], [2.1005], [10.0201]]),
    },
    {
        "description": "2 communities; shuffled index",
        "similarity_matrix": DataFrame(
            data=array(
                [
                    [0.5, 1, 0.2],
                    [1, 0.5, 0.1],
                    [0.1, 0.2, 1],
                ]
            ),
            columns=["species_1", "species_2", "species_3"],
            index=["species_2", "species_1", "species_3"],
        ),
        "species_subset": ["species_2", "species_1", "species_3"],
        "expected_similarity_matrix": DataFrame(
            data=array(
                [
                    [1, 0.5, 0.1],
                    [0.5, 1, 0.2],
                    [0.1, 0.2, 1],
                ]
            ),
            columns=["species_1", "species_2", "species_3"],
            index=["species_1", "species_2", "species_3"],
        ),
        "expected_species_ordering": array(["species_1", "species_2", "species_3"]),
        "relative_abundances": array([[1 / 1000, 1 / 100], [1 / 10, 1 / 1], [10, 100]]),
        "weighted_similarities": array(
            [[1.051, 10.51], [2.1005, 21.005], [10.0201, 100.201]]
        ),
    },
    {
        "description": "1 community; shuffled index",
        "similarity_matrix": DataFrame(
            data=array(
                [
                    [1, 0.5, 0.1],
                    [0.1, 0.2, 1],
                    [0.5, 1, 0.2],
                ]
            ),
            columns=["species_1", "species_2", "species_3"],
            index=["species_1", "species_3", "species_2"],
        ),
        "species_subset": ["species_1", "species_2", "species_3"],
        "expected_similarity_matrix": DataFrame(
            data=array(
                [
                    [1, 0.5, 0.1],
                    [0.5, 1, 0.2],
                    [0.1, 0.2, 1],
                ]
            ),
            columns=["species_1", "species_2", "species_3"],
            index=["species_1", "species_2", "species_3"],
        ),
        "expected_species_ordering": array(["species_1", "species_2", "species_3"]),
        "relative_abundances": array([[1 / 1000], [1 / 10], [10]]),
        "weighted_similarities": array([[1.051], [2.1005], [10.0201]]),
    },
    {
        "description": "species subset",
        "similarity_matrix": DataFrame(
            data=array(
                [
                    [1, 0.5, 0.1],
                    [0.5, 1, 0.2],
                    [0.1, 0.2, 1],
                ]
            ),
            columns=["species_1", "species_2", "species_3"],
            index=["species_1", "species_2", "species_3"],
        ),
        "species_subset": ["species_2", "species_3"],
        "expected_species_ordering": ["species_2", "species_3"],
        "expected_similarity_matrix": DataFrame(
            data=array(
                [
                    [1, 0.2],
                    [0.2, 1],
                ]
            ),
            columns=["species_2", "species_3"],
            index=["species_2", "species_3"],
        ),
        "relative_abundances": array([[1 / 10, 1 / 1], [10, 100]]),
        "weighted_similarities": array([[2.1, 21.0], [10.02, 100.2]]),
    },
]


class TestSimilarityFromMemory:
    """Tests metacommunity.Similarity."""

    @mark.parametrize("test_case", SIMILARITY_FROM_MEMORY_TEST_CASES)
    def test_init(self, test_case):
        """Tests initializer."""
        similarity = SimilarityFromMemory(
            similarity_matrix=test_case["similarity_matrix"],
            species_subset=test_case["species_subset"],
        )
        assert array_equal(
            similarity.species_ordering, test_case["expected_species_ordering"]
        )
        assert_frame_equal(
            similarity.similarity_matrix, test_case["expected_similarity_matrix"]
        )

    @mark.parametrize("test_case", SIMILARITY_FROM_MEMORY_TEST_CASES)
    def test_calculate_weighted_similarities(self, test_case, tmp_path):
        """Tests .calculate_weighted_similarities."""
        similarity = SimilarityFromMemory(
            similarity_matrix=test_case["similarity_matrix"],
            species_subset=test_case["species_subset"],
        )
        weighted_similarities = similarity.calculate_weighted_similarities(
            test_case["relative_abundances"]
        )
        assert weighted_similarities.shape == test_case["weighted_similarities"].shape
        assert allclose(weighted_similarities, test_case["weighted_similarities"])
