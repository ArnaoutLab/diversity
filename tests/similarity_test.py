"""Tests for diversity.similarity."""
from numpy import allclose, array, dtype, empty, memmap
from pandas import DataFrame
from pytest import fixture, warns

from diversity.exceptions import ArgumentWarning
from diversity.log import LOGGER
from diversity.similarity import (
    SimilarityFromArray,
    SimilarityFromDataFrame,
    SimilarityFromFile,
    make_similarity,
)

similarity_array_3x3 = array(
    [
        [1, 0.5, 0.1],
        [0.5, 1, 0.2],
        [0.1, 0.2, 1],
    ]
)

similarity_dataframe_3x3 = DataFrame(
    data=similarity_array_3x3,
    columns=["species_1", "species_2", "species_3"],
    index=["species_1", "species_2", "species_3"],
)

relative_abundances_3x2 = array([[1 / 1000, 1 / 100], [1 / 10, 1 / 1], [10, 100]])


class MockClass:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class MockSimilarityFromFile(MockClass):
    pass


class MockSimilarityFromDataFrame(MockClass):
    pass


class MockSimilarityFromArray(MockClass):
    pass


@fixture
def make_similarity_kwargs():
    def make(
        similarity: similarity_dataframe_3x3,
        chunk_size=None,
    ):
        return {
            "similarity": similarity,
            "chunk_size": chunk_size,
        }

    return make


def test_make_similarity_from_dataframe():
    similarity = make_similarity(similarity_dataframe_3x3)
    assert type(similarity) is SimilarityFromDataFrame


def test_make_similarity_from_array():
    similarity = make_similarity(similarity_array_3x3)
    assert type(similarity) is SimilarityFromArray


def test_make_similarity_from_memmap(tmp_path):
    memmapped = memmap(
        tmp_path / "similarity.npy",
        dtype=dtype("f8"),
        mode="w+",
        offset=0,
        shape=(similarity_array_3x3.shape[0], similarity_array_3x3.shape[0]),
        order="C",
    )
    memmapped[:, :] = similarity_array_3x3
    similarity = make_similarity(memmapped, chunk_size=None)
    assert type(similarity) is SimilarityFromArray


def test_make_similarity_from_file():
    similarity = make_similarity("fake_similarities_file.tsv", chunk_size=100)
    assert type(similarity) is SimilarityFromFile


def test_make_similarity_from_file_chunk_size():
    similarity = make_similarity("fake_similarities_file.tsv", chunk_size=3)
    assert type(similarity) is SimilarityFromFile


# TODO
# def test_make_similarity_from_function():
#     similarity = make_similarity()
#     assert similarity is SimilarityFromFunction


def test_do_not_make_similarity():
    similarity = make_similarity(None)
    assert similarity is None


SIMILARITY_FROM_FILE_TEST_CASES = [
    {
        "description": "tsv file; 2 communities",
        "similarity_matrix_filepath": "similarities_file.tsv",
        "chunk_size": 1,
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
        "out": None,
        "expect_warning": False,
    },
    {
        "description": "csv file; 2 communities",
        "similarity_matrix_filepath": "similarities_file.csv",
        "chunk_size": 1,
        "relative_abundances": array([[1 / 1000, 1 / 100], [1 / 10, 1 / 1], [10, 100]]),
        "weighted_similarities": array(
            [[2.011, 20.11], [5.1001, 51.001], [10.0502, 100.502]]
        ),
        "similarities_filecontents": (
            "species_3,species_1,species_2\n" "1,0.1,0.2\n" "0.1,1,0.5\n" "0.2,0.5,1\n"
        ),
        "out": None,
        "expect_warning": False,
    },
    {
        "description": "no file extension; 1 community",
        "similarity_matrix_filepath": "similarities_file",
        "chunk_size": 1,
        "relative_abundances": array([[1 / 1000], [1 / 10], [10]]),
        "weighted_similarities": array([[2.011], [5.1001], [10.0502]]),
        "similarities_filecontents": (
            "species_3\tspecies_1\tspecies_2\n"
            "1\t0.1\t0.2\n"
            "0.1\t1\t0.5\n"
            "0.2\t0.5\t1\n"
        ),
        "out": None,
        "expect_warning": True,
    },
    {
        "description": "non-default chunk_size",
        "similarity_matrix_filepath": "similarities_file.tsv",
        "chunk_size": 12,
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
        "out": None,
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
                "chunk_size",
                "weighted_similarities",
                "similarities_filecontents",
                "expect_warning",
            ]
        }
        test_case_["similarity"] = filepath
        test_case_["relative_abundances"] = request.param["relative_abundances"]
        if request.param["out"] is None:
            test_case_["out"] = None
        else:
            test_case_["out"] = request.param["out"]
        return test_case_

    def test_init(self, test_case):
        """Tests initializer."""
        if test_case["expect_warning"]:
            with warns(ArgumentWarning):
                similarity = SimilarityFromFile(
                    similarity=test_case["similarity"],
                    chunk_size=test_case["chunk_size"],
                )
        else:
            similarity = SimilarityFromFile(
                similarity=test_case["similarity"],
                chunk_size=test_case["chunk_size"],
            )
        assert similarity.similarity == test_case["similarity"]
        assert similarity.chunk_size == test_case["chunk_size"]

    def test_calculate_weighted_similarities(self, test_case):
        """Tests .calculate_weighted_similarities."""
        if test_case["expect_warning"]:
            with warns(ArgumentWarning):
                similarity = SimilarityFromFile(
                    similarity=test_case["similarity"],
                    chunk_size=test_case["chunk_size"],
                )
        else:
            similarity = SimilarityFromFile(
                similarity=test_case["similarity"],
                chunk_size=test_case["chunk_size"],
            )
        weighted_similarities = similarity.calculate_weighted_similarities(
            relative_abundances=test_case["relative_abundances"],
        )
        assert weighted_similarities.shape == test_case["weighted_similarities"].shape
        assert allclose(weighted_similarities, test_case["weighted_similarities"])
        with open(test_case["similarity"], "r") as file:
            similarities_filecontents = file.read()
        assert similarities_filecontents == test_case["similarities_filecontents"]
        if test_case["out"] is not None:
            assert weighted_similarities is test_case["out"]


SIMILARITY_FROM_MEMORY_TEST_CASES = [
    {
        "description": "2 communities",
        "similarity": DataFrame(
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
        "expected_similarity_matrix": array(
            [
                [1, 0.5, 0.1],
                [0.5, 1, 0.2],
                [0.1, 0.2, 1],
            ]
        ),
        "relative_abundances": array([[1 / 1000, 1 / 100], [1 / 10, 1 / 1], [10, 100]]),
        "out": None,
        "weighted_similarities": array(
            [[1.051, 10.51], [2.1005, 21.005], [10.0201, 100.201]]
        ),
    },
    {
        "description": "1 community",
        "similarity": DataFrame(
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
        "expected_similarity_matrix": array(
            [
                [1, 0.5, 0.1],
                [0.5, 1, 0.2],
                [0.1, 0.2, 1],
            ]
        ),
        "relative_abundances": array([[1 / 1000], [1 / 10], [10]]),
        "out": None,
        "weighted_similarities": array([[1.051], [2.1005], [10.0201]]),
    },
]


class TestSimilarityFromMemory:
    """Tests metacommunity.Similarity."""

    @fixture(params=SIMILARITY_FROM_MEMORY_TEST_CASES)
    def test_case(self, request):
        test_case_ = {
            key: request.param[key]
            for key in [
                "similarity",
                "expected_similarity_matrix",
                "weighted_similarities",
            ]
        }
        test_case_["relative_abundances"] = request.param["relative_abundances"]
        if request.param["out"] is None:
            test_case_["out"] = None
        else:
            test_case_["out"] = request.param["out"]
        return test_case_

    def test_init(self, test_case):
        """Tests initializer."""
        similarity = SimilarityFromDataFrame(similarity=test_case["similarity"])
        assert allclose(similarity.similarity, test_case["expected_similarity_matrix"])

    def test_calculate_weighted_similarities(self, test_case):
        """Tests .calculate_weighted_similarities."""
        similarity = SimilarityFromDataFrame(similarity=test_case["similarity"])
        weighted_similarities = similarity.calculate_weighted_similarities(
            relative_abundances=test_case["relative_abundances"]
        )
        assert weighted_similarities.shape == test_case["weighted_similarities"].shape
        assert allclose(weighted_similarities, test_case["weighted_similarities"])
        if test_case["out"] is not None:
            assert weighted_similarities is test_case["out"]

    def test_init_with_numpy_array(self):
        similarity_matrix = array(
            [
                [1.0, 0.9, 0.1],
                [0.9, 1.0, 0.2],
                [0.1, 0.2, 1.0],
            ]
        )
        similarity = SimilarityFromArray(similarity=similarity_matrix)
        assert allclose(similarity.similarity, similarity_matrix)

    def test_calculate_weighted_similarities_with_numpy_array(self):
        similarity_matrix = array(
            [
                [1.0, 0.9, 0.1],
                [0.9, 1.0, 0.2],
                [0.1, 0.2, 1.0],
            ]
        )
        similarity = SimilarityFromArray(similarity=similarity_matrix)
        relative_abundances = array(
            [
                [0.7, 0.3],
                [0.1, 0.3],
                [0.2, 0.4],
            ]
        )
        weighted_similarities = similarity.calculate_weighted_similarities(
            relative_abundances,
        )
        expected_weighted_similarities = array(
            [
                [0.81, 0.61],
                [0.77, 0.65],
                [0.29, 0.49],
            ]
        )
        assert allclose(weighted_similarities, expected_weighted_similarities)

    def test_calculate_weighted_similarities_with_numpy_array(self):
        similarity_matrix = array(
            [
                [1.0, 0.9, 0.1],
                [0.9, 1.0, 0.2],
                [0.1, 0.2, 1.0],
            ]
        )
        similarity = SimilarityFromArray(similarity=similarity_matrix)
        relative_abundances = array(
            [
                [0.7, 0.3],
                [0.1, 0.3],
                [0.2, 0.4],
            ]
        )
        out = empty(
            dtype=dtype("f8"),
            shape=(
                similarity_matrix.shape[0],
                relative_abundances.shape[1],
            ),
        )
        weighted_similarities = similarity.calculate_weighted_similarities(
            relative_abundances,
        )
        expected_weighted_similarities = array(
            [
                [0.81, 0.61],
                [0.77, 0.65],
                [0.29, 0.49],
            ]
        )
        assert allclose(weighted_similarities, expected_weighted_similarities)
        assert weighted_similarities is out

    def test_init_with_numpy_memmap(self, tmp_path):
        similarity_matrix = array(
            [
                [1.0, 0.9, 0.1],
                [0.9, 1.0, 0.2],
                [0.1, 0.2, 1.0],
            ]
        )
        memmapped_similarity_matrix = memmap(
            tmp_path / "similarity_matrix.npy",
            dtype=dtype("f8"),
            mode="w+",
            offset=0,
            shape=similarity_matrix.shape,
            order="C",
        )
        memmapped_similarity_matrix[:, :] = similarity_matrix
        similarity = SimilarityFromArray(similarity=memmapped_similarity_matrix)
        assert allclose(similarity.similarity, similarity_matrix)

    def test_calculate_weighted_similarities_with_numpy_memmap(self, tmp_path):
        similarity_matrix = array(
            [
                [1.0, 0.9, 0.1],
                [0.9, 1.0, 0.2],
                [0.1, 0.2, 1.0],
            ]
        )
        memmapped_similarity_matrix = memmap(
            tmp_path / "similarity_matrix.npy",
            dtype=dtype("f8"),
            mode="w+",
            offset=0,
            shape=similarity_matrix.shape,
            order="C",
        )
        memmapped_similarity_matrix[:, :] = similarity_matrix
        similarity = SimilarityFromArray(similarity=memmapped_similarity_matrix)
        relative_abundances = array(
            [
                [0.7, 0.3],
                [0.1, 0.3],
                [0.2, 0.4],
            ]
        )
        weighted_similarities = similarity.calculate_weighted_similarities(
            relative_abundances,
        )
        expected_weighted_similarities = array(
            [
                [0.81, 0.61],
                [0.77, 0.65],
                [0.29, 0.49],
            ]
        )
        assert allclose(weighted_similarities, expected_weighted_similarities)

    def test_calculate_weighted_similarities_with_numpy_array(self, tmp_path):
        similarity_matrix = array(
            [
                [1.0, 0.9, 0.1],
                [0.9, 1.0, 0.2],
                [0.1, 0.2, 1.0],
            ]
        )
        memmapped_similarity_matrix = memmap(
            tmp_path / "similarity_matrix.npy",
            dtype=dtype("f8"),
            mode="w+",
            offset=0,
            shape=similarity_matrix.shape,
            order="C",
        )
        memmapped_similarity_matrix[:, :] = similarity_matrix
        similarity = SimilarityFromArray(similarity=memmapped_similarity_matrix)
        relative_abundances = array(
            [
                [0.7, 0.3],
                [0.1, 0.3],
                [0.2, 0.4],
            ]
        )
        weighted_similarities = similarity.calculate_weighted_similarities(
            relative_abundances
        )
        expected_weighted_similarities = array(
            [
                [0.81, 0.61],
                [0.77, 0.65],
                [0.29, 0.49],
            ]
        )
        assert allclose(weighted_similarities, expected_weighted_similarities)
