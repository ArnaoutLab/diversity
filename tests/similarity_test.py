"""Tests for diversity.similarity."""
from multiprocessing import cpu_count

from numpy import allclose, array, array_equal, dtype, empty, memmap, ones, zeros
from pandas import DataFrame, Index
from pytest import fixture, raises, warns

from diversity.exceptions import ArgumentWarning, InvalidArgumentError
from diversity.log import LOGGER
from diversity.similarity import (
    SimilarityFromArray,
    SimilarityFromDataFrame,
    SimilarityFromMemmap,
    make_similarity,
    SimilarityFromFile,
)


class MockClass:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class MockSimilarityFromFile(MockClass):
    pass


class MockSimilarityFromDataFrame(MockClass):
    pass


class MockSimilarityFromArray(MockClass):
    pass


class MockSimilarityFromMemmap(MockClass):
    pass


FAKE_SPECIES_ORDERING = "fake_ordering"
FAKE_FEATURES = "fake_features"


def sim_func(a, b):
    return 1 / sum(a * b)


MAKE_SIMILARITY_TEST_CASES = [
    {
        "description": "SimilarityFromDataFrame",
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
        "species_order": None,
        "chunk_size": 1,
        "expect_raise": False,
        "expected_return_type": MockSimilarityFromDataFrame,
    },
    {
        "description": "SimilarityFromArray numpy array",
        "similarity": array(
            [
                [1, 0.5, 0.1],
                [0.5, 1, 0.2],
                [0.1, 0.2, 1],
            ]
        ),
        "species_order": ["species_1", "species_2", "species_3"],
        "chunk_size": 1,
        "expect_raise": False,
        "expected_return_type": MockSimilarityFromArray,
    },
    {
        "description": "SimilarityFromMemmap numpy memmap",
        "similarity": (
            "memmap",
            array(
                [
                    [1, 0.5, 0.1],
                    [0.5, 1, 0.2],
                    [0.1, 0.2, 1],
                ]
            ),
        ),
        "species_order": ["species_1", "species_2", "species_3"],
        "chunk_size": 1,
        "expect_raise": False,
        "expected_return_type": MockSimilarityFromMemmap,
    },
    {
        "description": "SimilarityFromFile",
        "similarity": "fake_similarities_file.tsv",
        "species_order": ["species_1", "species_2", "species_3"],
        "chunk_size": 1,
        "expect_raise": False,
        "expected_return_type": MockSimilarityFromFile,
    },
    {
        "description": "SimilarityFromFile with non-default chunk_size",
        "similarity": "fake_similarities_file.tsv",
        "species_order": ["species_1", "species_2", "species_3"],
        # Make chunk_size large to avoid builtin optimization assigning precomputed reference
        "chunk_size": 1228375972486598237,
        "expect_raise": False,
        "expected_return_type": MockSimilarityFromFile,
    },
]


class TestMakeSimilarity:
    @fixture(params=MAKE_SIMILARITY_TEST_CASES)
    def test_case(self, request, monkeypatch, tmp_path):
        mock_classes = [
            ("diversity.similarity.SimilarityFromFile", MockSimilarityFromFile),
            (
                "diversity.similarity.SimilarityFromDataFrame",
                MockSimilarityFromDataFrame,
            ),
            (
                "diversity.similarity.SimilarityFromArray",
                MockSimilarityFromArray,
            ),
            (
                "diversity.similarity.SimilarityFromMemmap",
                MockSimilarityFromMemmap,
            ),
        ]
        with monkeypatch.context() as patched_context:
            for target, mock_class in mock_classes:
                patched_context.setattr(target, mock_class)
            test_case_ = {
                key: request.param[key]
                for key in [
                    "species_order",
                    "chunk_size",
                    "expect_raise",
                    "expected_return_type",
                ]
            }

            if (
                type(request.param["similarity"]) == tuple
                and request.param["similarity"][0] == "memmap"
            ):
                similarity = request.param["similarity"][1]
                memmapped = memmap(
                    tmp_path / "similarity.npy",
                    dtype=dtype("f8"),
                    mode="w+",
                    offset=0,
                    shape=(similarity.shape[0], similarity.shape[0]),
                    order="C",
                )
                memmapped[:, :] = similarity
                test_case_["similarity"] = memmapped
            else:
                test_case_["similarity"] = request.param["similarity"]

            if request.param["expected_return_type"] == MockSimilarityFromFile:
                test_case_["expected_init_kwargs"] = {
                    "similarity": test_case_["similarity"],
                    "chunk_size": test_case_["chunk_size"],
                }
            elif request.param["expected_return_type"] == MockSimilarityFromDataFrame:
                test_case_["expected_init_kwargs"] = {
                    "similarity": test_case_["similarity"],
                }
            elif request.param["expected_return_type"] == MockSimilarityFromArray:
                test_case_["expected_init_kwargs"] = {
                    "similarity": test_case_["similarity"],
                    "species_order": test_case_["species_order"],
                }
            elif request.param["expected_return_type"] == MockSimilarityFromMemmap:
                test_case_["expected_init_kwargs"] = {
                    "similarity": test_case_["similarity"],
                    "species_order": test_case_["species_order"],
                }

            yield test_case_

    def test_make_similarity(self, test_case):
        if test_case["expect_raise"]:
            with raises(InvalidArgumentError):
                breakpoint()
                make_similarity(
                    similarity=test_case["similarity"],
                    species_order=test_case["species_order"],
                    chunk_size=test_case["chunk_size"],
                )
        else:
            similarity = make_similarity(
                similarity=test_case["similarity"],
                species_order=test_case["species_order"],
                chunk_size=test_case["chunk_size"],
            )
            assert isinstance(similarity, test_case["expected_return_type"])
            for key, arg in test_case["expected_init_kwargs"].items():
                assert similarity.kwargs[key] is arg


SIMILARITY_FROM_FILE_TEST_CASES = [
    {
        "description": "tsv file; 2 communities",
        "similarity_matrix_filepath": "similarities_file.tsv",
        "species_order": ["species_3", "species_1", "species_2"],
        "chunk_size": 1,
        "expected_species_ordering": Index(["species_3", "species_1", "species_2"]),
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
        "species_order": ["species_3", "species_1", "species_2"],
        "chunk_size": 1,
        "expected_species_ordering": Index(["species_3", "species_1", "species_2"]),
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
        "species_order": ["species_3", "species_1", "species_2"],
        "chunk_size": 1,
        "expected_species_ordering": Index(["species_3", "species_1", "species_2"]),
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
        "species_order": ["species_3", "species_1", "species_2"],
        "chunk_size": 12,
        "expected_species_ordering": Index(["species_3", "species_1", "species_2"]),
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
                "species_order",
                "chunk_size",
                "expected_species_ordering",
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
        assert array_equal(
            similarity.species_ordering, test_case["expected_species_ordering"]
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
        "species_order": ["species_1", "species_2", "species_3"],
        "expected_species_ordering": Index(["species_1", "species_2", "species_3"]),
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
        "species_order": ["species_1", "species_2", "species_3"],
        "expected_species_ordering": Index(["species_1", "species_2", "species_3"]),
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
                "species_order",
                "expected_species_ordering",
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
        assert array_equal(
            similarity.species_ordering, test_case["expected_species_ordering"]
        )
        assert allclose(similarity.similarity, test_case["expected_similarity_matrix"])

    def test_calculate_weighted_similarities(self, test_case, tmp_path):
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
        species_order = ["a", "b", "c"]
        similarity = SimilarityFromArray(
            similarity=similarity_matrix,
            species_order=species_order,
        )
        assert array_equal(similarity.species_ordering, species_order)
        assert allclose(similarity.similarity, similarity_matrix)

    def test_init_with_numpy_array_but_non_sequence_species_order(self):
        similarity_matrix = array(
            [
                [1.0, 0.9, 0.1],
                [0.9, 1.0, 0.2],
                [0.1, 0.2, 1.0],
            ]
        )
        species_order = {"a", "b", "c"}
        with raises(InvalidArgumentError):
            similarity = SimilarityFromArray(
                similarity=similarity_matrix,
                species_order=species_order,
            )

    def test_init_with_numpy_array_but_too_short_sequence_species_order(self):
        similarity_matrix = array(
            [
                [1.0, 0.9, 0.1],
                [0.9, 1.0, 0.2],
                [0.1, 0.2, 1.0],
            ]
        )
        species_order = ["a", "b"]
        with raises(InvalidArgumentError):
            similarity = SimilarityFromArray(
                similarity=similarity_matrix,
                species_order=species_order,
            )

    def test_calculate_weighted_similarities_with_numpy_array(self):
        similarity_matrix = array(
            [
                [1.0, 0.9, 0.1],
                [0.9, 1.0, 0.2],
                [0.1, 0.2, 1.0],
            ]
        )
        species_order = ["a", "b", "c"]
        similarity = SimilarityFromArray(
            similarity=similarity_matrix,
            species_order=species_order,
        )
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
        species_order = ["a", "b", "c"]
        similarity = SimilarityFromArray(
            similarity=similarity_matrix,
            species_order=species_order,
        )
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
        species_order = ["a", "b", "c"]
        similarity = SimilarityFromMemmap(
            similarity=memmapped_similarity_matrix,
            species_order=species_order,
        )
        assert array_equal(similarity.species_ordering, species_order)
        assert allclose(similarity.similarity, similarity_matrix)

    def test_init_with_numpy_memmap_but_non_sequence_species_order(self, tmp_path):
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
        species_order = {"a", "b", "c"}
        with raises(InvalidArgumentError):
            similarity = SimilarityFromMemmap(
                similarity=memmapped_similarity_matrix,
                species_order=species_order,
            )

    def test_init_with_numpy_memmap_but_too_short_sequence_species_order(
        self, tmp_path
    ):
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
        species_order = ["a", "b"]
        with raises(InvalidArgumentError):

            similarity = SimilarityFromMemmap(
                similarity=memmapped_similarity_matrix,
                species_order=species_order,
            )

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
        species_order = ["a", "b", "c"]
        similarity = SimilarityFromMemmap(
            similarity=memmapped_similarity_matrix,
            species_order=species_order,
        )
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
        species_order = ["a", "b", "c"]
        similarity = SimilarityFromMemmap(
            similarity=memmapped_similarity_matrix,
            species_order=species_order,
        )
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
