"""Tests for diversity.similarity."""
from multiprocessing import cpu_count

from numpy import allclose, array, array_equal, dtype, ones, zeros
from pandas import DataFrame, Index
from pandas.testing import assert_frame_equal
from pytest import fixture, mark, raises, warns

from diversity.exceptions import ArgumentWarning, InvalidArgumentError
from diversity.log import LOGGER
from diversity.similarity import (
    make_similarity,
    SimilarityFromFile,
    SimilarityFromFunction,
    SimilarityFromMemory,
)


class MockClass:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class MockSimilarityFromFile(MockClass):
    pass


FAKE_SPECIES_ORDERING = "fake_ordering"


class MockSimilarityFromFunction(MockClass):
    @staticmethod
    def read_shared_features(**kwargs):
        return (kwargs, FAKE_SPECIES_ORDERING)


class MockSimilarityFromMemory(MockClass):
    pass


def sim_func(a, b):
    return 1 / sum(a * b)


MAKE_SIMILARITY_TEST_CASES = [
    {
        "description": "SimilarityFromMemory",
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
        "species_subset": ["species_1", "species_2", "species_3"],
        "chunk_size": 1,
        "features_filepath": None,
        "species_column": None,
        "shared_array_manager": False,
        "num_processors": None,
        "expect_raise": False,
        "expected_return_type": MockSimilarityFromMemory,
    },
    {
        "description": "SimilarityFromFile",
        "similarity": "fake_similarities_file.tsv",
        "species_subset": ["species_1", "species_2", "species_3"],
        "chunk_size": 1,
        "features_filepath": None,
        "species_column": None,
        "shared_array_manager": False,
        "num_processors": None,
        "expect_raise": False,
        "expected_return_type": MockSimilarityFromFile,
    },
    {
        "description": "SimilarityFromFile with non-default chunk_size",
        "similarity": "fake_similarities_file.tsv",
        "species_subset": ["species_1", "species_2", "species_3"],
        # Make chunk_size large to avoid builtin optimization assigning precomputed reference
        "chunk_size": 1228375972486598237,
        "features_filepath": None,
        "species_column": None,
        "shared_array_manager": False,
        "num_processors": None,
        "expect_raise": False,
        "expected_return_type": MockSimilarityFromFile,
    },
    {
        "description": "SimilarityFromFunction",
        "similarity": sim_func,
        "species_subset": ["species_1", "species_2", "species_3"],
        "chunk_size": 1,
        "features_filepath": "fake_features_file.tsv",
        "species_column": "fake_species_col",
        "shared_array_manager": True,
        "num_processors": None,
        "expect_raise": False,
        "expected_return_type": MockSimilarityFromFunction,
    },
    {
        "description": "SimilarityFromFunction with non-default num_processors",
        "similarity": sim_func,
        "species_subset": ["species_1", "species_2", "species_3"],
        "chunk_size": 1,
        "features_filepath": "fake_features_file.tsv",
        "species_column": "fake_species_col",
        "shared_array_manager": True,
        # Make num_processors large to avoid builtin optimization assigning precomputed reference
        "num_processors": 1228375972486598237,
        "expect_raise": False,
        "expected_return_type": MockSimilarityFromFunction,
    },
    {
        "description": "SimilarityFromMemory with non-default features_filepath",
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
        "species_subset": ["species_1", "species_2", "species_3"],
        "chunk_size": 1,
        "features_filepath": "fake_features_file.tsv",
        "species_column": None,
        "shared_array_manager": False,
        "num_processors": None,
        "expect_raise": True,
        "expected_return_type": None,
    },
    {
        "description": "SimilarityFromMemory with non-default species_column",
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
        "species_subset": ["species_1", "species_2", "species_3"],
        "chunk_size": 1,
        "features_filepath": None,
        "species_column": "fake_species_col",
        "shared_array_manager": False,
        "num_processors": None,
        "expect_raise": True,
        "expected_return_type": None,
    },
    {
        "description": "SimilarityFromMemory with non-default shared_array_manager",
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
        "species_subset": ["species_1", "species_2", "species_3"],
        "chunk_size": 1,
        "features_filepath": None,
        "species_column": None,
        "shared_array_manager": True,
        "num_processors": None,
        "expect_raise": True,
        "expected_return_type": None,
    },
    {
        "description": "SimilarityFromMemory with non-default num_processors",
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
        "species_subset": ["species_1", "species_2", "species_3"],
        "chunk_size": 1,
        "features_filepath": None,
        "species_column": None,
        "shared_array_manager": False,
        "num_processors": 1,
        "expect_raise": True,
        "expected_return_type": None,
    },
    {
        "description": "SimilarityFromFile with non-default features_filepath",
        "similarity": "fake_similarities_file.tsv",
        "species_subset": ["species_1", "species_2", "species_3"],
        "chunk_size": 1,
        "features_filepath": "fake_features_file.tsv",
        "species_column": None,
        "shared_array_manager": False,
        "num_processors": None,
        "expect_raise": True,
        "expected_return_type": None,
    },
    {
        "description": "SimilarityFromFile with non-default species_column",
        "similarity": "fake_similarities_file.tsv",
        "species_subset": ["species_1", "species_2", "species_3"],
        "chunk_size": 1,
        "features_filepath": None,
        "species_column": "fake_species_col",
        "shared_array_manager": False,
        "num_processors": None,
        "expect_raise": True,
        "expected_return_type": None,
    },
    {
        "description": "SimilarityFromFile with non-default shared_array_manager",
        "similarity": "fake_similarities_file.tsv",
        "species_subset": ["species_1", "species_2", "species_3"],
        "chunk_size": 1,
        "features_filepath": None,
        "species_column": None,
        "shared_array_manager": True,
        "num_processors": None,
        "expect_raise": True,
        "expected_return_type": None,
    },
    {
        "description": "SimilarityFromFile with non-default num_processors",
        "similarity": "fake_similarities_file.tsv",
        "species_subset": ["species_1", "species_2", "species_3"],
        "chunk_size": 1,
        "features_filepath": None,
        "species_column": None,
        "shared_array_manager": False,
        "num_processors": 1,
        "expect_raise": True,
        "expected_return_type": None,
    },
    {
        "description": "SimilarityFromFunction with missing features_filepath",
        "similarity": sim_func,
        "species_subset": ["species_1", "species_2", "species_3"],
        "chunk_size": 1,
        "features_filepath": None,
        "species_column": "fake_species_col",
        "shared_array_manager": True,
        # Make num_processors large to avoid builtin optimization assigning precomputed reference
        "num_processors": None,
        "expect_raise": True,
        "expected_return_type": None,
    },
    {
        "description": "SimilarityFromFunction with missing species_column",
        "similarity": sim_func,
        "species_subset": ["species_1", "species_2", "species_3"],
        "chunk_size": 1,
        "features_filepath": "fake_features_file.tsv",
        "species_column": None,
        "shared_array_manager": True,
        # Make num_processors large to avoid builtin optimization assigning precomputed reference
        "num_processors": None,
        "expect_raise": True,
        "expected_return_type": None,
    },
    {
        "description": "SimilarityFromFunction with missing shared_array_manager",
        "similarity": sim_func,
        "species_subset": ["species_1", "species_2", "species_3"],
        "chunk_size": 1,
        "features_filepath": "fake_features_file.tsv",
        "species_column": "fake_species_col",
        "shared_array_manager": False,
        # Make num_processors large to avoid builtin optimization assigning precomputed reference
        "num_processors": None,
        "expect_raise": True,
        "expected_return_type": None,
    },
]


class TestMakeSimilarity:
    @fixture(params=MAKE_SIMILARITY_TEST_CASES)
    def test_case(self, request, monkeypatch, shared_array_manager):
        with monkeypatch.context() as patched_context:
            for target, mock_class in [
                ("diversity.similarity.SimilarityFromFile", MockSimilarityFromFile),
                (
                    "diversity.similarity.SimilarityFromFunction",
                    MockSimilarityFromFunction,
                ),
                ("diversity.similarity.SimilarityFromMemory", MockSimilarityFromMemory),
            ]:
                patched_context.setattr(target, mock_class)
            test_case_ = {
                key: request.param[key]
                for key in [
                    "similarity",
                    "species_subset",
                    "chunk_size",
                    "features_filepath",
                    "species_column",
                    "num_processors",
                    "expect_raise",
                    "expected_return_type",
                ]
            }
            if request.param["shared_array_manager"]:
                test_case_["shared_array_manager"] = shared_array_manager
            else:
                test_case_["shared_array_manager"] = None
            if request.param["expected_return_type"] == MockSimilarityFromFile:
                test_case_["expected_init_kwargs"] = {
                    "similarity": test_case_["similarity"],
                    "species_subset": test_case_["species_subset"],
                    "chunk_size": test_case_["chunk_size"],
                }
            elif request.param["expected_return_type"] == MockSimilarityFromFunction:
                test_case_["expected_read_shared_features_kwargs"] = {
                    "filepath": test_case_["features_filepath"],
                    "species_column": test_case_["species_column"],
                    "species_subset": test_case_["species_subset"],
                    "shared_array_manager": test_case_["shared_array_manager"],
                }
                test_case_["expected_init_kwargs"] = {
                    "similarity": test_case_["similarity"],
                    "features": test_case_["expected_read_shared_features_kwargs"],
                    "species_ordering": FAKE_SPECIES_ORDERING,
                    "shared_array_manager": test_case_["shared_array_manager"],
                    "num_processors": test_case_["num_processors"],
                }
            else:
                test_case_["expected_init_kwargs"] = {
                    "similarity": test_case_["similarity"],
                    "species_subset": test_case_["species_subset"],
                }
            yield test_case_

    def test_make_similarity(self, test_case):
        if test_case["expect_raise"]:
            make_similarity(
                similarity=test_case["similarity"],
                species_subset=test_case["species_subset"],
                chunk_size=test_case["chunk_size"],
                features_filepath=test_case["features_filepath"],
                species_column=test_case["species_column"],
                shared_array_manager=test_case["shared_array_manager"],
                num_processors=test_case["num_processors"],
            )
        else:
            similarity = make_similarity(
                similarity=test_case["similarity"],
                species_subset=test_case["species_subset"],
                chunk_size=test_case["chunk_size"],
                features_filepath=test_case["features_filepath"],
                species_column=test_case["species_column"],
                shared_array_manager=test_case["shared_array_manager"],
                num_processors=test_case["num_processors"],
            )
            assert isinstance(similarity, test_case["expected_return_type"])
            for key, arg in test_case["expected_init_kwargs"].items():
                if (
                    key == "features"
                    and test_case["expected_return_type"] == MockSimilarityFromFunction
                ):
                    for read_features_key, read_features_arg in arg.items():
                        assert (
                            read_features_arg
                            is test_case["expected_read_shared_features_kwargs"][
                                read_features_key
                            ]
                        )
                else:
                    assert similarity.kwargs[key] is arg


SIMILARITY_FROM_FILE_TEST_CASES = [
    {
        "description": "tsv file; 2 communities",
        "similarity_matrix_filepath": "similarities_file.tsv",
        "species_subset": ["species_3", "species_1", "species_2"],
        "chunk_size": 1,
        "expected_species_ordering": Index(["species_3", "species_1", "species_2"]),
        "relative_abundances": array([[1 / 1000, 1 / 100], [1 / 10, 1 / 1], [10, 100]]),
        "shared_abundances": False,
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
        "shared_out": None,
        "expect_warning": False,
    },
    {
        "description": "shared abundances",
        "similarity_matrix_filepath": "similarities_file.tsv",
        "species_subset": ["species_3", "species_1", "species_2"],
        "chunk_size": 1,
        "expected_species_ordering": Index(["species_3", "species_1", "species_2"]),
        "relative_abundances": array([[1 / 1000, 1 / 100], [1 / 10, 1 / 1], [10, 100]]),
        "shared_abundances": True,
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
        "shared_out": None,
        "expect_warning": False,
    },
    {
        "description": "numpy array out",
        "similarity_matrix_filepath": "similarities_file.tsv",
        "species_subset": ["species_3", "species_1", "species_2"],
        "chunk_size": 1,
        "expected_species_ordering": Index(["species_3", "species_1", "species_2"]),
        "relative_abundances": array([[1 / 1000, 1 / 100], [1 / 10, 1 / 1], [10, 100]]),
        "shared_abundances": False,
        "weighted_similarities": array(
            [[2.011, 20.11], [5.1001, 51.001], [10.0502, 100.502]]
        ),
        "similarities_filecontents": (
            "species_3\tspecies_1\tspecies_2\n"
            "1\t0.1\t0.2\n"
            "0.1\t1\t0.5\n"
            "0.2\t0.5\t1\n"
        ),
        "out": ones(shape=(3, 2), dtype=dtype("f8")),
        "shared_out": False,
        "expect_warning": False,
    },
    {
        "description": "shared array out",
        "similarity_matrix_filepath": "similarities_file.tsv",
        "species_subset": ["species_3", "species_1", "species_2"],
        "chunk_size": 1,
        "expected_species_ordering": Index(["species_3", "species_1", "species_2"]),
        "relative_abundances": array([[1 / 1000, 1 / 100], [1 / 10, 1 / 1], [10, 100]]),
        "shared_abundances": False,
        "weighted_similarities": array(
            [[2.011, 20.11], [5.1001, 51.001], [10.0502, 100.502]]
        ),
        "similarities_filecontents": (
            "species_3\tspecies_1\tspecies_2\n"
            "1\t0.1\t0.2\n"
            "0.1\t1\t0.5\n"
            "0.2\t0.5\t1\n"
        ),
        "out": ones(shape=(3, 2), dtype=dtype("f8")),
        "shared_out": True,
        "expect_warning": False,
    },
    {
        "description": "csv file; 2 communities",
        "similarity_matrix_filepath": "similarities_file.csv",
        "species_subset": ["species_3", "species_1", "species_2"],
        "chunk_size": 1,
        "expected_species_ordering": Index(["species_3", "species_1", "species_2"]),
        "relative_abundances": array([[1 / 1000, 1 / 100], [1 / 10, 1 / 1], [10, 100]]),
        "shared_abundances": False,
        "weighted_similarities": array(
            [[2.011, 20.11], [5.1001, 51.001], [10.0502, 100.502]]
        ),
        "similarities_filecontents": (
            "species_3,species_1,species_2\n" "1,0.1,0.2\n" "0.1,1,0.5\n" "0.2,0.5,1\n"
        ),
        "out": None,
        "shared_out": None,
        "expect_warning": False,
    },
    {
        "description": "no file extension; 1 community",
        "similarity_matrix_filepath": "similarities_file",
        "species_subset": ["species_3", "species_1", "species_2"],
        "chunk_size": 1,
        "expected_species_ordering": Index(["species_3", "species_1", "species_2"]),
        "relative_abundances": array([[1 / 1000], [1 / 10], [10]]),
        "shared_abundances": False,
        "weighted_similarities": array([[2.011], [5.1001], [10.0502]]),
        "similarities_filecontents": (
            "species_3\tspecies_1\tspecies_2\n"
            "1\t0.1\t0.2\n"
            "0.1\t1\t0.5\n"
            "0.2\t0.5\t1\n"
        ),
        "out": None,
        "shared_out": None,
        "expect_warning": True,
    },
    {
        "description": "species subset",
        "similarity_matrix_filepath": "similarities_file.tsv",
        "species_subset": ["species_1", "species_3"],
        "chunk_size": 1,
        "expected_species_ordering": Index(["species_3", "species_1"]),
        "relative_abundances": array([[1 / 1000, 1 / 100], [10, 100]]),
        "shared_abundances": False,
        "weighted_similarities": array([[1.001, 10.01], [10.0001, 100.001]]),
        "similarities_filecontents": (
            "species_3\tspecies_1\tspecies_2\n"
            "1\t0.1\t0.2\n"
            "0.1\t1\t0.5\n"
            "0.2\t0.5\t1\n"
        ),
        "out": None,
        "shared_out": None,
        "expect_warning": False,
    },
    {
        "description": "species subset; shared abundances",
        "similarity_matrix_filepath": "similarities_file.tsv",
        "species_subset": ["species_1", "species_3"],
        "chunk_size": 1,
        "expected_species_ordering": Index(["species_3", "species_1"]),
        "relative_abundances": array([[1 / 1000, 1 / 100], [10, 100]]),
        "shared_abundances": True,
        "weighted_similarities": array([[1.001, 10.01], [10.0001, 100.001]]),
        "similarities_filecontents": (
            "species_3\tspecies_1\tspecies_2\n"
            "1\t0.1\t0.2\n"
            "0.1\t1\t0.5\n"
            "0.2\t0.5\t1\n"
        ),
        "out": None,
        "shared_out": None,
        "expect_warning": False,
    },
    {
        "description": "species subset; numpy array out",
        "similarity_matrix_filepath": "similarities_file.tsv",
        "species_subset": ["species_1", "species_3"],
        "chunk_size": 1,
        "expected_species_ordering": Index(["species_3", "species_1"]),
        "relative_abundances": array([[1 / 1000, 1 / 100], [10, 100]]),
        "shared_abundances": False,
        "weighted_similarities": array([[1.001, 10.01], [10.0001, 100.001]]),
        "similarities_filecontents": (
            "species_3\tspecies_1\tspecies_2\n"
            "1\t0.1\t0.2\n"
            "0.1\t1\t0.5\n"
            "0.2\t0.5\t1\n"
        ),
        "out": ones(shape=(2, 2), dtype=dtype("f8")),
        "shared_out": False,
        "expect_warning": False,
    },
    {
        "description": "species subset; shared array out",
        "similarity_matrix_filepath": "similarities_file.tsv",
        "species_subset": ["species_1", "species_3"],
        "chunk_size": 1,
        "expected_species_ordering": Index(["species_3", "species_1"]),
        "relative_abundances": array([[1 / 1000, 1 / 100], [10, 100]]),
        "shared_abundances": False,
        "weighted_similarities": array([[1.001, 10.01], [10.0001, 100.001]]),
        "similarities_filecontents": (
            "species_3\tspecies_1\tspecies_2\n"
            "1\t0.1\t0.2\n"
            "0.1\t1\t0.5\n"
            "0.2\t0.5\t1\n"
        ),
        "out": ones(shape=(2, 2), dtype=dtype("f8")),
        "shared_out": True,
        "expect_warning": False,
    },
    {
        "description": "non-default chunk_size",
        "similarity_matrix_filepath": "similarities_file.tsv",
        "species_subset": ["species_3", "species_1", "species_2"],
        "chunk_size": 12,
        "expected_species_ordering": Index(["species_3", "species_1", "species_2"]),
        "relative_abundances": array([[1 / 1000, 1 / 100], [1 / 10, 1 / 1], [10, 100]]),
        "shared_abundances": False,
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
        "shared_out": None,
        "expect_warning": False,
    },
    {
        "description": "non-default chunk_size; shared abundances",
        "similarity_matrix_filepath": "similarities_file.tsv",
        "species_subset": ["species_3", "species_1", "species_2"],
        "chunk_size": 12,
        "expected_species_ordering": Index(["species_3", "species_1", "species_2"]),
        "relative_abundances": array([[1 / 1000, 1 / 100], [1 / 10, 1 / 1], [10, 100]]),
        "shared_abundances": True,
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
        "shared_out": None,
        "expect_warning": False,
    },
    {
        "description": "non-default chunk_size; numpy array out",
        "similarity_matrix_filepath": "similarities_file.tsv",
        "species_subset": ["species_3", "species_1", "species_2"],
        "chunk_size": 12,
        "expected_species_ordering": Index(["species_3", "species_1", "species_2"]),
        "relative_abundances": array([[1 / 1000, 1 / 100], [1 / 10, 1 / 1], [10, 100]]),
        "shared_abundances": False,
        "weighted_similarities": array(
            [[2.011, 20.11], [5.1001, 51.001], [10.0502, 100.502]]
        ),
        "similarities_filecontents": (
            "species_3\tspecies_1\tspecies_2\n"
            "1\t0.1\t0.2\n"
            "0.1\t1\t0.5\n"
            "0.2\t0.5\t1\n"
        ),
        "out": ones(shape=(3, 2), dtype=dtype("f8")),
        "shared_out": False,
        "expect_warning": False,
    },
    {
        "description": "non-default chunk_size; shared array out",
        "similarity_matrix_filepath": "similarities_file.tsv",
        "species_subset": ["species_3", "species_1", "species_2"],
        "chunk_size": 12,
        "expected_species_ordering": Index(["species_3", "species_1", "species_2"]),
        "relative_abundances": array([[1 / 1000, 1 / 100], [1 / 10, 1 / 1], [10, 100]]),
        "shared_abundances": False,
        "weighted_similarities": array(
            [[2.011, 20.11], [5.1001, 51.001], [10.0502, 100.502]]
        ),
        "similarities_filecontents": (
            "species_3\tspecies_1\tspecies_2\n"
            "1\t0.1\t0.2\n"
            "0.1\t1\t0.5\n"
            "0.2\t0.5\t1\n"
        ),
        "out": ones(shape=(3, 2), dtype=dtype("f8")),
        "shared_out": True,
        "expect_warning": False,
    },
]


class TestSimilarityFromFile:
    """Tests metacommunity.Similarity."""

    @fixture(params=SIMILARITY_FROM_FILE_TEST_CASES)
    def test_case(self, request, tmp_path, shared_array_manager):
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
                "weighted_similarities",
                "similarities_filecontents",
                "shared_out",
                "expect_warning",
            ]
        }
        test_case_["similarity"] = filepath
        if request.param["shared_abundances"]:
            test_case_["relative_abundances"] = shared_array_manager.from_array(
                request.param["relative_abundances"]
            )
        else:
            test_case_["relative_abundances"] = request.param["relative_abundances"]
        if request.param["out"] is None:
            test_case_["out"] = None
        elif request.param["shared_out"]:
            test_case_["out"] = shared_array_manager.from_array(request.param["out"])
        else:
            test_case_["out"] = request.param["out"]
        return test_case_

    def test_init(self, test_case):
        """Tests initializer."""
        if test_case["expect_warning"]:
            with warns(ArgumentWarning):
                similarity = SimilarityFromFile(
                    similarity=test_case["similarity"],
                    species_subset=test_case["species_subset"],
                    chunk_size=test_case["chunk_size"],
                )
        else:
            similarity = SimilarityFromFile(
                similarity=test_case["similarity"],
                species_subset=test_case["species_subset"],
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
                    species_subset=test_case["species_subset"],
                    chunk_size=test_case["chunk_size"],
                )
        else:
            similarity = SimilarityFromFile(
                similarity=test_case["similarity"],
                species_subset=test_case["species_subset"],
                chunk_size=test_case["chunk_size"],
            )
        weighted_similarities = similarity.calculate_weighted_similarities(
            relative_abundances=test_case["relative_abundances"],
            out=test_case["out"],
        )
        assert weighted_similarities.shape == test_case["weighted_similarities"].shape
        assert allclose(weighted_similarities, test_case["weighted_similarities"])
        with open(test_case["similarity"], "r") as file:
            similarities_filecontents = file.read()
        assert similarities_filecontents == test_case["similarities_filecontents"]
        if test_case["out"] is not None:
            if test_case["shared_out"]:
                assert weighted_similarities is test_case["out"].data
            else:
                assert weighted_similarities is test_case["out"]


SIMILARITY_FROM_FUNCTION_TEST_CASES = [
    {
        "description": "2 communities; 2 features; default num_processors",
        "similarity_function": sim_func,  # lambda a, b: 1 / sum(a * b)
        "features": array([[1, 2], [3, 5], [7, 11]]),
        "species_ordering": Index(["species_1", "species_2", "species_3"]),
        "num_processors": None,
        "expected_species_ordering": Index(["species_1", "species_2", "species_3"]),
        "relative_abundances": array([[1 / 1000, 1 / 100], [1 / 10, 1 / 1], [10, 100]]),
        "shared_abundances": False,
        "expected_weighted_similarities": array(
            [
                [0.35271989, 3.52719894],
                [0.13459705, 1.34597047],
                [0.0601738, 0.60173802],
            ]
        ),
        "out": None,
        "shared_out": None,
        "expect_raise": False,
        # similarity matrix
        # [0.2          , 0.07692307692, 0.03448275862]
        # [0.07692307692, 0.02941176471, 0.01315789474]
        # [0.03448275862, 0.01315789474, 0.005882352941]
    },
    {
        "description": "shared abundances, default num_processors",
        "similarity_function": sim_func,  # lambda a, b: 1 / sum(a * b)
        "features": array([[1, 2], [3, 5], [7, 11]]),
        "species_ordering": Index(["species_1", "species_2", "species_3"]),
        "num_processors": None,
        "expected_species_ordering": Index(["species_1", "species_2", "species_3"]),
        "relative_abundances": array([[1 / 1000, 1 / 100], [1 / 10, 1 / 1], [10, 100]]),
        "shared_abundances": True,
        "expected_weighted_similarities": array(
            [
                [0.35271989, 3.52719894],
                [0.13459705, 1.34597047],
                [0.0601738, 0.60173802],
            ]
        ),
        "out": None,
        "shared_out": None,
        "expect_raise": False,
        # similarity matrix
        # [0.2          , 0.07692307692, 0.03448275862]
        # [0.07692307692, 0.02941176471, 0.01315789474]
        # [0.03448275862, 0.01315789474, 0.005882352941]
    },
    {
        "description": "numpy array out; default num_processors",
        "similarity_function": sim_func,  # lambda a, b: 1 / sum(a * b)
        "features": array([[1, 2], [3, 5], [7, 11]]),
        "species_ordering": Index(["species_1", "species_2", "species_3"]),
        "num_processors": None,
        "expected_species_ordering": Index(["species_1", "species_2", "species_3"]),
        "relative_abundances": array([[1 / 1000, 1 / 100], [1 / 10, 1 / 1], [10, 100]]),
        "shared_abundances": False,
        "expected_weighted_similarities": array(
            [
                [0.35271989, 3.52719894],
                [0.13459705, 1.34597047],
                [0.0601738, 0.60173802],
            ]
        ),
        "out": ones(shape=(3, 2), dtype=dtype("f8")),
        "shared_out": False,
        "expect_raise": False,
        # similarity matrix
        # [0.2          , 0.07692307692, 0.03448275862]
        # [0.07692307692, 0.02941176471, 0.01315789474]
        # [0.03448275862, 0.01315789474, 0.005882352941]
    },
    {
        "description": "shared array out; default num_processors",
        "similarity_function": sim_func,  # lambda a, b: 1 / sum(a * b)
        "features": array([[1, 2], [3, 5], [7, 11]]),
        "species_ordering": Index(["species_1", "species_2", "species_3"]),
        "num_processors": None,
        "expected_species_ordering": Index(["species_1", "species_2", "species_3"]),
        "relative_abundances": array([[1 / 1000, 1 / 100], [1 / 10, 1 / 1], [10, 100]]),
        "shared_abundances": False,
        "expected_weighted_similarities": array(
            [
                [0.35271989, 3.52719894],
                [0.13459705, 1.34597047],
                [0.0601738, 0.60173802],
            ]
        ),
        "out": ones(shape=(3, 2), dtype=dtype("f8")),
        "shared_out": True,
        "expect_raise": False,
        # similarity matrix
        # [0.2          , 0.07692307692, 0.03448275862]
        # [0.07692307692, 0.02941176471, 0.01315789474]
        # [0.03448275862, 0.01315789474, 0.005882352941]
    },
    {
        "description": "2 communities; 2 features; num_processors=1",
        "similarity_function": sim_func,  # lambda a, b: 1 / sum(a * b)
        "features": array([[1, 2], [3, 5], [7, 11]]),
        "species_ordering": Index(["species_1", "species_2", "species_3"]),
        "num_processors": 1,
        "expected_species_ordering": Index(["species_1", "species_2", "species_3"]),
        "relative_abundances": array([[1 / 1000, 1 / 100], [1 / 10, 1 / 1], [10, 100]]),
        "shared_abundances": False,
        "expected_weighted_similarities": array(
            [
                [0.35271989, 3.52719894],
                [0.13459705, 1.34597047],
                [0.0601738, 0.60173802],
            ]
        ),
        "out": None,
        "shared_out": None,
        "expect_raise": False,
        # similarity matrix
        # [0.2          , 0.07692307692, 0.03448275862]
        # [0.07692307692, 0.02941176471, 0.01315789474]
        # [0.03448275862, 0.01315789474, 0.005882352941]
    },
    {
        "description": "shared abundances; num_processors=1",
        "similarity_function": sim_func,  # lambda a, b: 1 / sum(a * b)
        "features": array([[1, 2], [3, 5], [7, 11]]),
        "species_ordering": Index(["species_1", "species_2", "species_3"]),
        "num_processors": 1,
        "expected_species_ordering": Index(["species_1", "species_2", "species_3"]),
        "relative_abundances": array([[1 / 1000, 1 / 100], [1 / 10, 1 / 1], [10, 100]]),
        "shared_abundances": True,
        "expected_weighted_similarities": array(
            [
                [0.35271989, 3.52719894],
                [0.13459705, 1.34597047],
                [0.0601738, 0.60173802],
            ]
        ),
        "out": None,
        "shared_out": None,
        "expect_raise": False,
        # similarity matrix
        # [0.2          , 0.07692307692, 0.03448275862]
        # [0.07692307692, 0.02941176471, 0.01315789474]
        # [0.03448275862, 0.01315789474, 0.005882352941]
    },
    {
        "description": "numpy array out; num_processors=1",
        "similarity_function": sim_func,  # lambda a, b: 1 / sum(a * b)
        "features": array([[1, 2], [3, 5], [7, 11]]),
        "species_ordering": Index(["species_1", "species_2", "species_3"]),
        "num_processors": 1,
        "expected_species_ordering": Index(["species_1", "species_2", "species_3"]),
        "relative_abundances": array([[1 / 1000, 1 / 100], [1 / 10, 1 / 1], [10, 100]]),
        "shared_abundances": True,
        "expected_weighted_similarities": array(
            [
                [0.35271989, 3.52719894],
                [0.13459705, 1.34597047],
                [0.0601738, 0.60173802],
            ]
        ),
        "out": ones(shape=(3, 2), dtype=dtype("f8")),
        "shared_out": False,
        "expect_raise": False,
        # similarity matrix
        # [0.2          , 0.07692307692, 0.03448275862]
        # [0.07692307692, 0.02941176471, 0.01315789474]
        # [0.03448275862, 0.01315789474, 0.005882352941]
    },
    {
        "description": "shared array out; num_processors=1",
        "similarity_function": sim_func,  # lambda a, b: 1 / sum(a * b)
        "features": array([[1, 2], [3, 5], [7, 11]]),
        "species_ordering": Index(["species_1", "species_2", "species_3"]),
        "num_processors": 1,
        "expected_species_ordering": Index(["species_1", "species_2", "species_3"]),
        "relative_abundances": array([[1 / 1000, 1 / 100], [1 / 10, 1 / 1], [10, 100]]),
        "shared_abundances": True,
        "expected_weighted_similarities": array(
            [
                [0.35271989, 3.52719894],
                [0.13459705, 1.34597047],
                [0.0601738, 0.60173802],
            ]
        ),
        "out": ones(shape=(3, 2), dtype=dtype("f8")),
        "shared_out": True,
        "expect_raise": False,
        # similarity matrix
        # [0.2          , 0.07692307692, 0.03448275862]
        # [0.07692307692, 0.02941176471, 0.01315789474]
        # [0.03448275862, 0.01315789474, 0.005882352941]
    },
    {
        "description": "2 communities; 2 features; num_processors=2",
        "similarity_function": sim_func,  # lambda a, b: 1 / sum(a * b)
        "features": array([[1, 2], [3, 5], [7, 11]]),
        "species_ordering": Index(["species_1", "species_2", "species_3"]),
        "num_processors": 2,
        "expected_species_ordering": Index(["species_1", "species_2", "species_3"]),
        "relative_abundances": array([[1 / 1000, 1 / 100], [1 / 10, 1 / 1], [10, 100]]),
        "shared_abundances": False,
        "expected_weighted_similarities": array(
            [
                [0.35271989, 3.52719894],
                [0.13459705, 1.34597047],
                [0.0601738, 0.60173802],
            ]
        ),
        "out": None,
        "shared_out": None,
        "expect_raise": False,
        # similarity matrix
        # [0.2          , 0.07692307692, 0.03448275862]
        # [0.07692307692, 0.02941176471, 0.01315789474]
        # [0.03448275862, 0.01315789474, 0.005882352941]
    },
    {
        "description": "shared relative_abundances; num_processors=2",
        "similarity_function": sim_func,  # lambda a, b: 1 / sum(a * b)
        "features": array([[1, 2], [3, 5], [7, 11]]),
        "species_ordering": Index(["species_1", "species_2", "species_3"]),
        "num_processors": 2,
        "expected_species_ordering": Index(["species_1", "species_2", "species_3"]),
        "relative_abundances": array([[1 / 1000, 1 / 100], [1 / 10, 1 / 1], [10, 100]]),
        "shared_abundances": True,
        "expected_weighted_similarities": array(
            [
                [0.35271989, 3.52719894],
                [0.13459705, 1.34597047],
                [0.0601738, 0.60173802],
            ]
        ),
        "out": None,
        "shared_out": None,
        "expect_raise": False,
        # similarity matrix
        # [0.2          , 0.07692307692, 0.03448275862]
        # [0.07692307692, 0.02941176471, 0.01315789474]
        # [0.03448275862, 0.01315789474, 0.005882352941]
    },
    {
        "description": "numpy array out; num_processors=2",
        "similarity_function": sim_func,  # lambda a, b: 1 / sum(a * b)
        "features": array([[1, 2], [3, 5], [7, 11]]),
        "species_ordering": Index(["species_1", "species_2", "species_3"]),
        "num_processors": 2,
        "expected_species_ordering": Index(["species_1", "species_2", "species_3"]),
        "relative_abundances": array([[1 / 1000, 1 / 100], [1 / 10, 1 / 1], [10, 100]]),
        "shared_abundances": False,
        "expected_weighted_similarities": array(
            [
                [0.35271989, 3.52719894],
                [0.13459705, 1.34597047],
                [0.0601738, 0.60173802],
            ]
        ),
        "out": ones(shape=(3, 2), dtype=dtype("f8")),
        "shared_out": False,
        "expect_raise": False,
        # similarity matrix
        # [0.2          , 0.07692307692, 0.03448275862]
        # [0.07692307692, 0.02941176471, 0.01315789474]
        # [0.03448275862, 0.01315789474, 0.005882352941]
    },
    {
        "description": "shared array out; num_processors=2",
        "similarity_function": sim_func,  # lambda a, b: 1 / sum(a * b)
        "features": array([[1, 2], [3, 5], [7, 11]]),
        "species_ordering": Index(["species_1", "species_2", "species_3"]),
        "num_processors": 2,
        "expected_species_ordering": Index(["species_1", "species_2", "species_3"]),
        "relative_abundances": array([[1 / 1000, 1 / 100], [1 / 10, 1 / 1], [10, 100]]),
        "shared_abundances": False,
        "expected_weighted_similarities": array(
            [
                [0.35271989, 3.52719894],
                [0.13459705, 1.34597047],
                [0.0601738, 0.60173802],
            ]
        ),
        "out": ones(shape=(3, 2), dtype=dtype("f8")),
        "shared_out": True,
        "expect_raise": False,
        # similarity matrix
        # [0.2          , 0.07692307692, 0.03448275862]
        # [0.07692307692, 0.02941176471, 0.01315789474]
        # [0.03448275862, 0.01315789474, 0.005882352941]
    },
    {
        "description": "2 communities; 2 features; num_processors>num_species",
        "similarity_function": sim_func,  # lambda a, b: 1 / sum(a * b)
        "features": array([[1, 2], [3, 5], [7, 11]]),
        "species_ordering": Index(["species_1", "species_2", "species_3"]),
        "num_processors": 4,
        "expected_species_ordering": Index(["species_1", "species_2", "species_3"]),
        "relative_abundances": array([[1 / 1000, 1 / 100], [1 / 10, 1 / 1], [10, 100]]),
        "shared_abundances": False,
        "expected_weighted_similarities": array(
            [
                [0.35271989, 3.52719894],
                [0.13459705, 1.34597047],
                [0.0601738, 0.60173802],
            ]
        ),
        "out": None,
        "shared_out": None,
        "expect_raise": False,
        # similarity matrix
        # [0.2          , 0.07692307692, 0.03448275862]
        # [0.07692307692, 0.02941176471, 0.01315789474]
        # [0.03448275862, 0.01315789474, 0.005882352941]
    },
    {
        "description": "2 communities; 2 features; num_processors>cpu_count",
        "similarity_function": sim_func,  # lambda a, b: 1 / sum(a * b)
        "features": array([[1, 2], [3, 5], [7, 11]]),
        "species_ordering": Index(["species_1", "species_2", "species_3"]),
        "num_processors": cpu_count() + 1,
        "expected_species_ordering": Index(["species_1", "species_2", "species_3"]),
        "relative_abundances": array([[1 / 1000, 1 / 100], [1 / 10, 1 / 1], [10, 100]]),
        "shared_abundances": False,
        "expected_weighted_similarities": array(
            [
                [0.35271989, 3.52719894],
                [0.13459705, 1.34597047],
                [0.0601738, 0.60173802],
            ]
        ),
        "out": None,
        "shared_out": None,
        "expect_raise": False,
        # similarity matrix
        # [0.2          , 0.07692307692, 0.03448275862]
        # [0.07692307692, 0.02941176471, 0.01315789474]
        # [0.03448275862, 0.01315789474, 0.005882352941]
    },
    {
        "description": "2 communities; 1 feature; default num_processors",
        "similarity_function": sim_func,  # lambda a, b: 1 / sum(a * b)
        "features": array([[1], [3], [7]]),
        "species_ordering": Index(["species_1", "species_2", "species_3"]),
        "num_processors": None,
        "expected_species_ordering": Index(["species_1", "species_2", "species_3"]),
        "relative_abundances": array([[1 / 1000, 1 / 100], [1 / 10, 1 / 1], [10, 100]]),
        "shared_abundances": False,
        "expected_weighted_similarities": array(
            [
                [1.46290476, 14.62904762],
                [0.48763492, 4.87634921],
                [0.20898639, 2.08986395],
            ]
        ),
        "out": None,
        "shared_out": None,
        "expect_raise": False,
        # similarity matrix
        # [1,   1/3,  1/7]
        # [1/3, 1/9,  1/21]
        # [1/7, 1/21, 1/49]
    },
    {
        "description": "1 community; 2 features; default num_processors",
        "similarity_function": sim_func,  # lambda a, b: 1 / sum(a * b)
        "features": array([[1, 2], [3, 5], [7, 11]]),
        "species_ordering": Index(["species_1", "species_2", "species_3"]),
        "num_processors": None,
        "expected_species_ordering": Index(["species_1", "species_2", "species_3"]),
        "relative_abundances": array([[1 / 1000], [1 / 10], [10]]),
        "shared_abundances": False,
        "expected_weighted_similarities": array(
            [[0.35271989], [0.13459705], [0.0601738]]
        ),
        "out": None,
        "shared_out": None,
        "expect_raise": False,
        # similarity matrix
        # [0.2          , 0.07692307692, 0.03448275862]
        # [0.07692307692, 0.02941176471, 0.01315789474]
        # [0.03448275862, 0.01315789474, 0.005882352941]
    },
    {
        "description": "1 community; 1 feature; default num_processors",
        "similarity_function": sim_func,  # lambda a, b: 1 / sum(a * b)
        "features": array([[1], [3], [7]]),
        "species_ordering": Index(["species_1", "species_2", "species_3"]),
        "num_processors": None,
        "expected_species_ordering": Index(["species_1", "species_2", "species_3"]),
        "relative_abundances": array([[1 / 1000], [1 / 10], [10]]),
        "shared_abundances": False,
        "expected_weighted_similarities": array(
            [
                [1.46290476],
                [0.48763492],
                [0.20898639],
            ]
        ),
        "out": None,
        "shared_out": None,
        "expect_raise": False,
        # similarity matrix
        # [1,   1/3,  1/7]
        # [1/3, 1/9,  1/21]
        # [1/7, 1/21, 1/49]
    },
    {
        "description": "species superset",
        "similarity_function": sim_func,  # lambda a, b: 1 / sum(a * b)
        "features": array([[0, 1], [1, 2], [3, 2]]),
        "species_ordering": Index(
            [
                "species_0",
                "species_1",
                "species_1b",
                "species_2",
                "species_3",
                "species_4",
            ]
        ),
        "num_processors": None,
        "expected_species_ordering": None,
        "relative_abundances": None,
        "shared_abundances": None,
        "expected_weighted_similarities": None,
        "out": None,
        "shared_out": None,
        "expect_raise": True,
        # similarity matrix
        # [0.2          , 0.07692307692, 0.03448275862]
        # [0.07692307692, 0.02941176471, 0.01315789474]
        # [0.03448275862, 0.01315789474, 0.005882352941]
    },
    {
        "description": "species subset",
        "similarity_function": sim_func,  # lambda a, b: 1 / sum(a * b)
        "features": array([[0, 1], [1, 2], [3, 2], [3, 5], [7, 11], [13, 17]]),
        "species_ordering": Index(["species_1", "species_2", "species_3"]),
        "num_processors": None,
        "expected_species_ordering": None,
        "relative_abundances": None,
        "shared_abundances": None,
        "expected_weighted_similarities": None,
        "out": None,
        "shared_out": None,
        "expect_raise": True,
        # similarity matrix
        # [0.2          , 0.07692307692, 0.03448275862]
        # [0.07692307692, 0.02941176471, 0.01315789474]
        #
    },
]


class TestSimilarityFromFunction:
    """Tests metacommunity.Similarity."""

    @fixture(params=SIMILARITY_FROM_FUNCTION_TEST_CASES)
    def test_case(self, request, shared_array_manager):
        features = shared_array_manager.from_array(request.param["features"])
        test_case_ = {
            key: request.param[key]
            for key in [
                "description",
                "similarity_function",
                "species_ordering",
                "num_processors",
                "shared_out",
                "expected_species_ordering",
                "expected_weighted_similarities",
                "expect_raise",
            ]
        }
        if request.param["shared_abundances"]:
            test_case_["relative_abundances"] = shared_array_manager.from_array(
                request.param["relative_abundances"]
            )
        else:
            test_case_["relative_abundances"] = request.param["relative_abundances"]
        if request.param["out"] is None:
            test_case_["out"] = None
        elif request.param["shared_out"]:
            test_case_["out"] = shared_array_manager.from_array(request.param["out"])
        else:
            test_case_["out"] = request.param["out"]
        test_case_.update(
            {
                "features": features,
                "shared_array_manager": shared_array_manager,
            }
        )
        return test_case_

    def test_init(self, test_case, tmp_path):
        """Tests initializer."""
        if test_case["expect_raise"]:
            with raises(InvalidArgumentError):
                SimilarityFromFunction(
                    similarity=test_case["similarity_function"],
                    features=test_case["features"],
                    species_ordering=test_case["species_ordering"],
                    shared_array_manager=test_case["shared_array_manager"],
                    num_processors=test_case["num_processors"],
                )
        else:
            similarity = SimilarityFromFunction(
                similarity=test_case["similarity_function"],
                features=test_case["features"],
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
            similarity=test_case["similarity_function"],
            features=test_case["features"],
            species_ordering=test_case["species_ordering"],
            shared_array_manager=test_case["shared_array_manager"],
            num_processors=test_case["num_processors"],
        )
        weighted_similarities = similarity.calculate_weighted_similarities(
            relative_abundances=test_case["relative_abundances"],
            out=test_case["out"],
        )
        assert (
            weighted_similarities.data.shape
            == test_case["expected_weighted_similarities"].shape
        )
        assert allclose(
            weighted_similarities.data, test_case["expected_weighted_similarities"]
        )
        if test_case["out"] is not None:
            if test_case["shared_out"]:
                assert weighted_similarities is test_case["out"].data
            else:
                assert weighted_similarities is test_case["out"]


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
            func=test_case["func"]
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
        "species_subset": ["species_1", "species_2", "species_3"],
        "expected_species_ordering": Index(["species_1", "species_2", "species_3"]),
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
        "shared_abundances": False,
        "out": None,
        "shared_out": None,
        "weighted_similarities": array(
            [[1.051, 10.51], [2.1005, 21.005], [10.0201, 100.201]]
        ),
    },
    {
        "description": "shared abundances",
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
        "species_subset": ["species_1", "species_2", "species_3"],
        "expected_species_ordering": Index(["species_1", "species_2", "species_3"]),
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
        "shared_abundances": True,
        "out": None,
        "shared_out": None,
        "weighted_similarities": array(
            [[1.051, 10.51], [2.1005, 21.005], [10.0201, 100.201]]
        ),
    },
    {
        "description": "numpy array out",
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
        "species_subset": ["species_1", "species_2", "species_3"],
        "expected_species_ordering": Index(["species_1", "species_2", "species_3"]),
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
        "shared_abundances": False,
        "out": ones(shape=(3, 2), dtype=dtype("f8")),
        "shared_out": False,
        "weighted_similarities": array(
            [[1.051, 10.51], [2.1005, 21.005], [10.0201, 100.201]]
        ),
    },
    {
        "description": "shared array out",
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
        "species_subset": ["species_1", "species_2", "species_3"],
        "expected_species_ordering": Index(["species_1", "species_2", "species_3"]),
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
        "shared_abundances": False,
        "out": ones(shape=(3, 2), dtype=dtype("f8")),
        "shared_out": True,
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
        "expected_species_ordering": Index(["species_1", "species_2", "species_3"]),
        "relative_abundances": array([[1 / 1000], [1 / 10], [10]]),
        "shared_abundances": False,
        "out": None,
        "shared_out": None,
        "weighted_similarities": array([[1.051], [2.1005], [10.0201]]),
    },
    {
        "description": "2 communities; shuffled index",
        "similarity": DataFrame(
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
        "expected_species_ordering": Index(["species_1", "species_2", "species_3"]),
        "relative_abundances": array([[1 / 1000, 1 / 100], [1 / 10, 1 / 1], [10, 100]]),
        "shared_abundances": False,
        "out": None,
        "shared_out": None,
        "weighted_similarities": array(
            [[1.051, 10.51], [2.1005, 21.005], [10.0201, 100.201]]
        ),
    },
    {
        "description": "shared abundances; shuffled index",
        "similarity": DataFrame(
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
        "expected_species_ordering": Index(["species_1", "species_2", "species_3"]),
        "relative_abundances": array([[1 / 1000, 1 / 100], [1 / 10, 1 / 1], [10, 100]]),
        "shared_abundances": True,
        "out": None,
        "shared_out": None,
        "weighted_similarities": array(
            [[1.051, 10.51], [2.1005, 21.005], [10.0201, 100.201]]
        ),
    },
    {
        "description": "numpy array out; shuffled index",
        "similarity": DataFrame(
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
        "expected_species_ordering": Index(["species_1", "species_2", "species_3"]),
        "relative_abundances": array([[1 / 1000, 1 / 100], [1 / 10, 1 / 1], [10, 100]]),
        "shared_abundances": False,
        "out": ones(shape=(3, 2), dtype=dtype("f8")),
        "shared_out": False,
        "weighted_similarities": array(
            [[1.051, 10.51], [2.1005, 21.005], [10.0201, 100.201]]
        ),
    },
    {
        "description": "shared array out; shuffled index",
        "similarity": DataFrame(
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
        "expected_species_ordering": Index(["species_1", "species_2", "species_3"]),
        "relative_abundances": array([[1 / 1000, 1 / 100], [1 / 10, 1 / 1], [10, 100]]),
        "shared_abundances": False,
        "out": ones(shape=(3, 2), dtype=dtype("f8")),
        "shared_out": True,
        "weighted_similarities": array(
            [[1.051, 10.51], [2.1005, 21.005], [10.0201, 100.201]]
        ),
    },
    {
        "description": "1 community; shuffled index",
        "similarity": DataFrame(
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
        "expected_species_ordering": Index(["species_1", "species_2", "species_3"]),
        "relative_abundances": array([[1 / 1000], [1 / 10], [10]]),
        "shared_abundances": False,
        "out": None,
        "shared_out": None,
        "weighted_similarities": array([[1.051], [2.1005], [10.0201]]),
    },
    {
        "description": "species subset",
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
        "shared_abundances": False,
        "out": None,
        "shared_out": None,
        "weighted_similarities": array([[2.1, 21.0], [10.02, 100.2]]),
    },
    {
        "description": "shared abundances; species subset",
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
        "shared_abundances": True,
        "out": None,
        "shared_out": None,
        "weighted_similarities": array([[2.1, 21.0], [10.02, 100.2]]),
    },
    {
        "description": "numpy array out; species subset",
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
        "shared_abundances": False,
        "out": ones(shape=(2, 2), dtype=dtype("f8")),
        "shared_out": False,
        "weighted_similarities": array([[2.1, 21.0], [10.02, 100.2]]),
    },
    {
        "description": "shared array out; species subset",
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
        "shared_abundances": False,
        "out": ones(shape=(2, 2), dtype=dtype("f8")),
        "shared_out": True,
        "weighted_similarities": array([[2.1, 21.0], [10.02, 100.2]]),
    },
]


class TestSimilarityFromMemory:
    """Tests metacommunity.Similarity."""

    @fixture(params=SIMILARITY_FROM_MEMORY_TEST_CASES)
    def test_case(self, request, shared_array_manager):
        test_case_ = {
            key: request.param[key]
            for key in [
                "similarity",
                "species_subset",
                "expected_species_ordering",
                "expected_similarity_matrix",
                "shared_out",
                "weighted_similarities",
            ]
        }
        if request.param["shared_abundances"]:
            test_case_["relative_abundances"] = shared_array_manager.from_array(
                request.param["relative_abundances"]
            )
        else:
            test_case_["relative_abundances"] = request.param["relative_abundances"]
        if request.param["out"] is None:
            test_case_["out"] = None
        elif request.param["shared_out"]:
            test_case_["out"] = shared_array_manager.from_array(request.param["out"])
        else:
            test_case_["out"] = request.param["out"]
        return test_case_

    def test_init(self, test_case):
        """Tests initializer."""
        similarity = SimilarityFromMemory(
            similarity=test_case["similarity"],
            species_subset=test_case["species_subset"],
        )
        assert array_equal(
            similarity.species_ordering, test_case["expected_species_ordering"]
        )
        assert_frame_equal(
            similarity.similarity, test_case["expected_similarity_matrix"]
        )

    def test_calculate_weighted_similarities(self, test_case, tmp_path):
        """Tests .calculate_weighted_similarities."""
        similarity = SimilarityFromMemory(
            similarity=test_case["similarity"],
            species_subset=test_case["species_subset"],
        )
        weighted_similarities = similarity.calculate_weighted_similarities(
            relative_abundances=test_case["relative_abundances"], out=test_case["out"]
        )
        assert weighted_similarities.shape == test_case["weighted_similarities"].shape
        assert allclose(weighted_similarities, test_case["weighted_similarities"])
        if test_case["out"] is not None:
            if test_case["shared_out"]:
                assert weighted_similarities is test_case["out"].data
            else:
                assert weighted_similarities is test_case["out"]
