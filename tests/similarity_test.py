"""Tests for diversity.similarity"""
from numpy import allclose, array, dtype, memmap
from pandas import DataFrame
from pytest import fixture, warns, raises, mark

from diversity.exceptions import ArgumentWarning
from diversity.log import LOGGER
from diversity.similarity import (
    SimilarityFromArray,
    SimilarityFromDataFrame,
    SimilarityFromFile,
    SimilarityFromFunction,
    make_similarity,
)


def similarity_function(a, b):
    return 1 / sum(a * b)


similarity_array_3x3 = array(
    [
        [1, 0.5, 0.1],
        [0.5, 1, 0.2],
        [0.1, 0.2, 1],
    ]
)
similarity_array_3x3_2 = array(
    [
        [1.0, 0.9, 0.1],
        [0.9, 1.0, 0.2],
        [0.1, 0.2, 1.0],
    ]
)
similarity_dataframe_3x3 = DataFrame(
    data=similarity_array_3x3,
    columns=["species_1", "species_2", "species_3"],
    index=["species_1", "species_2", "species_3"],
)
similarity_filecontent_3x3_tsv = (
    "species_1\tspecies_2\tspecies_3\n"
    "1.0\t0.5\t0.1\n"
    "0.5\t1.0\t0.2\n"
    "0.1\t0.2\t1.0\n"
)
similarities_filecontents_3x3_csv = (
    "species_1,species_2,species_3\n" "1.0,0.5,0.1\n" "0.5,1.0,0.2\n" "0.1,0.2,1.0\n"
)
relative_abundances_3x1 = array([[1 / 1000], [1 / 10], [10]])
relative_abundances_3x2 = array([[1 / 1000, 1 / 100], [1 / 10, 1 / 1], [10, 100]])
relative_abundances_3x2_2 = array(
    [
        [0.7, 0.3],
        [0.1, 0.3],
        [0.2, 0.4],
    ]
)
weighted_similarities_3x1 = array([[1.051], [2.1005], [10.0201]])
weighted_similarities_3x2 = array(
    [[1.051, 10.51], [2.1005, 21.005], [10.0201, 100.201]]
)
weighted_similarities_3x2_2 = array(
    [
        [0.81, 0.61],
        [0.77, 0.65],
        [0.29, 0.49],
    ]
)
weighted_similarities_3x2_3 = (
    array(
        [
            [0.35271989, 3.52719894],
            [0.13459705, 1.34597047],
            [0.0601738, 0.60173802],
        ]
    ),
)
X_3x2 = array([[1, 2], [3, 5], [7, 11]])


class MockClass:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class MockSimilarityFromFile(MockClass):
    pass


class MockSimilarityFromDataFrame(MockClass):
    pass


class MockSimilarityFromArray(MockClass):
    pass


@mark.parametrize(
    "similarity_kwargs, similarity_type",
    [
        ({"similarity": similarity_dataframe_3x3}, SimilarityFromDataFrame),
        ({"similarity": similarity_array_3x3}, SimilarityFromArray),
        (
            {"similarity": "fake_similarities_file.tsv", "chunk_size": 2},
            SimilarityFromFile,
        ),
        (
            {
                "similarity": similarity_function,
                "X": X_3x2,
                "chunk_size": 2,
            },
            SimilarityFromFunction,
        ),
    ],
)
def test_make_similarity(similarity_kwargs, similarity_type):
    similarity = make_similarity(**similarity_kwargs)
    assert type(similarity) is similarity_type


def test_do_not_make_similarity():
    similarity = make_similarity(similarity=None)
    assert similarity is None


def test_make_similarity_from_memmap(memmapped_similarity_matrix):
    similarity = make_similarity(memmapped_similarity_matrix)
    assert type(similarity) is SimilarityFromArray


def test_make_similarity_not_implemented():
    with raises(NotImplementedError):
        make_similarity(similarity=1)


@fixture
def make_similarity_from_file(tmp_path):
    def make(
        filename="similarities_file.tsv",
        filecontent=similarity_filecontent_3x3_tsv,
        chunk_size=1,
    ):
        filepath = f"{tmp_path}/{filename}"
        with open(filepath, "w") as file:
            file.write(filecontent)
        return SimilarityFromFile(similarity=filepath, chunk_size=chunk_size)

    return make


@mark.parametrize(
    "relative_abundances, expected, kwargs",
    [
        (relative_abundances_3x2, weighted_similarities_3x2, {}),
        (relative_abundances_3x2, weighted_similarities_3x2, {"chunk_size": 2}),
        (
            relative_abundances_3x2,
            weighted_similarities_3x2,
            {
                "filename": "similarities_file.csv",
                "filecontent": similarities_filecontents_3x3_csv,
            },
        ),
        (relative_abundances_3x1, weighted_similarities_3x1, {}),
    ],
)
def test_weighted_similarities(
    relative_abundances, expected, kwargs, make_similarity_from_file
):
    similarity = make_similarity_from_file(**kwargs)
    assert allclose(similarity.weighted_similarities(relative_abundances), expected)


def test_weighted_similarities_warning(make_similarity_from_file):
    with warns(ArgumentWarning):
        make_similarity_from_file(filename="similarities_file")


@mark.parametrize(
    "similarity, relative_abundances, expected",
    [
        (
            similarity_dataframe_3x3,
            relative_abundances_3x2,
            weighted_similarities_3x2,
        ),
        (
            similarity_dataframe_3x3,
            relative_abundances_3x1,
            weighted_similarities_3x1,
        ),
        (
            similarity_array_3x3,
            relative_abundances_3x1,
            weighted_similarities_3x1,
        ),
        (
            similarity_array_3x3_2,
            relative_abundances_3x2_2,
            weighted_similarities_3x2_2,
        ),
    ],
)
def test_weighted_similarities_from_array(similarity, relative_abundances, expected):
    similarity = make_similarity(similarity=similarity)
    weighted_similarities = similarity.weighted_similarities(relative_abundances)
    assert allclose(weighted_similarities, expected)


def test_weighted_similarities_from_memmap(memmapped_similarity_matrix):
    similarity = make_similarity(similarity=memmapped_similarity_matrix)
    weighted_similarities = similarity.weighted_similarities(relative_abundances_3x2)
    assert allclose(weighted_similarities, weighted_similarities_3x2)


def test_weighted_similarities_from_function():
    similarity = make_similarity(similarity=similarity_function, X=X_3x2, chunk_size=2)
    weighted_similarities = similarity.weighted_similarities(relative_abundances_3x2)
    assert allclose(weighted_similarities, weighted_similarities_3x2_3)
