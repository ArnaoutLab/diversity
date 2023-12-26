"""Tests for diversity.similarity"""
from numpy import allclose, array, dtype, memmap
from pandas import DataFrame
from ray import get, init, shutdown
from pytest import fixture, raises, mark

from metacommunity_diversity.log import LOGGER
from metacommunity_diversity.similarity import (
    SimilarityFromArray,
    SimilarityFromDataFrame,
    SimilarityFromFile,
    SimilarityFromFunction,
    make_similarity,
    weighted_similarity_chunk,
)


@fixture(scope="module")
def ray_fix():
    init(num_cpus=1)
    yield None
    shutdown()


@fixture
def similarity_function():
    return lambda a, b: 1 / sum(a * b)


similarity_array_3by3_1 = array(
    [
        [1, 0.5, 0.1],
        [0.5, 1, 0.2],
        [0.1, 0.2, 1],
    ]
)
similarity_array_3by3_2 = array(
    [
        [1.0, 0.9, 0.1],
        [0.9, 1.0, 0.2],
        [0.1, 0.2, 1.0],
    ]
)
similarity_dataframe_3by3 = DataFrame(
    data=similarity_array_3by3_1,
    columns=["species_1", "species_2", "species_3"],
    index=["species_1", "species_2", "species_3"],
)
similarity_filecontent_3by3_tsv = (
    "species_1\tspecies_2\tspecies_3\n"
    "1.0\t0.5\t0.1\n"
    "0.5\t1.0\t0.2\n"
    "0.1\t0.2\t1.0\n"
)
similarities_filecontents_3by3_csv = (
    "species_1,species_2,species_3\n" "1.0,0.5,0.1\n" "0.5,1.0,0.2\n" "0.1,0.2,1.0\n"
)
relative_abundance_3by1 = array([[1 / 1000], [1 / 10], [10]])
relative_abundance_3by2 = array([[1 / 1000, 1 / 100], [1 / 10, 1 / 1], [10, 100]])
relative_abundance_3by2_2 = array(
    [
        [0.7, 0.3],
        [0.1, 0.3],
        [0.2, 0.4],
    ]
)
weighted_similarities_3by1_1 = array([[1.051], [2.1005], [10.0201]])
weighted_similarities_3by1_2 = array([[0.35271989], [0.13459705], [0.0601738]])
weighted_similarities_3by1_3 = array([[0.35271989], [0.13459705], [0.0601738]])
weighted_similarities_3by2_1 = array(
    [[1.051, 10.51], [2.1005, 21.005], [10.0201, 100.201]]
)
weighted_similarities_3by2_2 = array(
    [
        [0.81, 0.61],
        [0.77, 0.65],
        [0.29, 0.49],
    ]
)
weighted_similarities_3by2_3 = (
    array(
        [
            [0.35271989, 3.52719894],
            [0.13459705, 1.34597047],
            [0.0601738, 0.60173802],
        ]
    ),
)
weighted_similarities_3by2_4 = array(
    [
        [1.46290476, 14.62904762],
        [0.48763492, 4.87634921],
        [0.20898639, 2.08986395],
    ]
)
X_3by1 = array([[1], [3], [7]])
X_3by2 = array([[1, 2], [3, 5], [7, 11]])


@fixture
def memmapped_similarity_matrix(tmp_path):
    memmapped = memmap(
        tmp_path / "similarity_matrix.npy",
        dtype=dtype("f8"),
        mode="w+",
        offset=0,
        shape=similarity_array_3by3_1.shape,
        order="C",
    )
    memmapped[:, :] = similarity_array_3by3_1
    return memmapped


@mark.parametrize(
    "similarity_kwargs, similarity_type",
    [
        ({"similarity": similarity_dataframe_3by3}, SimilarityFromDataFrame),
        ({"similarity": similarity_array_3by3_1}, SimilarityFromArray),
        (
            {"similarity": "similarity_matrix.tsv", "chunk_size": 2},
            SimilarityFromFile,
        ),
        (
            {
                "similarity": similarity_function,
                "X": X_3by2,
                "chunk_size": 2,
            },
            SimilarityFromFunction,
        ),
    ],
)
def test_make_similarity(similarity_kwargs, similarity_type):
    similarity = make_similarity(**similarity_kwargs)
    assert isinstance(similarity, similarity_type)


def test_do_not_make_similarity():
    similarity = make_similarity(similarity=None)
    assert similarity is None


def test_make_similarity_from_memmap(memmapped_similarity_matrix):
    similarity = make_similarity(memmapped_similarity_matrix)
    assert isinstance(similarity, SimilarityFromArray)


def test_make_similarity_not_implemented():
    with raises(NotImplementedError):
        make_similarity(similarity=1)


@fixture
def make_similarity_from_file(tmp_path):
    def make(
        filename="similarity_matrix.tsv",
        filecontent=similarity_filecontent_3by3_tsv,
        chunk_size=1,
    ):
        filepath = tmp_path / filename
        with open(filepath, "w") as file:
            file.write(filecontent)
        return SimilarityFromFile(similarity=filepath, chunk_size=chunk_size)

    return make


@mark.parametrize(
    "relative_abundance, expected, kwargs",
    [
        (relative_abundance_3by2, weighted_similarities_3by2_1, {}),
        (relative_abundance_3by2, weighted_similarities_3by2_1, {"chunk_size": 2}),
        (
            relative_abundance_3by2,
            weighted_similarities_3by2_1,
            {
                "filename": "similarity_matrix.csv",
                "filecontent": similarities_filecontents_3by3_csv,
            },
        ),
        (relative_abundance_3by1, weighted_similarities_3by1_1, {}),
    ],
)
def test_weighted_similarities(
    relative_abundance, expected, kwargs, make_similarity_from_file
):
    similarity = make_similarity_from_file(**kwargs)
    assert allclose(similarity.weighted_similarities(relative_abundance), expected)


@mark.parametrize(
    "similarity, relative_abundance, expected",
    [
        (
            similarity_dataframe_3by3,
            relative_abundance_3by2,
            weighted_similarities_3by2_1,
        ),
        (
            similarity_dataframe_3by3,
            relative_abundance_3by1,
            weighted_similarities_3by1_1,
        ),
        (
            similarity_array_3by3_1,
            relative_abundance_3by1,
            weighted_similarities_3by1_1,
        ),
        (
            similarity_array_3by3_2,
            relative_abundance_3by2_2,
            weighted_similarities_3by2_2,
        ),
    ],
)
def test_weighted_similarities_from_array(similarity, relative_abundance, expected):
    similarity = make_similarity(similarity=similarity)
    weighted_similarities = similarity.weighted_similarities(
        relative_abundance=relative_abundance
    )
    assert allclose(weighted_similarities, expected)


def test_weighted_similarities_from_memmap(memmapped_similarity_matrix):
    similarity = make_similarity(similarity=memmapped_similarity_matrix)
    weighted_similarities = similarity.weighted_similarities(
        relative_abundance=relative_abundance_3by2
    )
    assert allclose(weighted_similarities, weighted_similarities_3by2_1)


@mark.parametrize(
    "relative_abundance, X, chunk_size, expected",
    [
        (relative_abundance_3by2, X_3by2, 2, weighted_similarities_3by2_3),
        (relative_abundance_3by2, X_3by1, 1, weighted_similarities_3by2_4),
        (relative_abundance_3by1, X_3by2, 4, weighted_similarities_3by1_2),
        (relative_abundance_3by1, X_3by2, 2, weighted_similarities_3by1_2),
    ],
)
def test_weighted_similarities_from_function(
    ray_fix, relative_abundance, similarity_function, X, chunk_size, expected
):
    similarity = make_similarity(
        similarity=similarity_function, X=X, chunk_size=chunk_size
    )
    weighted_similarities = similarity.weighted_similarities(
        relative_abundance=relative_abundance
    )
    assert allclose(weighted_similarities, expected)


def test_weighted_similarity_chunk(ray_fix, similarity_function):
    chunk = get(
        weighted_similarity_chunk.remote(
            similarity=similarity_function,
            X=X_3by2,
            relative_abundance=relative_abundance_3by2,
            chunk_size=3,
            chunk_index=0,
        )
    )
    assert allclose(chunk, weighted_similarities_3by2_3)
