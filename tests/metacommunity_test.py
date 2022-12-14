"""Tests for diversity.metacommunity."""
from copy import deepcopy

from numpy import (
    allclose,
    array,
    dtype,
    isclose,
)
from pandas import DataFrame
from pandas.testing import assert_frame_equal
from pytest import fixture, mark

from diversity.log import LOGGER
from diversity.abundance import Abundance
from diversity.similarity import Similarity
from diversity.metacommunity import Metacommunity
from tests.similarity_test import (
    similarity_dataframe_3by3,
    similarity_array_3by3_1,
    similarity_filecontent_3by3_tsv,
    similarity_function,
    X_3by2,
)


measures = [
    "alpha",
    "rho",
    "beta",
    "gamma",
    "normalized_alpha",
    "normalized_rho",
    "normalized_beta",
]
counts_dataframe_3by3 = DataFrame(
    [[2, 5, 0], [4, 3, 2], [0, 0, 3]],
    index=["species_1", "species_2", "species_3"],
    columns=["subcommunity_1", "subcommunity_2", "subcommunity_3"],
)
counts_dataframe_3by2 = DataFrame(
    [[1, 5], [3, 0], [0, 1]], columns=["subcommunity_1", "subcommunity_2"]
)
counts_array_6by2 = array([[1, 0], [1, 0], [1, 0], [0, 1], [0, 1], [0, 1]])
similarity_matrix_6x6 = array(
    [
        [1.0, 0.5, 0.5, 0.7, 0.7, 0.7],
        [0.5, 1.0, 0.5, 0.7, 0.7, 0.7],
        [0.5, 0.5, 1.0, 0.7, 0.7, 0.7],
        [0.7, 0.7, 0.7, 1.0, 0.5, 0.5],
        [0.7, 0.7, 0.7, 0.5, 1.0, 0.5],
        [0.7, 0.7, 0.7, 0.5, 0.5, 1.0],
    ]
)
subcommunity_results_1 = DataFrame(
    {
        "community": [0, 1],
        "viewpoint": [0, 0],
        "alpha": [6.0, 6.0],
        "rho": [1.0, 1.0],
        "beta": [1.0, 1.0],
        "gamma": [6.0, 6.0],
        "normalized_alpha": [3.0, 3.0],
        "normalized_rho": [0.5, 0.5],
        "normalized_beta": [2.0, 2.0],
    },
)
subcommunity_results_2 = DataFrame(
    {
        "community": ["subcommunity_1", "subcommunity_2"],
        "viewpoint": [2, 2],
        "alpha": [4.0, 2.30769231],
        "rho": [1.26315789, 1.16129032],
        "beta": [0.79166667, 0.86111111],
        "gamma": [2.66666667, 1.93548387],
        "normalized_alpha": [1.6, 1.38461538],
        "normalized_rho": [0.50526316, 0.69677419],
        "normalized_beta": [1.97916667, 1.43518519],
    }
)
subcommunity_results_3 = DataFrame(
    {
        "community": [0, 1],
        "viewpoint": [0, 0],
        "alpha": [3.0, 3.0],
        "rho": [2.05, 2.05],
        "beta": [0.487805, 0.487805],
        "gamma": [1.463415, 1.463415],
        "normalized_alpha": [1.5, 1.5],
        "normalized_rho": [1.025, 1.025],
        "normalized_beta": [0.97561, 0.97561],
    }
)
subcommunity_results_4 = DataFrame(
    {
        "community": array(["subcommunity_1", "subcommunity_2"]),
        "viewpoint": [2, 2],
        "alpha": array([3.07692308, 2.22222222]),
        "rho": array([1.97775446, 1.48622222]),
        "beta": array([0.50562394, 0.67284689]),
        "gamma": array([1.52671756, 1.49253731]),
        "normalized_alpha": array([1.23076923, 1.33333333]),
        "normalized_rho": array([0.79110178, 0.89173333]),
        "normalized_beta": array([1.26405985, 1.12141148]),
    }
)
metacommunity_results_1 = DataFrame(
    {
        "community": ["metacommunity"],
        "viewpoint": [0],
        "alpha": [6.0],
        "rho": [1.0],
        "beta": [1.0],
        "gamma": [6.0],
        "normalized_alpha": [3.0],
        "normalized_rho": [0.5],
        "normalized_beta": [2.0],
    },
    index=[0],
)
metacommunity_results_2 = DataFrame(
    {
        "community": ["metacommunity"],
        "viewpoint": [2],
        "alpha": [2.7777777777777777],
        "rho": [1.2],
        "beta": [0.8319209039548022],
        "gamma": [2.173913043478261],
        "normalized_alpha": [1.4634146341463414],
        "normalized_rho": [0.6050420168067228],
        "normalized_beta": [1.612461673236969],
    },
    index=[0],
)
metacommunity_results_3 = DataFrame(
    {
        "community": ["metacommunity"],
        "viewpoint": [0],
        "alpha": [3.0],
        "rho": [2.05],
        "beta": [0.487805],
        "gamma": [1.463415],
        "normalized_alpha": [1.5],
        "normalized_rho": [1.025],
        "normalized_beta": [0.97561],
    },
    index=[0],
)
metacommunity_results_4 = DataFrame(
    {
        "community": ["metacommunity"],
        "viewpoint": [2],
        "alpha": [2.5],
        "rho": [1.6502801833927663],
        "beta": [0.5942352817544037],
        "gamma": [1.5060240963855422],
        "normalized_alpha": [1.2903225806451613],
        "normalized_rho": [0.8485572790897555],
        "normalized_beta": [1.1744247216675028],
    },
    index=[0],
)
metacommunity_similarity_1 = array(
    [
        [0.68333333],
        [0.68333333],
        [0.68333333],
        [0.68333333],
        [0.68333333],
        [0.68333333],
    ]
)
metacommunity_similarity_2 = array([[0.76], [0.62], [0.22]])
subcommunity_similarity_1 = (
    array(
        [
            [0.33333333, 0.35],
            [0.33333333, 0.35],
            [0.33333333, 0.35],
            [0.35, 0.33333333],
            [0.35, 0.33333333],
            [0.35, 0.33333333],
        ],
    ),
)
subcommunity_similarity_2 = (
    array(
        [
            [0.25, 0.51],
            [0.35, 0.27],
            [0.07, 0.15],
        ]
    ),
)
normalized_subcommunity_similarity_1 = array(
    [
        [0.66666667, 0.7],
        [0.66666667, 0.7],
        [0.66666667, 0.7],
        [0.7, 0.66666667],
        [0.7, 0.66666667],
        [0.7, 0.66666667],
    ]
)
normalized_subcommunity_similarity_2 = array(
    [
        [0.625, 0.85],
        [0.875, 0.45],
        [0.175, 0.25],
    ]
)

viewpoints = [0, 2, 0, 2]
counts_arrays = [
    counts_array_6by2,
    counts_dataframe_3by2,
    counts_array_6by2,
    counts_dataframe_3by2,
]
similarities = [None, None, similarity_matrix_6x6, similarity_dataframe_3by3]
metacommunity_results = [
    metacommunity_results_1,
    metacommunity_results_2,
    metacommunity_results_3,
    metacommunity_results_4,
]
subcommunity_results = [
    subcommunity_results_1,
    subcommunity_results_2,
    subcommunity_results_3,
    subcommunity_results_4,
]


def make_diversity_measure_params(results):
    return [
        (
            measure,
            viewpoint,
            counts,
            similarity,
            result[measure],
        )
        for viewpoint, counts, similarity, result in zip(
            viewpoints,
            counts_arrays,
            similarities,
            results,
        )
        for measure in measures
    ]


metacommunity_diversity_params = make_diversity_measure_params(metacommunity_results)
subcommunity_diversity_params = make_diversity_measure_params(subcommunity_results)


@fixture
def write_similarity_matrix(
    tmp_path,
    filename="similarity_matrix.tsv",
    filecontent=similarity_filecontent_3by3_tsv,
):
    filepath = tmp_path / filename
    with open(filepath, "w") as file:
        file.write(filecontent)
    return filepath


@mark.parametrize(
    "similarity, X, chunk_size",
    [
        (None, None, None),
        (similarity_dataframe_3by3, None, None),
        (similarity_array_3by3_1, None, None),
        ("write_similarity_matrix.tsv", None, 2),
        (similarity_function, X_3by2, 2),
    ],
)
def test_metacommunity(similarity, X, chunk_size, request):
    if similarity is str:
        similarity = request.getfixturevalue(similarity)
    metacommunity = Metacommunity(
        counts=counts_dataframe_3by3, similarity=similarity, X=X, chunk_size=chunk_size
    )
    assert isinstance(metacommunity, Metacommunity)
    assert isinstance(metacommunity.abundance, Abundance)
    assert isinstance(metacommunity.measure_components, dict)
    if similarity is not None:
        assert isinstance(metacommunity.similarity, Similarity)


@mark.parametrize(
    "viewpoint, counts, similarity, expected",
    zip(viewpoints, counts_arrays, similarities, subcommunity_results),
)
def test_subcommunities_to_dataframe(viewpoint, counts, similarity, expected):
    metacommunity = Metacommunity(counts=counts, similarity=similarity)
    subcommunities_df = metacommunity.subcommunities_to_dataframe(viewpoint)
    assert_frame_equal(subcommunities_df, expected)


@mark.parametrize(
    "viewpoint, counts, similarity, expected",
    zip(viewpoints, counts_arrays, similarities, metacommunity_results),
)
def test_metacommunties_to_dataframe(viewpoint, counts, similarity, expected):
    metacommunity = Metacommunity(counts=counts, similarity=similarity)
    metacommunity_df = metacommunity.metacommunity_to_dataframe(viewpoint)
    assert_frame_equal(metacommunity_df, expected)


@mark.parametrize(
    "measure, viewpoint, counts, similarity, expected",
    metacommunity_diversity_params,
)
def test_metacommunity_diversity(
    measure,
    viewpoint,
    counts,
    similarity,
    expected,
):
    metacommunity = Metacommunity(counts=counts, similarity=similarity)
    metacommunity_diversity = metacommunity.metacommunity_diversity(
        measure=measure, viewpoint=viewpoint
    )
    assert allclose(metacommunity_diversity, expected)


@mark.parametrize(
    "measure, viewpoint, counts, similarity, expected",
    subcommunity_diversity_params,
)
def test_subcommunity_diversity(
    measure,
    viewpoint,
    counts,
    similarity,
    expected,
):
    metacommunity = Metacommunity(counts=counts, similarity=similarity)
    subcommunity_diversity = metacommunity.subcommunity_diversity(
        measure=measure, viewpoint=viewpoint
    )
    assert allclose(subcommunity_diversity, expected)


@mark.parametrize(
    "counts, similarity, expected",
    [
        (counts_array_6by2, similarity_matrix_6x6, metacommunity_similarity_1),
        (counts_dataframe_3by2, similarity_dataframe_3by3, metacommunity_similarity_2),
    ],
)
def test_metacommunity_similarity(counts, similarity, expected):
    metacommunity = Metacommunity(counts=counts, similarity=similarity)
    metacommunity_similarity = metacommunity.metacommunity_similarity()
    assert allclose(metacommunity_similarity, expected)


@mark.parametrize(
    "counts, similarity, expected",
    [
        (counts_array_6by2, similarity_matrix_6x6, subcommunity_similarity_1),
        (counts_dataframe_3by2, similarity_dataframe_3by3, subcommunity_similarity_2),
    ],
)
def test_subcommunity_similarity(counts, similarity, expected):
    metacommunity = Metacommunity(counts=counts, similarity=similarity)
    subcommunity_similarity = metacommunity.subcommunity_similarity()
    assert allclose(subcommunity_similarity, expected)


@mark.parametrize(
    "counts, similarity, expected",
    [
        (
            counts_array_6by2,
            similarity_matrix_6x6,
            normalized_subcommunity_similarity_1,
        ),
        (
            counts_dataframe_3by2,
            similarity_dataframe_3by3,
            normalized_subcommunity_similarity_2,
        ),
    ],
)
def test_normalized_subcommunity_similarity(counts, similarity, expected):
    metacommunity = Metacommunity(counts=counts, similarity=similarity)
    normalized_subcommunity_similarity = (
        metacommunity.normalized_subcommunity_similarity()
    )
    assert allclose(normalized_subcommunity_similarity, expected)
