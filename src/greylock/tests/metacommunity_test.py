"""Tests for diversity.metacommunity."""

from dataclasses import dataclass, field
from numpy import (
    allclose,
    array,
    ndarray,
)
from pandas import DataFrame, concat
from pandas.testing import assert_frame_equal
from pytest import mark, raises
from greylock.exceptions import InvalidArgumentError

from greylock.log import LOGGER
from greylock.abundance import Abundance
from greylock.similarity import Similarity
from greylock import Metacommunity
from greylock.tests.similarity_test import similarity_dataframe_3by3

MEASURES = (
    "alpha",
    "rho",
    "beta",
    "gamma",
    "normalized_alpha",
    "normalized_rho",
    "normalized_beta",
    "rho_hat",
    "beta_hat",
)

subcommunity_names = ["subcommunity_1", "subcommunity_2"]
counts_3by2 = DataFrame([[1, 5], [3, 0], [0, 1]], columns=subcommunity_names)
counts_6by2 = DataFrame(
    [[1, 0], [1, 0], [1, 0], [0, 1], [0, 1], [0, 1]],
    columns=subcommunity_names,
)


@dataclass
class FrequencyMetacommunity6by2:
    description = "frequency-sensitive metacommunity; 6 species, 2 subcommunities"
    viewpoint: float = 0.0
    counts: DataFrame = field(default_factory=lambda: counts_6by2)
    similarity: None = None
    metacommunity_similarity: None = None
    subcommunity_similarity: None = None
    normalized_subcommunity_similarity: None = None
    subcommunity_results: DataFrame = field(
        default_factory=lambda: DataFrame(
            {
                "community": subcommunity_names,
                "viewpoint": [0.0, 0.0],
                "alpha": [6.0, 6.0],
                "rho": [1.0, 1.0],
                "beta": [1.0, 1.0],
                "gamma": [6.0, 6.0],
                "normalized_alpha": [3.0, 3.0],
                "normalized_rho": [0.5, 0.5],
                "normalized_beta": [2.0, 2.0],
                "rho_hat": [0.0, 0.0],
                "beta_hat": [1.0, 1.0],
            },
        )
    )
    metacommunity_results: DataFrame = field(
        default_factory=lambda: DataFrame(
            {
                "community": ["metacommunity"],
                "viewpoint": [0.0],
                "alpha": [6.0],
                "rho": [1.0],
                "beta": [1.0],
                "gamma": [6.0],
                "normalized_alpha": [3.0],
                "normalized_rho": [0.5],
                "normalized_beta": [2.0],
                "rho_hat": [0.0],
                "beta_hat": [1.0],
            },
            index=[0],
        )
    )


@dataclass
class FrequencyMetacommunity3by2:
    description = "frequency-sensitive metacommunity; 3 species, 2 subcommunities"
    viewpoint: float = 2.0
    counts: DataFrame = field(default_factory=lambda: counts_3by2)
    similarity: None = None
    metacommunity_similarity: None = None
    subcommunity_similarity: None = None
    normalized_subcommunity_similarity: None = None
    subcommunity_results: DataFrame = field(
        default_factory=lambda: DataFrame(
            {
                "community": subcommunity_names,
                "viewpoint": [2.0, 2.0],
                "alpha": [4.0, 2.30769231],
                "rho": [1.26315789, 1.16129032],
                "beta": [0.79166667, 0.86111111],
                "gamma": [2.66666667, 1.93548387],
                "normalized_alpha": [1.6, 1.38461538],
                "normalized_rho": [0.50526316, 0.69677419],
                "normalized_beta": [1.97916667, 1.43518519],
                "rho_hat": [0.263158, 0.161290],
                "beta_hat": [0.583333, 0.722222],
            }
        )
    )
    metacommunity_results: DataFrame = field(
        default_factory=lambda: DataFrame(
            {
                "community": ["metacommunity"],
                "viewpoint": [2.0],
                "alpha": [2.7777777777777777],
                "rho": [1.2],
                "beta": [0.8319209039548022],
                "gamma": [2.173913043478261],
                "normalized_alpha": [1.4634146341463414],
                "normalized_rho": [0.6050420168067228],
                "normalized_beta": [1.612461673236969],
                "rho_hat": [0.190840],
                "beta_hat": [0.659420],
            },
            index=[0],
        )
    )


@dataclass
class SimilarityMetacommunity6by2:
    description = "similarity-sensitive metacommunity; 6 species, 2 subcommunities"
    viewpoint: float = 0.0
    counts: ndarray = field(default_factory=lambda: counts_6by2)
    similarity: ndarray = field(
        default_factory=lambda: array(
            [
                [1.0, 0.5, 0.5, 0.7, 0.7, 0.7],
                [0.5, 1.0, 0.5, 0.7, 0.7, 0.7],
                [0.5, 0.5, 1.0, 0.7, 0.7, 0.7],
                [0.7, 0.7, 0.7, 1.0, 0.5, 0.5],
                [0.7, 0.7, 0.7, 0.5, 1.0, 0.5],
                [0.7, 0.7, 0.7, 0.5, 0.5, 1.0],
            ]
        )
    )
    metacommunity_similarity: ndarray = field(
        default_factory=lambda: array(
            [
                [0.68333333],
                [0.68333333],
                [0.68333333],
                [0.68333333],
                [0.68333333],
                [0.68333333],
            ]
        )
    )
    subcommunity_similarity: ndarray = field(
        default_factory=lambda: (
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
    )
    normalized_subcommunity_similarity: ndarray = field(
        default_factory=lambda: array(
            [
                [0.66666667, 0.7],
                [0.66666667, 0.7],
                [0.66666667, 0.7],
                [0.7, 0.66666667],
                [0.7, 0.66666667],
                [0.7, 0.66666667],
            ]
        )
    )
    subcommunity_results: DataFrame = field(
        default_factory=lambda: DataFrame(
            {
                "community": subcommunity_names,
                "viewpoint": [0.0, 0.0],
                "alpha": [3.0, 3.0],
                "rho": [2.05, 2.05],
                "beta": [0.487805, 0.487805],
                "gamma": [1.463415, 1.463415],
                "normalized_alpha": [1.5, 1.5],
                "normalized_rho": [1.025, 1.025],
                "normalized_beta": [0.97561, 0.97561],
                "rho_hat": [1.05, 1.05],
                "beta_hat": [-0.02439, -0.02439],
            }
        )
    )
    metacommunity_results: DataFrame = field(
        default_factory=lambda: DataFrame(
            {
                "community": ["metacommunity"],
                "viewpoint": [0.0],
                "alpha": [3.0],
                "rho": [2.05],
                "beta": [0.487805],
                "gamma": [1.463415],
                "normalized_alpha": [1.5],
                "normalized_rho": [1.025],
                "normalized_beta": [0.97561],
                "rho_hat": [1.05],
                "beta_hat": [-0.02439],
            },
            index=[0],
        )
    )


@dataclass
class SimilarityMetacommunity3by2:
    description = "similarity-sensitive metacommunity; 3 species, 2 subcommunities"
    viewpoint: float = 2.0
    counts: DataFrame = field(default_factory=lambda: counts_3by2)
    similarity: DataFrame = field(default_factory=lambda: similarity_dataframe_3by3)
    metacommunity_similarity: ndarray = field(
        default_factory=lambda: array([[0.76], [0.62], [0.22]])
    )
    subcommunity_similarity: ndarray = field(
        default_factory=lambda: (
            array(
                [
                    [0.25, 0.51],
                    [0.35, 0.27],
                    [0.07, 0.15],
                ]
            ),
        )
    )
    normalized_subcommunity_similarity: ndarray = field(
        default_factory=lambda: array(
            [
                [0.625, 0.85],
                [0.875, 0.45],
                [0.175, 0.25],
            ]
        )
    )
    subcommunity_results: DataFrame = field(
        default_factory=lambda: DataFrame(
            {
                "community": subcommunity_names,
                "viewpoint": [2.0, 2.0],
                "alpha": [3.07692308, 2.22222222],
                "rho": [1.97775446, 1.48622222],
                "beta": [0.50562394, 0.67284689],
                "gamma": [1.52671756, 1.49253731],
                "normalized_alpha": [1.23076923, 1.33333333],
                "normalized_rho": [0.79110178, 0.89173333],
                "normalized_beta": [1.26405985, 1.12141148],
                "rho_hat": [0.977754, 0.486222],
                "beta_hat": [0.011247877758913116, 0.345694],
            }
        )
    )
    metacommunity_results: DataFrame = field(
        default_factory=lambda: DataFrame(
            {
                "community": ["metacommunity"],
                "viewpoint": [2.0],
                "alpha": [2.5],
                "rho": [1.6502801833927663],
                "beta": [0.5942352817544037],
                "gamma": [1.5060240963855422],
                "normalized_alpha": [1.2903225806451613],
                "normalized_rho": [0.8485572790897555],
                "normalized_beta": [1.1744247216675028],
                "rho_hat": [0.608604],
                "beta_hat": [0.026811],
            },
            index=[0],
        )
    )


metacommunity_data = (
    FrequencyMetacommunity6by2(),
    FrequencyMetacommunity3by2(),
    SimilarityMetacommunity6by2(),
    SimilarityMetacommunity3by2(),
)


@mark.parametrize(
    "data, expected",
    zip(metacommunity_data, (type(None), type(None), Similarity, Similarity)),
)
def test_metacommunity(data, expected):
    metacommunity = Metacommunity(counts=data.counts, similarity=data.similarity)
    assert isinstance(metacommunity, Metacommunity)
    assert isinstance(metacommunity.abundance, Abundance)
    assert isinstance(metacommunity.similarity, expected)


@mark.parametrize("measure", MEASURES)
@mark.parametrize("data", metacommunity_data)
def test_metacommunity_diversity(data, measure):
    metacommunity = Metacommunity(counts=data.counts, similarity=data.similarity)
    metacommunity_diversity = metacommunity.metacommunity_diversity(
        measure=measure, viewpoint=data.viewpoint
    )
    assert allclose(metacommunity_diversity, data.metacommunity_results[measure])


@mark.parametrize("measure", MEASURES)
@mark.parametrize("data", metacommunity_data)
def test_subcommunity_diversity(data, measure):
    metacommunity = Metacommunity(counts=data.counts, similarity=data.similarity)
    subcommunity_diversity = metacommunity.subcommunity_diversity(
        measure=measure, viewpoint=data.viewpoint
    )
    assert allclose(subcommunity_diversity, data.subcommunity_results[measure])


def test_subcommunity_diversity_invalid_measure():
    with raises(InvalidArgumentError):
        Metacommunity(counts=counts_3by2).subcommunity_diversity(
            measure="omega", viewpoint=0
        )


@mark.parametrize("data", metacommunity_data)
def test_subcommunities_to_dataframe(data):
    metacommunity = Metacommunity(counts=data.counts, similarity=data.similarity)
    subcommunities_df = metacommunity.subcommunities_to_dataframe(data.viewpoint)
    assert_frame_equal(subcommunities_df, data.subcommunity_results)


@mark.parametrize("data", metacommunity_data)
def test_metacommunities_to_dataframe(data):
    metacommunity = Metacommunity(counts=data.counts, similarity=data.similarity)
    metacommunity_df = metacommunity.metacommunity_to_dataframe(
        viewpoint=data.viewpoint
    )
    assert_frame_equal(metacommunity_df, data.metacommunity_results)


@mark.parametrize("data", metacommunity_data)
def test_to_dataframe(data):
    metacommunity = Metacommunity(counts=data.counts, similarity=data.similarity)
    expected = concat(
        [data.metacommunity_results, data.subcommunity_results]
    ).reset_index(drop=True)
    assert_frame_equal(metacommunity.to_dataframe(viewpoint=data.viewpoint), expected)


@mark.parametrize("data", metacommunity_data)
def test_select_measures(data):
    metacommunity = Metacommunity(counts=data.counts, similarity=data.similarity)
    selected_measures = [
        "alpha",
        "gamma",
        "normalized_rho",
    ]
    expected_columns = selected_measures + ["community", "viewpoint"]
    df = metacommunity.to_dataframe(
        viewpoint=data.viewpoint, measures=selected_measures
    )
    for col in df:
        assert col in expected_columns
    for col in expected_columns:
        assert col in df
