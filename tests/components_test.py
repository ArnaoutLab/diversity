from numpy import allclose
from pytest import mark

from metacommunity_diversity.metacommunity import Metacommunity
from metacommunity_diversity.components import (
    FrequencySensitiveComponents,
    SimilaritySensitiveComponents,
)
from metacommunity_diversity.abundance import make_abundance
from metacommunity_diversity.similarity import make_similarity
from metacommunity_diversity.components import make_components
from tests.metacommunity_test import metacommunity_data


@mark.parametrize(
    "data, expected",
    zip(
        metacommunity_data,
        [
            FrequencySensitiveComponents,
            FrequencySensitiveComponents,
            SimilaritySensitiveComponents,
            SimilaritySensitiveComponents,
        ],
    ),
)
def test_make_components(data, expected):
    abundance = make_abundance(counts=data.counts)
    similarity = make_similarity(similarity=data.similarity)
    components = make_components(abundance=abundance, similarity=similarity)
    assert isinstance(components, expected)


@mark.parametrize("data", metacommunity_data[2:])
def test_metacommunity_similarity(data):
    metacommunity = Metacommunity(counts=data.counts, similarity=data.similarity)
    assert allclose(
        metacommunity.components.metacommunity_similarity, data.metacommunity_similarity
    )


@mark.parametrize("data", metacommunity_data[2:])
def test_subcommunity_similarity(data):
    metacommunity = Metacommunity(counts=data.counts, similarity=data.similarity)
    assert allclose(
        metacommunity.components.subcommunity_similarity, data.subcommunity_similarity
    )


@mark.parametrize("data", metacommunity_data[2:])
def test_normalized_subcommunity_similarity(data):
    metacommunity = Metacommunity(counts=data.counts, similarity=data.similarity)
    assert allclose(
        metacommunity.components.normalized_subcommunity_similarity,
        data.normalized_subcommunity_similarity,
    )
