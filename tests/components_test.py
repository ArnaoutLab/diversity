from numpy import allclose
from pytest import mark

from diversity.metacommunity import Metacommunity
from tests.metacommunity_test import metacommunity_data


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
