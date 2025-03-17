from numpy import allclose
from pytest import mark

from greylock.metacommunity import Metacommunity
from greylock.components import Components
from greylock.abundance import make_abundance
from greylock.tests.base_tests.metacommunity_test import metacommunity_data
from greylock.tests.base_tests.similarity_test import similarity_array_3by3_1


@mark.parametrize(
    "data",
    metacommunity_data,
)
def test_make_components(data):
    metacommunity = Metacommunity(counts=data.counts, similarity=data.similarity)
    assert isinstance(metacommunity.components, Components)


@mark.parametrize("data", metacommunity_data[2:])
def test_metacommunity_similarity(data):
    metacommunity = Metacommunity(counts=data.counts, similarity=data.similarity)
    assert allclose(
        metacommunity.components.metacommunity_ordinariness,
        data.metacommunity_similarity,
    )


@mark.parametrize("data", metacommunity_data[2:])
def test_subcommunity_similarity(data):
    metacommunity = Metacommunity(counts=data.counts, similarity=data.similarity)
    assert allclose(
        metacommunity.components.subcommunity_ordinariness, data.subcommunity_similarity
    )


@mark.parametrize("data", metacommunity_data[2:])
def test_normalized_subcommunity_similarity(data):
    metacommunity = Metacommunity(counts=data.counts, similarity=data.similarity)
    assert allclose(
        metacommunity.components.normalized_subcommunity_ordinariness,
        data.normalized_subcommunity_similarity,
    )
