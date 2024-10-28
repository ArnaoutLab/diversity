from numpy import allclose, ndarray, array, dtype, memmap, inf, float32, zeros
import ray
from greylock.ray import SimilarityFromRayFunction
import greylock.tests.mockray as mockray
from pytest import fixture, raises, mark
from greylock.tests.similarity_test import (
    relative_abundance_3by2,
    relative_abundance_3by1,
    weighted_abundances_3by2_3,
    weighted_abundances_3by2_4,
    weighted_abundances_3by1_2,
    X_3by1,
    X_3by2,
)


def ray_fix(monkeypatch):
    monkeypatch.setattr(ray, "put", mockray.put)
    monkeypatch.setattr(ray, "get", mockray.get)
    monkeypatch.setattr(ray, "remote", mockray.remote)
    monkeypatch.setattr(ray, "wait", mockray.wait)


@fixture(autouse=True)
def setup(monkeypatch):
    ray_fix(monkeypatch)


@fixture
def similarity_function():
    return lambda a, b: 1 / sum(a * b)


@mark.parametrize(
    "relative_abundance, X, chunk_size, expected",
    [
        (relative_abundance_3by2, X_3by2, 2, weighted_abundances_3by2_3),
        (relative_abundance_3by2, X_3by1, 1, weighted_abundances_3by2_4),
        (relative_abundance_3by1, X_3by2, 4, weighted_abundances_3by1_2),
        (relative_abundance_3by1, X_3by2, 2, weighted_abundances_3by1_2),
    ],
)
def test_weighted_abundances_from_function(
    relative_abundance, similarity_function, X, chunk_size, expected
):
    similarity = SimilarityFromRayFunction(
        func=similarity_function, X=X, chunk_size=chunk_size
    )
    weighted_abundances = similarity.weighted_abundances(
        relative_abundance=relative_abundance
    )
    assert allclose(weighted_abundances, expected)
