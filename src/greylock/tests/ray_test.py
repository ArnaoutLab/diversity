from numpy import (
    sum,
    sqrt,
    allclose,
    ndarray,
    array,
    dtype,
    memmap,
    inf,
    float32,
    zeros,
)
import ray
from greylock.similarity import (
    SimilarityFromArray,
    SimilarityFromFunction,
    SimilarityFromSymmetricFunction,
)
from greylock.ray import SimilarityFromRayFunction, SimilarityFromSymmetricRayFunction
import greylock.tests.mockray as mockray
from pytest import fixture, raises, mark
from greylock.tests.similarity_test import (
    relative_abundance_3by2,
    relative_abundance_3by1,
    X_3by1,
    X_3by2,
)
from greylock import Metacommunity


def ray_fix(monkeypatch):
    monkeypatch.setattr(ray, "put", mockray.put)
    monkeypatch.setattr(ray, "get", mockray.get)
    monkeypatch.setattr(ray, "remote", mockray.remote)
    monkeypatch.setattr(ray, "wait", mockray.wait)


@fixture(autouse=True)
def setup(monkeypatch):
    ray_fix(monkeypatch)


MEASURES = (
    "alpha",
    "rho",
    "beta",
    "gamma",
    "normalized_alpha",
    "normalized_rho",
    "normalized_beta",
    "rho_hat",
)

abundances_large = array(
    [
        [45, 23],
        [4, 54],
        [23, 1],
        [623, 0],
        [23, 7],
        [23, 90],
        [1, 1],
        [34, 62],
        [13, 72],
        [23, 23],
        [72, 3],
        [62, 3],
        [623, 4],
        [234, 90],
        [23, 12],
        [96, 5],
        [6, 24],
        [6, 4],
        [65, 91],
        [345, 4],
        [23, 62],
        [62, 73],
        [23, 7],
        [23, 90],
        [1, 1],
        [34, 62],
        [13, 72],
        [23, 23],
        [72, 3],
        [62, 3],
        [623, 4],
        [234, 90],
        [23, 12],
        [13, 72],
        [23, 23],
        [72, 3],
        [62, 3],
        [623, 4],
        [234, 90],
        [23, 12],
        [96, 5],
        [6, 24],
        [6, 4],
        [65, 91],
        [345, 4],
        [23, 62],
        [62, 73],
        [23, 7],
        [23, 90],
        [1, 1],
        [34, 62],
        [13, 72],
        [23, 23],
        [72, 3],
        [62, 3],
        [623, 4],
        [234, 90],
        [23, 12],
        [96, 5],
        [6, 24],
        [6, 4],
        [65, 91],
        [345, 4],
        [23, 62],
        [62, 73],
        [23, 7],
    ]
)
X_large = array(
    [
        [6, 1, 1],
        [7, 34, 62],
        [8, 13, 72],
        [9, 23, 23],
        [11, 72, 3],
        [12, 62, 3],
        [13, 623, 4],
        [14, 234, 90],
        [34, 62, 3],
        [62, 43, 4],
        [23, 34, 90],
        [1, 23, 12],
        [2, 96, 5],
        [0, 45, 23],
        [1, 4, 54],
        [2, 23, 1],
        [3, 623, 0],
        [4, 23, 7],
        [5, 23, 90],
        [3, 6, 24],
        [5, 6, 4],
        [6, 65, 91],
        [34, 75, 4],
        [1, 23, 62],
        [2, 62, 73],
        [3, 23, 7],
        [15, 23, 12],
        [16, 96, 5],
        [17, 6, 24],
        [18, 6, 4],
        [19, 65, 91],
        [3, 45, 4],
        [20, 23, 62],
        [21, 62, 73],
        [22, 23, 7],
        [23, 84, 90],
        [14, 1, 1],
        [24, 34, 62],
        [25, 13, 72],
        [23, 75, 23],
        [26, 72, 3],
        [27, 62, 3],
        [62, 73, 4],
        [21, 34, 90],
        [21, 23, 12],
        [22, 13, 72],
        [45, 23, 23],
        [23, 72, 3],
        [24, 62, 3],
        [62, 63, 4],
        [24, 34, 90],
        [24, 23, 12],
        [45, 96, 5],
        [24, 6, 24],
        [25, 6, 4],
        [26, 65, 91],
        [35, 45, 4],
        [23, 23, 62],
        [27, 62, 73],
        [28, 23, 7],
        [29, 23, 90],
        [4, 1, 1],
        [29, 34, 62],
        [31, 13, 72],
        [32, 3, 23],
        [33, 2, 3],
    ]
)


def similarity_function(a, b):
    a = a / sqrt(sum(a * a))
    b = b / sqrt(sum(b * b))
    return sum(a * b)


@mark.parametrize(
    "relative_abundance, X, chunk_size",
    [
        (relative_abundance_3by2, X_3by2, 2),
        (relative_abundance_3by2, X_3by1, 1),
        (relative_abundance_3by1, X_3by2, 4),
        (relative_abundance_3by1, X_3by2, 2),
    ],
)
def test_weighted_abundances_from_function(relative_abundance, X, chunk_size):
    sim_matrix = zeros(shape=(X.shape[0], X.shape[0]))
    for i in range(X.shape[0]):
        for j in range(X.shape[0]):
            sim_matrix[i, j] = similarity_function(X[i], X[j])
    similarity1 = SimilarityFromArray(sim_matrix)
    expected = similarity1.weighted_abundances(relative_abundance=relative_abundance)
    similarity = SimilarityFromRayFunction(
        func=similarity_function, X=X, chunk_size=chunk_size
    )
    weighted_abundances = similarity.weighted_abundances(
        relative_abundance=relative_abundance
    )
    assert allclose(weighted_abundances, expected)


def test_comparisons():
    results = []
    for i, simclass in enumerate(
        [
            SimilarityFromFunction,
            SimilarityFromRayFunction,
            SimilarityFromSymmetricFunction,
            SimilarityFromSymmetricRayFunction,
        ]
    ):
        if i % 2:
            similarity = simclass(
                func=similarity_function, X=X_large, chunk_size=4, max_inflight_tasks=2
            )
        else:
            similarity = simclass(func=similarity_function, X=X_large, chunk_size=4)
        m = Metacommunity(abundances_large, similarity)
        df = m.to_dataframe(viewpoint=[0, 1, 2, 200], measures=MEASURES)
        results.append(df.drop(columns="community"))
    for result in results[1:]:
        assert allclose(results[0].to_numpy(), result.to_numpy())
