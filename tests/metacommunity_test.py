"""Tests for diversity.metacommunity."""
from numpy import (
    allclose,
    array,
    dtype,
    isclose,
)
from pandas import DataFrame
from pytest import fixture, mark

from diversity.abundance import Abundance, SharedAbundance
from diversity.metacommunity import (
    make_metacommunity,
    Metacommunity,
)
from diversity.shared import SharedArrayManager, SharedArrayView
from diversity.similarity import SimilarityFromFunction, SimilarityFromMemory


def sim_func(a, b):
    distance_table = {
        ("species_1", "species_1"): 1.0,
        ("species_1", "species_2"): 0.5,
        ("species_1", "species_3"): 0.1,
        ("species_2", "species_1"): 0.5,
        ("species_2", "species_2"): 1.0,
        ("species_2", "species_3"): 0.2,
        ("species_3", "species_1"): 0.1,
        ("species_3", "species_2"): 0.2,
        ("species_3", "species_3"): 1.0,
    }
    return distance_table[(a[0], b[0])]


METACOMMUNITY_TEST_CASES = [
    {
        "description": "disjoint communities; uniform counts; uniform inter-community similarities; viewpoint 0.",
        "similarity": SimilarityFromMemory(
            similarity_matrix=DataFrame(
                data=array(
                    [
                        [1.0, 0.5, 0.5, 0.7, 0.7, 0.7],
                        [0.5, 1.0, 0.5, 0.7, 0.7, 0.7],
                        [0.5, 0.5, 1.0, 0.7, 0.7, 0.7],
                        [0.7, 0.7, 0.7, 1.0, 0.5, 0.5],
                        [0.7, 0.7, 0.7, 0.5, 1.0, 0.5],
                        [0.7, 0.7, 0.7, 0.5, 0.5, 1.0],
                    ]
                ),
                columns=[
                    "species_1",
                    "species_2",
                    "species_3",
                    "species_4",
                    "species_5",
                    "species_6",
                ],
                index=[
                    "species_1",
                    "species_2",
                    "species_3",
                    "species_4",
                    "species_5",
                    "species_6",
                ],
            ),
            species_subset=[
                "species_1",
                "species_2",
                "species_3",
                "species_4",
                "species_5",
                "species_6",
            ],
        ),
        "abundance": Abundance(
            counts=array([[1, 0], [1, 0], [1, 0], [0, 1], [0, 1], [0, 1]])
        ),
        "viewpoint": 0,
        "metacommunity_similarity": array(
            [
                [0.68333333],
                [0.68333333],
                [0.68333333],
                [0.68333333],
                [0.68333333],
                [0.68333333],
            ]
        ),
        "subcommunity_similarity": array(
            [
                [0.33333333, 0.35],
                [0.33333333, 0.35],
                [0.33333333, 0.35],
                [0.35, 0.33333333],
                [0.35, 0.33333333],
                [0.35, 0.33333333],
            ],
        ),
        "normalized_subcommunity_similarity": array(
            [
                [0.66666667, 0.7],
                [0.66666667, 0.7],
                [0.66666667, 0.7],
                [0.7, 0.66666667],
                [0.7, 0.66666667],
                [0.7, 0.66666667],
            ]
        ),
        "subcommunity_alpha": array([3.0, 3.0]),
        "subcommunity_rho": array([2.05, 2.05]),
        "subcommunity_beta": array([0.487805, 0.487805]),
        "subcommunity_gamma": array([1.463415, 1.463415]),
        "normalized_subcommunity_alpha": array([1.5, 1.5]),
        "normalized_subcommunity_rho": array([1.025, 1.025]),
        "normalized_subcommunity_beta": array([0.97561, 0.97561]),
        "metacommunity_alpha": 3.0,
        "metacommunity_rho": 2.05,
        "metacommunity_beta": 0.487805,
        "metacommunity_gamma": 1.463415,
        "metacommunity_normalized_alpha": 1.5,
        "metacommunity_normalized_rho": 1.025,
        "metacommunity_normalized_beta": 0.97561,
    },
    {
        "description": "overlapping communities; non-uniform counts; non-uniform inter-community similarities; viewpoint 2.",
        "similarity": SimilarityFromMemory(
            similarity_matrix=DataFrame(
                data=array([[1.0, 0.5, 0.1], [0.5, 1.0, 0.2], [0.1, 0.2, 1.0]]),
                columns=[
                    "species_1",
                    "species_2",
                    "species_3",
                ],
                index=[
                    "species_1",
                    "species_2",
                    "species_3",
                ],
            ),
            species_subset=[
                "species_1",
                "species_2",
                "species_3",
            ],
        ),
        "abundance": Abundance(counts=array([[1, 5], [3, 0], [0, 1]])),
        "viewpoint": 2,
        "metacommunity_similarity": array([[0.76], [0.62], [0.22]]),
        "subcommunity_similarity": array(
            [
                [0.25, 0.51],
                [0.35, 0.27],
                [0.07, 0.15],
            ]
        ),
        "normalized_subcommunity_similarity": array(
            [
                [0.625, 0.85],
                [0.875, 0.45],
                [0.175, 0.25],
            ]
        ),
        "subcommunity_alpha": array([3.07692308, 2.22222222]),
        "subcommunity_rho": array([1.97775446, 1.48622222]),
        "subcommunity_beta": array([0.50562394, 0.67284689]),
        "subcommunity_gamma": array([1.52671756, 1.49253731]),
        "normalized_subcommunity_alpha": array([1.23076923, 1.33333333]),
        "normalized_subcommunity_rho": array([0.79110178, 0.89173333]),
        "normalized_subcommunity_beta": array([1.26405985, 1.12141148]),
        "metacommunity_alpha": 2.5,
        "metacommunity_rho": 1.6502801833927663,
        "metacommunity_beta": 0.5942352817544037,
        "metacommunity_gamma": 1.5060240963855422,
        "metacommunity_normalized_alpha": 1.2903225806451613,
        "metacommunity_normalized_rho": 0.8485572790897555,
        "metacommunity_normalized_beta": 1.1744247216675028,
    },
    {
        "description": "similarity from function; default num_processors.",
        "similarity": {
            "similarity_function": sim_func,
            "species_ordering": array(
                [
                    "species_1",
                    "species_2",
                    "species_3",
                ]
            ),
            "num_processors": None,
        },
        "abundance": array([[1, 5], [3, 0], [0, 1]], dtype=dtype("f8")),
        "viewpoint": 2,
        "metacommunity_similarity": array([[0.76], [0.62], [0.22]]),
        "subcommunity_similarity": array(
            [
                [0.25, 0.51],
                [0.35, 0.27],
                [0.07, 0.15],
            ]
        ),
        "normalized_subcommunity_similarity": array(
            [
                [0.625, 0.85],
                [0.875, 0.45],
                [0.175, 0.25],
            ]
        ),
        "subcommunity_alpha": array([3.07692308, 2.22222222]),
        "subcommunity_rho": array([1.97775446, 1.48622222]),
        "subcommunity_beta": array([0.50562394, 0.67284689]),
        "subcommunity_gamma": array([1.52671756, 1.49253731]),
        "normalized_subcommunity_alpha": array([1.23076923, 1.33333333]),
        "normalized_subcommunity_rho": array([0.79110178, 0.89173333]),
        "normalized_subcommunity_beta": array([1.26405985, 1.12141148]),
        "metacommunity_alpha": 2.5,
        "metacommunity_rho": 1.6502801833927663,
        "metacommunity_beta": 0.5942352817544037,
        "metacommunity_gamma": 1.5060240963855422,
        "metacommunity_normalized_alpha": 1.2903225806451613,
        "metacommunity_normalized_rho": 0.8485572790897555,
        "metacommunity_normalized_beta": 1.1744247216675028,
    },
]


class TestMetacommunity:
    """Tests metacommunity.Metacommunity."""

    @fixture(params=METACOMMUNITY_TEST_CASES, scope="class")
    def test_case(self, request):
        if isinstance(request.param["similarity"], dict):
            kwargs = request.param["similarity"]
            with SharedArrayManager() as shared_array_manager:
                shared_features = shared_array_manager.from_array(
                    kwargs["species_ordering"].reshape(-1, 1)
                )
                kwargs["shared_array_manager"] = shared_array_manager
                kwargs["features_spec"] = shared_features.spec
                shared_features.data[:] = kwargs["species_ordering"].reshape(-1, 1)
                similarity = SimilarityFromFunction(**kwargs)
                counts = shared_array_manager.from_array(request.param["abundance"])
                abundance = SharedAbundance(
                    counts=counts, shared_array_manager=shared_array_manager
                )
                request.param["similarity"] = similarity
                request.param["abundance"] = abundance

                yield request.param
        else:
            yield request.param

    def test_metacommunity_similarity(self, test_case):
        metacommunity = Metacommunity(
            similarity=test_case["similarity"], abundance=test_case["abundance"]
        )
        metacommunity_similarity = metacommunity.metacommunity_similarity()
        if isinstance(metacommunity_similarity, SharedArrayView):
            metacommunity_similarity = metacommunity_similarity.data
        assert (
            metacommunity_similarity.shape
            == test_case["metacommunity_similarity"].shape
        )
        assert allclose(metacommunity_similarity, test_case["metacommunity_similarity"])

    def test_subcommunity_similarity(self, test_case):
        metacommunity = Metacommunity(
            similarity=test_case["similarity"], abundance=test_case["abundance"]
        )
        subcommunity_similarity = metacommunity.subcommunity_similarity()
        if isinstance(subcommunity_similarity, SharedArrayView):
            subcommunity_similarity = subcommunity_similarity.data

        assert (
            subcommunity_similarity.shape == test_case["subcommunity_similarity"].shape
        )
        assert allclose(subcommunity_similarity, test_case["subcommunity_similarity"])

    def test_normalized_subcommunity_similarity(self, test_case):
        metacommunity = Metacommunity(
            similarity=test_case["similarity"], abundance=test_case["abundance"]
        )
        normalized_subcommunity_similarity = (
            metacommunity.normalized_subcommunity_similarity()
        )
        if isinstance(normalized_subcommunity_similarity, SharedArrayView):
            normalized_subcommunity_similarity = normalized_subcommunity_similarity.data

        assert (
            normalized_subcommunity_similarity.shape
            == test_case["normalized_subcommunity_similarity"].shape
        )
        assert allclose(
            normalized_subcommunity_similarity,
            test_case["normalized_subcommunity_similarity"],
        )

    def test_subcommunity_alpha(self, test_case):
        metacommunity = Metacommunity(
            similarity=test_case["similarity"], abundance=test_case["abundance"]
        )
        subcommunity_alpha = metacommunity.subcommunity_measure(
            test_case["viewpoint"], measure="alpha"
        )

        assert subcommunity_alpha.shape == test_case["subcommunity_alpha"].shape
        assert allclose(subcommunity_alpha, test_case["subcommunity_alpha"])

    def test_subcommunity_rho(self, test_case):
        metacommunity = Metacommunity(
            similarity=test_case["similarity"], abundance=test_case["abundance"]
        )
        subcommunity_rho = metacommunity.subcommunity_measure(
            test_case["viewpoint"], measure="rho"
        )

        assert subcommunity_rho.shape == test_case["subcommunity_rho"].shape
        assert allclose(subcommunity_rho, test_case["subcommunity_rho"])

    def test_subcommunity_beta(self, test_case):
        metacommunity = Metacommunity(
            similarity=test_case["similarity"], abundance=test_case["abundance"]
        )
        subcommunity_beta = metacommunity.subcommunity_measure(
            test_case["viewpoint"], measure="beta"
        )

        assert subcommunity_beta.shape == test_case["subcommunity_beta"].shape
        assert allclose(subcommunity_beta, test_case["subcommunity_beta"])

    def test_subcommunity_gamma(self, test_case):
        metacommunity = Metacommunity(
            similarity=test_case["similarity"], abundance=test_case["abundance"]
        )
        subcommunity_gamma = metacommunity.subcommunity_measure(
            test_case["viewpoint"], measure="gamma"
        )

        assert subcommunity_gamma.shape == test_case["subcommunity_gamma"].shape
        assert allclose(subcommunity_gamma, test_case["subcommunity_gamma"])

    def test_normalized_subcommunity_alpha(self, test_case):
        metacommunity = Metacommunity(
            similarity=test_case["similarity"], abundance=test_case["abundance"]
        )
        normalized_subcommunity_alpha = metacommunity.subcommunity_measure(
            test_case["viewpoint"], measure="normalized_alpha"
        )

        assert (
            normalized_subcommunity_alpha.shape
            == test_case["normalized_subcommunity_alpha"].shape
        )
        assert allclose(
            normalized_subcommunity_alpha, test_case["normalized_subcommunity_alpha"]
        )

    def test_normalized_subcommunity_rho(self, test_case):
        metacommunity = Metacommunity(
            similarity=test_case["similarity"], abundance=test_case["abundance"]
        )
        normalized_subcommunity_rho = metacommunity.subcommunity_measure(
            test_case["viewpoint"], measure="normalized_rho"
        )

        assert (
            normalized_subcommunity_rho.shape
            == test_case["normalized_subcommunity_rho"].shape
        )
        assert allclose(
            normalized_subcommunity_rho, test_case["normalized_subcommunity_rho"]
        )

    def test_normalized_subcommunity_beta(self, test_case):
        metacommunity = Metacommunity(
            similarity=test_case["similarity"], abundance=test_case["abundance"]
        )
        normalized_subcommunity_beta = metacommunity.subcommunity_measure(
            test_case["viewpoint"], measure="normalized_beta"
        )

        assert (
            normalized_subcommunity_beta.shape
            == test_case["normalized_subcommunity_beta"].shape
        )
        assert allclose(
            normalized_subcommunity_beta, test_case["normalized_subcommunity_beta"]
        )

    def test_metacommunity_alpha(self, test_case):
        metacommunity = Metacommunity(
            similarity=test_case["similarity"], abundance=test_case["abundance"]
        )
        metacommunity_alpha = metacommunity.metacommunity_measure(
            test_case["viewpoint"], "alpha"
        )
        assert isclose(metacommunity_alpha, test_case["metacommunity_alpha"])

    def test_metacommunity_rho(self, test_case):
        metacommunity = Metacommunity(
            similarity=test_case["similarity"], abundance=test_case["abundance"]
        )
        metacommunity_rho = metacommunity.metacommunity_measure(
            test_case["viewpoint"], "rho"
        )
        assert isclose(metacommunity_rho, test_case["metacommunity_rho"])

    def test_metacommunity_beta(self, test_case):
        metacommunity = Metacommunity(
            similarity=test_case["similarity"], abundance=test_case["abundance"]
        )
        metacommunity_beta = metacommunity.metacommunity_measure(
            test_case["viewpoint"], "beta"
        )
        assert isclose(metacommunity_beta, test_case["metacommunity_beta"])

    def test_metacommunity_gamma(self, test_case):
        metacommunity = Metacommunity(
            similarity=test_case["similarity"], abundance=test_case["abundance"]
        )
        metacommunity_gamma = metacommunity.metacommunity_measure(
            test_case["viewpoint"], "gamma"
        )
        assert isclose(metacommunity_gamma, test_case["metacommunity_gamma"])

    def test_metacommunity_normalized_alpha(self, test_case):
        metacommunity = Metacommunity(
            similarity=test_case["similarity"], abundance=test_case["abundance"]
        )
        metacommunity_normalized_alpha = metacommunity.metacommunity_measure(
            test_case["viewpoint"], "normalized_alpha"
        )
        assert isclose(
            metacommunity_normalized_alpha, test_case["metacommunity_normalized_alpha"]
        )

    def test_metacommunity_normalized_rho(self, test_case):
        metacommunity = Metacommunity(
            similarity=test_case["similarity"], abundance=test_case["abundance"]
        )
        metacommunity_normalized_rho = metacommunity.metacommunity_measure(
            test_case["viewpoint"], "normalized_rho"
        )
        assert isclose(
            metacommunity_normalized_rho, test_case["metacommunity_normalized_rho"]
        )

    def test_metacommunity_normalized_beta(self, test_case):
        metacommunity = Metacommunity(
            similarity=test_case["similarity"], abundance=test_case["abundance"]
        )
        metacommunity_normalized_beta = metacommunity.metacommunity_measure(
            test_case["viewpoint"], "normalized_beta"
        )
        assert isclose(
            metacommunity_normalized_beta, test_case["metacommunity_normalized_beta"]
        )


# MAKE_METACOMMUNITY_TEST_CASES = [
#     {
#         "description": "SimilarityFromMemory strategy; simple use case",
#         "counts": DataFrame(
#             {
#                 "subcommunity": [
#                     "community_1",
#                     "community_1",
#                     "community_2",
#                     "community_2",
#                 ],
#                 "species": ["species_1", "species_2", "species_1", "species_3"],
#                 "count": [2, 3, 4, 1],
#             }
#         ),
#         "similarity_matrix": DataFrame(
#             data=array([[1.0, 0.5, 0.1], [0.5, 1.0, 0.2], [0.1, 0.2, 1.0]]),
#             columns=["species_1", "species_2", "species_3"],
#             index=["species_1", "species_2", "species_3"],
#         ),
#         "subcommunities": None,
#         "chunk_size": 1,
#         "subcommunity_column": "subcommunity",
#         "species_column": "species",
#         "count_column": "count",
#         "abundance": Abundance(
#             counts=DataFrame(
#                 {
#                     "subcommunity": [
#                         "community_1",
#                         "community_1",
#                         "community_2",
#                         "community_2",
#                     ],
#                     "species": ["species_1", "species_2", "species_1", "species_3"],
#                     "count": [2, 3, 4, 1],
#                 }
#             ),
#             species_order=array(["species_1", "species_2", "species_3"], dtype=object),
#             subcommunity_order=unique(
#                 array(
#                     ["community_1", "community_1", "community_2", "community_2"],
#                     dtype=object,
#                 )
#             ),
#             subcommunity_column="subcommunity",
#             species_column="species",
#             count_column="count",
#         ),
#         "similarity_type": SimilarityFromMemory,
#         "similarity_init_kwargs": {
#             "similarity_matrix": DataFrame(
#                 data=array([[1.0, 0.5, 0.1], [0.5, 1.0, 0.2], [0.1, 0.2, 1.0]]),
#                 columns=["species_1", "species_2", "species_3"],
#                 index=["species_1", "species_2", "species_3"],
#             ),
#             "species_subset": ["species_1", "species_2", "species_3"],
#         },
#     },
#     {
#         "description": "SimilarityFromMemory strategy; non-standard counts columns",
#         "counts": DataFrame(
#             {
#                 "subcommunity": [
#                     "community_1_",
#                     "community_1_",
#                     "community_2_",
#                     "community_2_",
#                 ],
#                 "species_subset": ["species_9", "species_7", "species_8", "species_6"],
#                 "count": [40, 1, 14, 21],
#                 "subcommunity_": [
#                     "community_1",
#                     "community_1",
#                     "community_2",
#                     "community_2",
#                 ],
#                 "species_": ["species_1", "species_2", "species_1", "species_3"],
#                 "count_": [2, 3, 4, 1],
#             }
#         ),
#         "similarity_matrix": DataFrame(
#             data=array([[1.0, 0.5, 0.1], [0.5, 1.0, 0.2], [0.1, 0.2, 1.0]]),
#             columns=["species_1", "species_2", "species_3"],
#             index=["species_1", "species_2", "species_3"],
#         ),
#         "subcommunities": None,
#         "chunk_size": 1,
#         "subcommunity_column": "subcommunity_",
#         "species_column": "species_",
#         "count_column": "count_",
#         "abundance": Abundance(
#             counts=DataFrame(
#                 {
#                     "subcommunity_": [
#                         "community_1",
#                         "community_1",
#                         "community_2",
#                         "community_2",
#                     ],
#                     "species_": ["species_1", "species_2", "species_1", "species_3"],
#                     "count_": [2, 3, 4, 1],
#                 }
#             ),
#             species_order=array(["species_1", "species_2", "species_3"], dtype=object),
#             subcommunity_order=unique(
#                 array(
#                     ["community_1", "community_1", "community_2", "community_2"],
#                     dtype=object,
#                 )
#             ),
#             subcommunity_column="subcommunity_",
#             species_column="species_",
#             count_column="count_",
#         ),
#         "similarity_type": SimilarityFromMemory,
#         "similarity_init_kwargs": {
#             "similarity_matrix": DataFrame(
#                 data=array([[1.0, 0.5, 0.1], [0.5, 1.0, 0.2], [0.1, 0.2, 1.0]]),
#                 columns=["species_1", "species_2", "species_3"],
#                 index=["species_1", "species_2", "species_3"],
#             ),
#             "species_subset": ["species_1", "species_2", "species_3"],
#         },
#     },
#     {
#         "description": "SimilarityFromMemory strategy; shuffled similarity matrix index",
#         "counts": DataFrame(
#             {
#                 "subcommunity": [
#                     "community_1",
#                     "community_1",
#                     "community_2",
#                     "community_2",
#                 ],
#                 "species": ["species_1", "species_2", "species_1", "species_3"],
#                 "count": [2, 3, 4, 1],
#             }
#         ),
#         "similarity_matrix": DataFrame(
#             data=array([[0.5, 1.0, 0.2], [1.0, 0.5, 0.1], [0.1, 0.2, 1.0]]),
#             columns=["species_1", "species_2", "species_3"],
#             index=["species_2", "species_1", "species_3"],
#         ),
#         "subcommunities": None,
#         "chunk_size": 1,
#         "subcommunity_column": "subcommunity",
#         "species_column": "species",
#         "count_column": "count",
#         "abundance": Abundance(
#             counts=DataFrame(
#                 {
#                     "subcommunity": [
#                         "community_1",
#                         "community_1",
#                         "community_2",
#                         "community_2",
#                     ],
#                     "species": ["species_1", "species_2", "species_1", "species_3"],
#                     "count": [2, 3, 4, 1],
#                 }
#             ),
#             species_order=array(["species_1", "species_2", "species_3"], dtype=object),
#             subcommunity_order=unique(
#                 array(
#                     ["community_1", "community_1", "community_2", "community_2"],
#                     dtype=object,
#                 )
#             ),
#             subcommunity_column="subcommunity",
#             species_column="species",
#             count_column="count",
#         ),
#         "similarity_type": SimilarityFromMemory,
#         "similarity_init_kwargs": {
#             "similarity_matrix": DataFrame(
#                 data=array([[1.0, 0.5, 0.1], [0.5, 1.0, 0.2], [0.1, 0.2, 1.0]]),
#                 columns=["species_1", "species_2", "species_3"],
#                 index=["species_1", "species_2", "species_3"],
#             ),
#             "species_subset": ["species_1", "species_2", "species_3"],
#         },
#     },
#     {
#         "description": "SimilarityFromMemory strategy; subcommunity subset",
#         "counts": DataFrame(
#             {
#                 "subcommunity": [
#                     "community_1",
#                     "community_1",
#                     "community_2",
#                     "community_2",
#                     "community_3",
#                     "community_3",
#                 ],
#                 "species": [
#                     "species_1",
#                     "species_2",
#                     "species_1",
#                     "species_3",
#                     "species_2",
#                     "species_4",
#                 ],
#                 "count": [2, 3, 4, 1, 3, 5],
#             }
#         ),
#         "similarity_matrix": DataFrame(
#             data=array(
#                 [
#                     [1.0, 0.5, 0.1, 0.4],
#                     [0.5, 1.0, 0.2, 0.3],
#                     [0.1, 0.2, 1.0, 0.9],
#                     [0.4, 0.3, 0.9, 1.0],
#                 ]
#             ),
#             columns=["species_1", "species_2", "species_3", "species_4"],
#             index=["species_1", "species_2", "species_3", "species_4"],
#         ),
#         "subcommunities": ["community_1", "community_3"],
#         "chunk_size": 1,
#         "subcommunity_column": "subcommunity",
#         "species_column": "species",
#         "count_column": "count",
#         "abundance": Abundance(
#             counts=DataFrame(
#                 {
#                     "subcommunity": [
#                         "community_1",
#                         "community_1",
#                         "community_3",
#                         "community_3",
#                     ],
#                     "species": [
#                         "species_1",
#                         "species_2",
#                         "species_2",
#                         "species_4",
#                     ],
#                     "count": [2, 3, 3, 5],
#                 }
#             ),
#             species_order=array(["species_1", "species_2", "species_4"], dtype=object),
#             subcommunity_order=unique(
#                 array(
#                     ["community_1", "community_1", "community_3", "community_3"],
#                     dtype=object,
#                 )
#             ),
#             subcommunity_column="subcommunity",
#             species_column="species",
#             count_column="count",
#         ),
#         "similarity_type": SimilarityFromMemory,
#         "similarity_init_kwargs": {
#             "similarity_matrix": DataFrame(
#                 data=array(
#                     [
#                         [1.0, 0.5, 0.4],
#                         [0.5, 1.0, 0.3],
#                         [0.4, 0.3, 1.0],
#                     ]
#                 ),
#                 columns=["species_1", "species_2", "species_4"],
#                 index=["species_1", "species_2", "species_4"],
#             ),
#             "species_subset": ["species_1", "species_2", "species_4"],
#         },
#     },
#     {
#         "description": "SimilarityFromFile strategy; simple use case",
#         "counts": DataFrame(
#             {
#                 "subcommunity": [
#                     "community_1",
#                     "community_1",
#                     "community_2",
#                     "community_2",
#                 ],
#                 "species": ["species_1", "species_2", "species_1", "species_3"],
#                 "count": [2, 3, 4, 1],
#             }
#         ),
#         "similarity_matrix": "foo_similarities.tsv",
#         "subcommunities": None,
#         "chunk_size": 1,
#         "subcommunity_column": "subcommunity",
#         "species_column": "species",
#         "count_column": "count",
#         "abundance": Abundance(
#             counts=DataFrame(
#                 {
#                     "subcommunity": [
#                         "community_1",
#                         "community_1",
#                         "community_2",
#                         "community_2",
#                     ],
#                     "species": ["species_1", "species_2", "species_1", "species_3"],
#                     "count": [2, 3, 4, 1],
#                 }
#             ),
#             species_order=array(["species_1", "species_2", "species_3"], dtype=object),
#             subcommunity_order=unique(
#                 array(
#                     ["community_1", "community_1", "community_2", "community_2"],
#                     dtype=object,
#                 )
#             ),
#             subcommunity_column="subcommunity",
#             species_column="species",
#             count_column="count",
#         ),
#         "similarity_type": SimilarityFromFile,
#         "similarity_init_kwargs": {
#             "similarity_matrix": "foo_similarities.tsv",
#             "species_subset": ["species_1", "species_2", "species_3"],
#             "chunk_size": 1,
#         },
#     },
#     {
#         "description": "SimilarityFromFile strategy; non-standard counts columns",
#         "counts": DataFrame(
#             {
#                 "subcommunity": [
#                     "community_1_",
#                     "community_1_",
#                     "community_2_",
#                     "community_2_",
#                 ],
#                 "species": ["species_9", "species_7", "species_8", "species_6"],
#                 "count": [40, 1, 14, 21],
#                 "subcommunity_": [
#                     "community_1",
#                     "community_1",
#                     "community_2",
#                     "community_2",
#                 ],
#                 "species_": ["species_1", "species_2", "species_1", "species_3"],
#                 "count_": [2, 3, 4, 1],
#             }
#         ),
#         "similarity_matrix": "bar_similarities.csv",
#         "subcommunities": None,
#         "chunk_size": 1,
#         "subcommunity_column": "subcommunity_",
#         "species_column": "species_",
#         "count_column": "count_",
#         "abundance": Abundance(
#             counts=DataFrame(
#                 {
#                     "subcommunity_": [
#                         "community_1",
#                         "community_1",
#                         "community_2",
#                         "community_2",
#                     ],
#                     "species_": ["species_1", "species_2", "species_1", "species_3"],
#                     "count_": [2, 3, 4, 1],
#                 }
#             ),
#             species_order=array(["species_1", "species_2", "species_3"], dtype=object),
#             subcommunity_order=unique(
#                 array(
#                     ["community_1", "community_1", "community_2", "community_2"],
#                     dtype=object,
#                 )
#             ),
#             subcommunity_column="subcommunity_",
#             species_column="species_",
#             count_column="count_",
#         ),
#         "similarity_type": SimilarityFromFile,
#         "similarity_init_kwargs": {
#             "similarity_matrix": "bar_similarities.csv",
#             "species_subset": ["species_1", "species_2", "species_3"],
#             "chunk_size": 1,
#         },
#     },
#     {
#         "description": "SimilarityFromFile strategy; non-default chunk_size",
#         "counts": DataFrame(
#             {
#                 "subcommunity": [
#                     "community_1",
#                     "community_1",
#                     "community_2",
#                     "community_2",
#                 ],
#                 "species": ["species_1", "species_2", "species_1", "species_3"],
#                 "count": [2, 3, 4, 1],
#             }
#         ),
#         "similarity_matrix": "foo_similarities.tsv",
#         "subcommunities": None,
#         "chunk_size": 10,
#         "subcommunity_column": "subcommunity",
#         "species_column": "species",
#         "count_column": "count",
#         "abundance": Abundance(
#             counts=DataFrame(
#                 {
#                     "subcommunity": [
#                         "community_1",
#                         "community_1",
#                         "community_2",
#                         "community_2",
#                     ],
#                     "species": ["species_1", "species_2", "species_1", "species_3"],
#                     "count": [2, 3, 4, 1],
#                 }
#             ),
#             species_order=array(["species_1", "species_2", "species_3"], dtype=object),
#             subcommunity_order=unique(
#                 array(
#                     ["community_1", "community_1", "community_2", "community_2"],
#                     dtype=object,
#                 )
#             ),
#             subcommunity_column="subcommunity",
#             species_column="species",
#             count_column="count",
#         ),
#         "similarity_type": SimilarityFromFile,
#         "similarity_init_kwargs": {
#             "similarity_matrix": "foo_similarities.tsv",
#             "species_subset": ["species_1", "species_2", "species_3"],
#             "chunk_size": 10,
#         },
#     },
#     {
#         "description": "SimilarityFromFile strategy; subcommunity subset",
#         "counts": DataFrame(
#             {
#                 "subcommunity": [
#                     "community_1",
#                     "community_1",
#                     "community_2",
#                     "community_2",
#                     "community_3",
#                     "community_3",
#                 ],
#                 "species": [
#                     "species_1",
#                     "species_2",
#                     "species_1",
#                     "species_3",
#                     "species_2",
#                     "species_4",
#                 ],
#                 "count": [2, 3, 4, 1, 3, 5],
#             }
#         ),
#         "similarity_matrix": "bar_similarities.csv",
#         "subcommunities": ["community_1", "community_3"],
#         "chunk_size": 1,
#         "subcommunity_column": "subcommunity",
#         "species_column": "species",
#         "count_column": "count",
#         "abundance": Abundance(
#             counts=DataFrame(
#                 {
#                     "subcommunity": [
#                         "community_1",
#                         "community_1",
#                         "community_3",
#                         "community_3",
#                     ],
#                     "species": [
#                         "species_1",
#                         "species_2",
#                         "species_2",
#                         "species_4",
#                     ],
#                     "count": [2, 3, 3, 5],
#                 }
#             ),
#             species_order=array(["species_1", "species_2", "species_4"], dtype=object),
#             subcommunity_order=unique(
#                 array(
#                     ["community_1", "community_1", "community_3", "community_3"],
#                     dtype=object,
#                 )
#             ),
#             subcommunity_column="subcommunity",
#             species_column="species",
#             count_column="count",
#         ),
#         "similarity_type": SimilarityFromFile,
#         "similarity_init_kwargs": {
#             "similarity_matrix": "bar_similarities.csv",
#             "species_subset": ["species_1", "species_2", "species_4"],
#             "chunk_size": 1,
#         },
#     },
# ]


# class TestMakeMetacommunity:
#     """Tests metacommunity.make_metacommunity."""

#     @mark.parametrize("test_case", MAKE_METACOMMUNITY_TEST_CASES)
#     def test_make_metacommunity(self, test_case, tmp_path):
#         """Tests make_metacommunity test cases."""
#         if test_case["similarity_type"] == SimilarityFromFile:
#             delimiter = get_file_delimiter(test_case["similarity_matrix"])
#             full_path = f"{tmp_path}/{test_case['similarity_matrix']}"
#             test_case["similarity_matrix"] = full_path
#             test_case["similarity_init_kwargs"]["similarity_matrix"] = full_path
#             with open(test_case["similarity_matrix"], "w") as file:
#                 file.write(delimiter.join(test_case["abundance"].species_order) + "\n")
#         similarity = test_case["similarity_type"](**test_case["similarity_init_kwargs"])
#         metacommunity = make_metacommunity(
#             counts=test_case["counts"],
#             similarity_matrix=test_case["similarity_matrix"],
#             subcommunities=test_case["subcommunities"],
#             chunk_size=test_case["chunk_size"],
#             subcommunity_column=test_case["subcommunity_column"],
#             species_column=test_case["species_column"],
#             count_column=test_case["count_column"],
#         )

#         # Test abundance attribute
#         assert (
#             metacommunity.abundance.subcommunity_column
#             == test_case["subcommunity_column"]
#         )
#         assert metacommunity.abundance.species_column == test_case["species_column"]
#         assert metacommunity.abundance.count_column == test_case["count_column"]
#         array_equal(
#             metacommunity.abundance.counts[
#                 [
#                     test_case["subcommunity_column"],
#                     test_case["species_column"],
#                     test_case["count_column"],
#                 ]
#             ].to_numpy(),
#             test_case["abundance"]
#             .counts[
#                 [
#                     test_case["subcommunity_column"],
#                     test_case["species_column"],
#                     test_case["count_column"],
#                 ]
#             ]
#             .to_numpy(),
#         )
#         assert array_equal(
#             metacommunity.abundance.species_order, test_case["abundance"].species_order
#         )
#         assert array_equal(
#             metacommunity.abundance.subcommunity_order,
#             test_case["abundance"].subcommunity_order,
#         )

#         # Test similarity attribute
#         assert isinstance(metacommunity.similarity, test_case["similarity_type"])
#         # Test common properties
#         assert (
#             metacommunity.similarity.species_order.shape
#             == similarity.species_order.shape
#         ), (
#             f"\nactual species_order: {metacommunity.similarity.species_order}"
#             f"\nexpected species_order: {similarity.species_order}."
#         )
#         assert (
#             metacommunity.similarity.species_order == similarity.species_order
#         ).all()

#         # Attributes specific to SimilarityFromMemory
#         if test_case["similarity_type"] == SimilarityFromMemory:
#             assert_frame_equal(
#                 metacommunity.similarity.similarity_matrix,
#                 similarity.similarity_matrix,
#             )
#         # SimilarityFromFile attributes
#         elif test_case["similarity_type"] == SimilarityFromFile:
#             assert (
#                 metacommunity.similarity.similarity_matrix
#                 == similarity.similarity_matrix
#             )
#             assert metacommunity.similarity.chunk_size == similarity.chunk_size
