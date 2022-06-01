"""Tests for diversity.metacommunity."""
from copy import deepcopy

from numpy import (
    allclose,
    array,
    array_equal,
    dtype,
    isclose,
    unique,
)
from pandas import DataFrame, Index
from pandas.testing import assert_frame_equal
from pytest import fixture, raises

from diversity.abundance import Abundance, SharedAbundance
from diversity.exceptions import InvalidArgumentError
from diversity.log import LOGGER
from diversity.metacommunity import (
    make_metacommunity,
    SimilarityInsensitiveMetacommunity,
    SimilaritySensitiveMetacommunity,
    SharedSimilaritySensitiveMetacommunity,
)
from diversity.shared import SharedArrayManager, SharedArrayView
from diversity.similarity import (
    ISimilarity,
    SimilarityFromFunction,
    SimilarityFromMemory,
)


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


class MockObject:
    pass


class MockClass:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeSimilarity:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.species_ordering = unique(kwargs["species_subset"])


class MockSimilarityInsensitiveMetacommunity(MockClass):
    pass


class MockSimilaritySensitiveMetacommunity(MockClass):
    pass


class MockSharedSimilaritySensitiveMetacommunity(MockClass):
    pass


def mock_make_abundance(**kwargs):
    return MockClass(**kwargs)


def mock_make_similarity(**kwargs):
    return FakeSimilarity(**kwargs)


MAKE_METACOMMUNITY_TEST_CASES = [
    {
        "description": "SimilarityInsensitiveMetacommunity",
        "counts": DataFrame(
            {
                "subcommunity": [
                    "subcommunity_1",
                    "subcommunity_1",
                    "subcommunity_2",
                    "subcommunity_2",
                    "subcommunity_2",
                    "subcommunity_3",
                ],
                "species": [
                    "species_1",
                    "species_2",
                    "species_1",
                    "species_2",
                    "species_3",
                    "species_3",
                ],
                "count": [2, 5, 4, 3, 2, 3],
            }
        ),
        "subcommunities": array(["subcommunity_1", "subcommunity_2", "subcommunity_3"]),
        "similarity": None,
        "subcommunity_column": "subcommunity",
        "species_column": "species",
        "count_column": "count",
        "shared_array_manager": False,
        "abundance_kwargs": {},
        "similarity_kwargs": {},
        "expect_raise": False,
        "expected_return_type": MockSimilarityInsensitiveMetacommunity,
        "expected_keywords": {"abundance", "subcommunity_ordering"},
        "expected_counts": array([[2, 4, 0], [5, 3, 0], [0, 2, 3]]),
        "expected_subcommunity_ordering": array(
            ["subcommunity_1", "subcommunity_2", "subcommunity_3"]
        ),
        "expected_species_subset": None,
    },
    {
        "description": "SimilarityInsensitiveMetacommunity; subset",
        "counts": DataFrame(
            {
                "subcommunity": [
                    "subcommunity_1",
                    "subcommunity_1",
                    "subcommunity_2",
                    "subcommunity_2",
                    "subcommunity_2",
                    "subcommunity_3",
                ],
                "species": [
                    "species_1",
                    "species_2",
                    "species_1",
                    "species_2",
                    "species_3",
                    "species_3",
                ],
                "count": [2, 5, 4, 3, 2, 3],
            }
        ),
        "subcommunities": array(["subcommunity_3", "subcommunity_1"]),
        "similarity": None,
        "subcommunity_column": "subcommunity",
        "species_column": "species",
        "count_column": "count",
        "shared_array_manager": False,
        "abundance_kwargs": {},
        "similarity_kwargs": {},
        "expect_raise": False,
        "expected_return_type": MockSimilarityInsensitiveMetacommunity,
        "expected_keywords": {"abundance", "subcommunity_ordering"},
        "expected_counts": array([[0, 2], [0, 5], [3, 0]]),
        "expected_subcommunity_ordering": array(["subcommunity_3", "subcommunity_1"]),
        "expected_species_subset": None,
    },
    {
        "description": "SimilarityInsensitiveMetacommunity; abundance kwargs",
        "counts": DataFrame(
            {
                "subcommunity": [
                    "subcommunity_1",
                    "subcommunity_1",
                    "subcommunity_2",
                    "subcommunity_2",
                    "subcommunity_2",
                    "subcommunity_3",
                ],
                "species": [
                    "species_1",
                    "species_2",
                    "species_1",
                    "species_2",
                    "species_3",
                    "species_3",
                ],
                "count": [2, 5, 4, 3, 2, 3],
            }
        ),
        "subcommunities": array(["subcommunity_1", "subcommunity_2", "subcommunity_3"]),
        "similarity": None,
        "subcommunity_column": "subcommunity",
        "species_column": "species",
        "count_column": "count",
        "shared_array_manager": False,
        "abundance_kwargs": {"mock": MockObject(), "object": MockObject()},
        "similarity_kwargs": {},
        "expect_raise": False,
        "expected_return_type": MockSimilarityInsensitiveMetacommunity,
        "expected_keywords": {"abundance", "subcommunity_ordering"},
        "expected_counts": array([[2, 4, 0], [5, 3, 0], [0, 2, 3]]),
        "expected_subcommunity_ordering": array(
            ["subcommunity_1", "subcommunity_2", "subcommunity_3"]
        ),
        "expected_species_subset": None,
    },
    {
        "description": "SimilaritySensitiveMetacommunity",
        "counts": DataFrame(
            {
                "subcommunity": [
                    "subcommunity_1",
                    "subcommunity_1",
                    "subcommunity_2",
                    "subcommunity_2",
                    "subcommunity_2",
                    "subcommunity_3",
                ],
                "species": [
                    "species_1",
                    "species_2",
                    "species_1",
                    "species_2",
                    "species_3",
                    "species_3",
                ],
                "count": [2, 5, 4, 3, 2, 3],
            }
        ),
        "subcommunities": array(["subcommunity_1", "subcommunity_2", "subcommunity_3"]),
        "similarity": DataFrame(
            data=array(
                [
                    [1, 0.5, 0.1],
                    [0.5, 1, 0.2],
                    [0.1, 0.2, 1],
                ]
            ),
            columns=["species_1", "species_2", "species_3"],
            index=["species_1", "species_2", "species_3"],
        ),
        "subcommunity_column": "subcommunity",
        "species_column": "species",
        "count_column": "count",
        "shared_array_manager": False,
        "abundance_kwargs": {},
        "similarity_kwargs": {},
        "expect_raise": False,
        "expected_return_type": MockSimilaritySensitiveMetacommunity,
        "expected_keywords": {"abundance", "subcommunity_ordering", "similarity"},
        "expected_counts": array([[2, 4, 0], [5, 3, 0], [0, 2, 3]]),
        "expected_subcommunity_ordering": array(
            ["subcommunity_1", "subcommunity_2", "subcommunity_3"]
        ),
        "expected_species_subset": array(["species_1", "species_2", "species_3"]),
    },
    {
        "description": "SharedSimilaritySensitiveMetacommunity; similarity kwargs",
        "counts": DataFrame(
            {
                "subcommunity": [
                    "subcommunity_1",
                    "subcommunity_1",
                    "subcommunity_2",
                    "subcommunity_2",
                    "subcommunity_2",
                    "subcommunity_3",
                ],
                "species": [
                    "species_1",
                    "species_2",
                    "species_1",
                    "species_2",
                    "species_3",
                    "species_3",
                ],
                "count": [2, 5, 4, 3, 2, 3],
            }
        ),
        "subcommunities": array(["subcommunity_1", "subcommunity_2", "subcommunity_3"]),
        "similarity": DataFrame(
            data=array(
                [
                    [1, 0.5, 0.1],
                    [0.5, 1, 0.2],
                    [0.1, 0.2, 1],
                ]
            ),
            columns=["species_1", "species_2", "species_3"],
            index=["species_1", "species_2", "species_3"],
        ),
        "subcommunity_column": "subcommunity",
        "species_column": "species",
        "count_column": "count",
        "shared_array_manager": False,
        "abundance_kwargs": {},
        "similarity_kwargs": {"mock": MockObject(), "object": MockObject()},
        "expect_raise": False,
        "expected_return_type": MockSimilaritySensitiveMetacommunity,
        "expected_keywords": {"abundance", "subcommunity_ordering", "similarity"},
        "expected_counts": array([[2, 4, 0], [5, 3, 0], [0, 2, 3]]),
        "expected_subcommunity_ordering": array(
            ["subcommunity_1", "subcommunity_2", "subcommunity_3"]
        ),
        "expected_species_subset": array(["species_1", "species_2", "species_3"]),
    },
    {
        "description": "SimilaritySensitiveMetacommunity; subset",
        "counts": DataFrame(
            {
                "subcommunity": [
                    "subcommunity_1",
                    "subcommunity_1",
                    "subcommunity_2",
                    "subcommunity_2",
                    "subcommunity_2",
                    "subcommunity_3",
                ],
                "species": [
                    "species_1",
                    "species_2",
                    "species_1",
                    "species_2",
                    "species_3",
                    "species_3",
                ],
                "count": [2, 5, 4, 3, 2, 3],
            }
        ),
        "subcommunities": array(["subcommunity_3"]),
        "similarity": DataFrame(
            data=array(
                [
                    [1, 0.5, 0.1],
                    [0.5, 1, 0.2],
                    [0.1, 0.2, 1],
                ]
            ),
            columns=["species_1", "species_2", "species_3"],
            index=["species_1", "species_2", "species_3"],
        ),
        "subcommunity_column": "subcommunity",
        "species_column": "species",
        "count_column": "count",
        "shared_array_manager": False,
        "abundance_kwargs": {},
        "similarity_kwargs": {},
        "expect_raise": False,
        "expected_return_type": MockSimilaritySensitiveMetacommunity,
        "expected_keywords": {"abundance", "subcommunity_ordering", "similarity"},
        "expected_counts": array([[3]]),
        "expected_subcommunity_ordering": array(["subcommunity_3"]),
        "expected_species_subset": array(["species_3"]),
    },
    {
        "description": "SimilaritySensitiveMetacommunity; abundance kwargs",
        "counts": DataFrame(
            {
                "subcommunity": [
                    "subcommunity_1",
                    "subcommunity_1",
                    "subcommunity_2",
                    "subcommunity_2",
                    "subcommunity_2",
                    "subcommunity_3",
                ],
                "species": [
                    "species_1",
                    "species_2",
                    "species_1",
                    "species_2",
                    "species_3",
                    "species_3",
                ],
                "count": [2, 5, 4, 3, 2, 3],
            }
        ),
        "subcommunities": array(["subcommunity_1", "subcommunity_2", "subcommunity_3"]),
        "similarity": DataFrame(
            data=array(
                [
                    [1, 0.5, 0.1],
                    [0.5, 1, 0.2],
                    [0.1, 0.2, 1],
                ]
            ),
            columns=["species_1", "species_2", "species_3"],
            index=["species_1", "species_2", "species_3"],
        ),
        "subcommunity_column": "subcommunity",
        "species_column": "species",
        "count_column": "count",
        "shared_array_manager": False,
        "abundance_kwargs": {"mock": MockObject(), "object": MockObject()},
        "similarity_kwargs": {},
        "expect_raise": False,
        "expected_return_type": MockSimilaritySensitiveMetacommunity,
        "expected_keywords": {"abundance", "subcommunity_ordering", "similarity"},
        "expected_counts": array([[2, 4, 0], [5, 3, 0], [0, 2, 3]]),
        "expected_subcommunity_ordering": array(
            ["subcommunity_1", "subcommunity_2", "subcommunity_3"]
        ),
        "expected_species_subset": array(["species_1", "species_2", "species_3"]),
    },
    {
        "description": "SharedSimilaritySensitiveMetacommunity",
        "counts": DataFrame(
            {
                "subcommunity": [
                    "subcommunity_1",
                    "subcommunity_1",
                    "subcommunity_2",
                    "subcommunity_2",
                    "subcommunity_2",
                    "subcommunity_3",
                ],
                "species": [
                    "species_1",
                    "species_2",
                    "species_1",
                    "species_2",
                    "species_3",
                    "species_3",
                ],
                "count": [2, 5, 4, 3, 2, 3],
            }
        ),
        "subcommunities": array(["subcommunity_1", "subcommunity_2", "subcommunity_3"]),
        "similarity": DataFrame(
            data=array(
                [
                    [1, 0.5, 0.1],
                    [0.5, 1, 0.2],
                    [0.1, 0.2, 1],
                ]
            ),
            columns=["species_1", "species_2", "species_3"],
            index=["species_1", "species_2", "species_3"],
        ),
        "subcommunity_column": "subcommunity",
        "species_column": "species",
        "count_column": "count",
        "shared_array_manager": True,
        "abundance_kwargs": {},
        "similarity_kwargs": {},
        "expect_raise": False,
        "expected_return_type": MockSharedSimilaritySensitiveMetacommunity,
        "expected_keywords": {
            "abundance",
            "subcommunity_ordering",
            "similarity",
            "shared_array_manager",
        },
        "expected_counts": array([[2, 4, 0], [5, 3, 0], [0, 2, 3]]),
        "expected_subcommunity_ordering": array(
            ["subcommunity_1", "subcommunity_2", "subcommunity_3"]
        ),
        "expected_species_subset": array(["species_1", "species_2", "species_3"]),
    },
    {
        "description": "SharedSimilaritySensitiveMetacommunity; similarity kwargs",
        "counts": DataFrame(
            {
                "subcommunity": [
                    "subcommunity_1",
                    "subcommunity_1",
                    "subcommunity_2",
                    "subcommunity_2",
                    "subcommunity_2",
                    "subcommunity_3",
                ],
                "species": [
                    "species_1",
                    "species_2",
                    "species_1",
                    "species_2",
                    "species_3",
                    "species_3",
                ],
                "count": [2, 5, 4, 3, 2, 3],
            }
        ),
        "subcommunities": array(["subcommunity_1", "subcommunity_2", "subcommunity_3"]),
        "similarity": DataFrame(
            data=array(
                [
                    [1, 0.5, 0.1],
                    [0.5, 1, 0.2],
                    [0.1, 0.2, 1],
                ]
            ),
            columns=["species_1", "species_2", "species_3"],
            index=["species_1", "species_2", "species_3"],
        ),
        "subcommunity_column": "subcommunity",
        "species_column": "species",
        "count_column": "count",
        "shared_array_manager": True,
        "abundance_kwargs": {},
        "similarity_kwargs": {"mock": MockObject(), "object": MockObject()},
        "expect_raise": False,
        "expected_return_type": MockSharedSimilaritySensitiveMetacommunity,
        "expected_keywords": {
            "abundance",
            "subcommunity_ordering",
            "similarity",
            "shared_array_manager",
        },
        "expected_counts": array([[2, 4, 0], [5, 3, 0], [0, 2, 3]]),
        "expected_subcommunity_ordering": array(
            ["subcommunity_1", "subcommunity_2", "subcommunity_3"]
        ),
        "expected_species_subset": array(["species_1", "species_2", "species_3"]),
    },
    {
        "description": "SharedSimilaritySensitiveMetacommunity; subset",
        "counts": DataFrame(
            {
                "subcommunity": [
                    "subcommunity_1",
                    "subcommunity_1",
                    "subcommunity_2",
                    "subcommunity_2",
                    "subcommunity_2",
                    "subcommunity_3",
                ],
                "species": [
                    "species_1",
                    "species_2",
                    "species_1",
                    "species_2",
                    "species_3",
                    "species_3",
                ],
                "count": [2, 5, 4, 3, 2, 3],
            }
        ),
        "subcommunities": array(["subcommunity_3"]),
        "similarity": DataFrame(
            data=array(
                [
                    [1, 0.5, 0.1],
                    [0.5, 1, 0.2],
                    [0.1, 0.2, 1],
                ]
            ),
            columns=["species_1", "species_2", "species_3"],
            index=["species_1", "species_2", "species_3"],
        ),
        "subcommunity_column": "subcommunity",
        "species_column": "species",
        "count_column": "count",
        "shared_array_manager": True,
        "abundance_kwargs": {},
        "similarity_kwargs": {},
        "expect_raise": False,
        "expected_return_type": MockSharedSimilaritySensitiveMetacommunity,
        "expected_keywords": {
            "abundance",
            "subcommunity_ordering",
            "similarity",
            "shared_array_manager",
        },
        "expected_counts": array([[3]]),
        "expected_subcommunity_ordering": array(["subcommunity_3"]),
        "expected_species_subset": array(["species_3"]),
    },
    {
        "description": "SharedSimilaritySensitiveMetacommunity; abundance kwargs",
        "counts": DataFrame(
            {
                "subcommunity": [
                    "subcommunity_1",
                    "subcommunity_1",
                    "subcommunity_2",
                    "subcommunity_2",
                    "subcommunity_2",
                    "subcommunity_3",
                ],
                "species": [
                    "species_1",
                    "species_2",
                    "species_1",
                    "species_2",
                    "species_3",
                    "species_3",
                ],
                "count": [2, 5, 4, 3, 2, 3],
            }
        ),
        "subcommunities": array(["subcommunity_1", "subcommunity_2", "subcommunity_3"]),
        "similarity": DataFrame(
            data=array(
                [
                    [1, 0.5, 0.1],
                    [0.5, 1, 0.2],
                    [0.1, 0.2, 1],
                ]
            ),
            columns=["species_1", "species_2", "species_3"],
            index=["species_1", "species_2", "species_3"],
        ),
        "subcommunity_column": "subcommunity",
        "species_column": "species",
        "count_column": "count",
        "shared_array_manager": True,
        "abundance_kwargs": {"mock": MockObject(), "object": MockObject()},
        "similarity_kwargs": {},
        "expect_raise": False,
        "expected_return_type": MockSharedSimilaritySensitiveMetacommunity,
        "expected_keywords": {
            "abundance",
            "subcommunity_ordering",
            "similarity",
            "shared_array_manager",
        },
        "expected_counts": array([[2, 4, 0], [5, 3, 0], [0, 2, 3]]),
        "expected_subcommunity_ordering": array(
            ["subcommunity_1", "subcommunity_2", "subcommunity_3"]
        ),
        "expected_species_subset": array(["species_1", "species_2", "species_3"]),
    },
    {
        "description": "Non-default subcommunity column",
        "counts": DataFrame(
            {
                "subcommunity_": [
                    "subcommunity_1",
                    "subcommunity_1",
                    "subcommunity_2",
                    "subcommunity_2",
                    "subcommunity_2",
                    "subcommunity_3",
                ],
                "subcommunity": [
                    "subcommunity_1",
                    "subcommunity_4",
                    "subcommunity_4",
                    "subcommunity_5",
                    "subcommunity_6",
                    "subcommunity_7",
                ],
                "species": [
                    "species_1",
                    "species_2",
                    "species_1",
                    "species_2",
                    "species_3",
                    "species_3",
                ],
                "count": [2, 5, 4, 3, 2, 3],
            }
        ),
        "subcommunities": array(["subcommunity_1", "subcommunity_2", "subcommunity_3"]),
        "similarity": None,
        "subcommunity_column": "subcommunity_",
        "species_column": "species",
        "count_column": "count",
        "shared_array_manager": False,
        "abundance_kwargs": {},
        "similarity_kwargs": {},
        "expect_raise": False,
        "expected_return_type": MockSimilarityInsensitiveMetacommunity,
        "expected_keywords": {"abundance", "subcommunity_ordering"},
        "expected_counts": array([[2, 4, 0], [5, 3, 0], [0, 2, 3]]),
        "expected_subcommunity_ordering": array(
            ["subcommunity_1", "subcommunity_2", "subcommunity_3"]
        ),
        "expected_species_subset": None,
    },
    {
        "description": "Non-default species column",
        "counts": DataFrame(
            {
                "species_": [
                    "species_1",
                    "species_2",
                    "species_1",
                    "species_2",
                    "species_3",
                    "species_3",
                ],
                "subcommunity": [
                    "subcommunity_1",
                    "subcommunity_1",
                    "subcommunity_2",
                    "subcommunity_2",
                    "subcommunity_2",
                    "subcommunity_3",
                ],
                "species": [
                    "species_1",
                    "species_4",
                    "species_1",
                    "species_2",
                    "species_2",
                    "species_3",
                ],
                "count": [2, 5, 4, 3, 2, 3],
            }
        ),
        "subcommunities": array(["subcommunity_1", "subcommunity_2", "subcommunity_3"]),
        "similarity": None,
        "subcommunity_column": "subcommunity",
        "species_column": "species_",
        "count_column": "count",
        "shared_array_manager": False,
        "abundance_kwargs": {},
        "similarity_kwargs": {},
        "expect_raise": False,
        "expected_return_type": MockSimilarityInsensitiveMetacommunity,
        "expected_keywords": {"abundance", "subcommunity_ordering"},
        "expected_counts": array([[2, 4, 0], [5, 3, 0], [0, 2, 3]]),
        "expected_subcommunity_ordering": array(
            ["subcommunity_1", "subcommunity_2", "subcommunity_3"]
        ),
        "expected_species_subset": None,
    },
    {
        "description": "Non-default count column",
        "counts": DataFrame(
            {
                "count_": [2, 5, 4, 3, 2, 3],
                "subcommunity": [
                    "subcommunity_1",
                    "subcommunity_1",
                    "subcommunity_2",
                    "subcommunity_2",
                    "subcommunity_2",
                    "subcommunity_3",
                ],
                "species": [
                    "species_1",
                    "species_2",
                    "species_1",
                    "species_2",
                    "species_3",
                    "species_3",
                ],
                "count": [4, 7, 3, 5, 6, 12],
            }
        ),
        "subcommunities": array(["subcommunity_1", "subcommunity_2", "subcommunity_3"]),
        "similarity": None,
        "subcommunity_column": "subcommunity",
        "species_column": "species",
        "count_column": "count_",
        "shared_array_manager": False,
        "abundance_kwargs": {},
        "similarity_kwargs": {},
        "expect_raise": False,
        "expected_return_type": MockSimilarityInsensitiveMetacommunity,
        "expected_keywords": {"abundance", "subcommunity_ordering"},
        "expected_counts": array([[2, 4, 0], [5, 3, 0], [0, 2, 3]]),
        "expected_subcommunity_ordering": array(
            ["subcommunity_1", "subcommunity_2", "subcommunity_3"]
        ),
        "expected_species_subset": None,
    },
]


class TestMakeMetacommunity:
    @fixture(params=MAKE_METACOMMUNITY_TEST_CASES)
    def test_case(self, request, monkeypatch, shared_array_manager):
        with monkeypatch.context() as patched_context:
            for target, mocked in [
                (
                    "diversity.metacommunity.SimilarityInsensitiveMetacommunity",
                    MockSimilarityInsensitiveMetacommunity,
                ),
                (
                    "diversity.metacommunity.SimilaritySensitiveMetacommunity",
                    MockSimilaritySensitiveMetacommunity,
                ),
                (
                    "diversity.metacommunity.SharedSimilaritySensitiveMetacommunity",
                    MockSharedSimilaritySensitiveMetacommunity,
                ),
                ("diversity.metacommunity.make_abundance", mock_make_abundance),
                ("diversity.metacommunity.make_similarity", mock_make_similarity),
                ("diversity.metacommunity.ISimilarity", FakeSimilarity),
            ]:
                patched_context.setattr(target, mocked)
            test_case_ = deepcopy(request.param)
            if request.param["shared_array_manager"]:
                test_case_["shared_array_manager"] = shared_array_manager
            else:
                test_case_["shared_array_manager"] = None
            if "shared_array_manager" in request.param["abundance_kwargs"]:
                request.param["abundance_kwargs"][
                    "shared_array_manager"
                ] = shared_array_manager
            yield test_case_

    def test_make_metacommunity(self, test_case):
        if test_case["expect_raise"]:
            with raises(InvalidArgumentError):
                make_metacommunity(
                    counts=test_case["counts"],
                    subcommunities=test_case["subcommunities"],
                    similarity=test_case["similarity"],
                    subcommunity_column=test_case["subcommunity_column"],
                    species_column=test_case["species_column"],
                    count_column=test_case["count_column"],
                    shared_array_manager=test_case["shared_array_manager"],
                    abundance_kwargs=test_case["abundance_kwargs"],
                    similarity_kwargs=test_case["similarity_kwargs"],
                )
        else:
            metacommunity = make_metacommunity(
                counts=test_case["counts"],
                subcommunities=test_case["subcommunities"],
                similarity=test_case["similarity"],
                subcommunity_column=test_case["subcommunity_column"],
                species_column=test_case["species_column"],
                count_column=test_case["count_column"],
                shared_array_manager=test_case["shared_array_manager"],
                abundance_kwargs=test_case["abundance_kwargs"],
                similarity_kwargs=test_case["similarity_kwargs"],
            )
            assert isinstance(metacommunity, test_case["expected_return_type"])
            assert set(metacommunity.kwargs.keys()) == test_case["expected_keywords"]
            assert array_equal(
                metacommunity.kwargs["abundance"].kwargs["counts"],
                test_case["expected_counts"],
            )
            assert set(metacommunity.kwargs["abundance"].kwargs.keys()) == {
                "counts",
                *test_case["abundance_kwargs"].keys(),
            }
            for key, arg in test_case["abundance_kwargs"].items():
                assert metacommunity.kwargs["abundance"].kwargs[key] is arg
            assert array_equal(
                metacommunity.kwargs["subcommunity_ordering"],
                test_case["expected_subcommunity_ordering"],
            )
            if test_case["shared_array_manager"] is not None:
                assert (
                    metacommunity.kwargs["shared_array_manager"]
                    is test_case["shared_array_manager"]
                )
            if test_case["similarity"] is not None:
                assert (
                    metacommunity.kwargs["similarity"].kwargs["similarity"]
                    is test_case["similarity"]
                )
                array_equal(
                    metacommunity.kwargs["similarity"].kwargs["species_subset"],
                    test_case["expected_species_subset"],
                )
                assert set(metacommunity.kwargs["similarity"].kwargs.keys()) == {
                    "similarity",
                    "species_subset",
                    *test_case["similarity_kwargs"].keys(),
                }
                for key, arg in test_case["similarity_kwargs"].items():
                    assert metacommunity.kwargs["similarity"].kwargs[key] is arg


SIMILARITY_INSENSITIVE_METACOMMUNITY_TEST_CASES = [
    {
        "description": "disjoint communities; uniform counts; viewpoint 0.",
        "abundance": Abundance(
            counts=array([[1, 0], [1, 0], [1, 0], [0, 1], [0, 1], [0, 1]])
        ),
        "subcommunity_ordering": array(["subcommunity_1", "subcommunity_2"]),
        "viewpoint": 0,
        "subcommunity_alpha": array([6.0, 6.0]),
        "subcommunity_rho": array([1.0, 1.0]),
        "subcommunity_beta": array([1.0, 1.0]),
        "subcommunity_gamma": array([6.0, 6.0]),
        "normalized_subcommunity_alpha": array([3.0, 3.0]),
        "normalized_subcommunity_rho": array([0.5, 0.5]),
        "normalized_subcommunity_beta": array([2.0, 2.0]),
        "metacommunity_alpha": 6.0,
        "metacommunity_rho": 1.0,
        "metacommunity_beta": 1.0,
        "metacommunity_gamma": 6.0,
        "metacommunity_normalized_alpha": 3.0,
        "metacommunity_normalized_rho": 0.5,
        "metacommunity_normalized_beta": 2.0,
        "subcommunity_dataframe": DataFrame(
            {
                "alpha": array([6.0, 6.0]),
                "rho": array([1.0, 1.0]),
                "beta": array([1.0, 1.0]),
                "gamma": array([6.0, 6.0]),
                "normalized_alpha": array([3.0, 3.0]),
                "normalized_rho": array([0.5, 0.5]),
                "normalized_beta": array([2.0, 2.0]),
                "viewpoint": [0, 0],
                "community": array(["subcommunity_1", "subcommunity_2"]),
            },
        ),
        "metacommunity_dataframe": DataFrame(
            {
                "alpha": [6.0],
                "rho": [1.0],
                "beta": [1.0],
                "gamma": [6.0],
                "normalized_alpha": [3.0],
                "normalized_rho": [0.5],
                "normalized_beta": [2.0],
                "viewpoint": [0],
                "community": ["metacommunity"],
            },
            index=range(1),
        ),
    },
    {
        "description": "overlapping communities; non-uniform counts; viewpoint 2.",
        "abundance": {"counts": array([[1, 5], [3, 0], [0, 1]], dtype=dtype("f8"))},
        "subcommunity_ordering": array(["subcommunity_1", "subcommunity_2"]),
        "viewpoint": 2,
        "subcommunity_alpha": array([4.0, 2.30769231]),
        "subcommunity_rho": array([1.26315789, 1.16129032]),
        "subcommunity_beta": array([0.79166667, 0.86111111]),
        "subcommunity_gamma": array([2.66666667, 1.93548387]),
        "normalized_subcommunity_alpha": array([1.6, 1.38461538]),
        "normalized_subcommunity_rho": array([0.50526316, 0.69677419]),
        "normalized_subcommunity_beta": array([1.97916667, 1.43518519]),
        "metacommunity_alpha": 2.7777777777777777,
        "metacommunity_rho": 1.2,
        "metacommunity_beta": 0.8319209039548022,
        "metacommunity_gamma": 2.173913043478261,
        "metacommunity_normalized_alpha": 1.4634146341463414,
        "metacommunity_normalized_rho": 0.6050420168067228,
        "metacommunity_normalized_beta": 1.612461673236969,
        "subcommunity_dataframe": DataFrame(
            {
                "alpha": array([4.0, 2.30769231]),
                "rho": array([1.26315789, 1.16129032]),
                "beta": array([0.79166667, 0.86111111]),
                "gamma": array([2.66666667, 1.93548387]),
                "normalized_alpha": array([1.6, 1.38461538]),
                "normalized_rho": array([0.50526316, 0.69677419]),
                "normalized_beta": array([1.97916667, 1.43518519]),
                "viewpoint": [2, 2],
                "community": array(["subcommunity_1", "subcommunity_2"]),
            }
        ),
        "metacommunity_dataframe": DataFrame(
            {
                "alpha": [2.7777777777777777],
                "rho": [1.2],
                "beta": [0.8319209039548022],
                "gamma": [2.173913043478261],
                "normalized_alpha": [1.4634146341463414],
                "normalized_rho": [0.6050420168067228],
                "normalized_beta": [1.612461673236969],
                "viewpoint": [2],
                "community": ["metacommunity"],
            },
            index=range(1),
        ),
    },
]


class TestSimilarityInsensitiveMetacommunity:
    @fixture(params=SIMILARITY_INSENSITIVE_METACOMMUNITY_TEST_CASES, scope="class")
    def test_case(self, request):
        test_case_ = deepcopy(request.param)
        with SharedArrayManager() as shared_array_manager:
            if isinstance(request.param["abundance"], dict):
                counts = shared_array_manager.from_array(
                    request.param["abundance"]["counts"]
                )
                abundance = SharedAbundance(
                    counts=counts, shared_array_manager=shared_array_manager
                )
                test_case_["abundance"] = abundance

            yield test_case_

    def test_subcommunity_alpha(self, test_case):
        metacommunity = SimilarityInsensitiveMetacommunity(
            abundance=test_case["abundance"],
            subcommunity_ordering=test_case["subcommunity_ordering"],
        )
        subcommunity_alpha = metacommunity.subcommunity_diversity(
            test_case["viewpoint"], measure="alpha"
        )

        assert subcommunity_alpha.shape == test_case["subcommunity_alpha"].shape
        assert allclose(subcommunity_alpha, test_case["subcommunity_alpha"])

    def test_subcommunity_rho(self, test_case):
        metacommunity = SimilarityInsensitiveMetacommunity(
            abundance=test_case["abundance"],
            subcommunity_ordering=test_case["subcommunity_ordering"],
        )
        subcommunity_rho = metacommunity.subcommunity_diversity(
            test_case["viewpoint"], measure="rho"
        )

        assert subcommunity_rho.shape == test_case["subcommunity_rho"].shape
        assert allclose(subcommunity_rho, test_case["subcommunity_rho"])

    def test_subcommunity_beta(self, test_case):
        metacommunity = SimilarityInsensitiveMetacommunity(
            abundance=test_case["abundance"],
            subcommunity_ordering=test_case["subcommunity_ordering"],
        )
        subcommunity_beta = metacommunity.subcommunity_diversity(
            test_case["viewpoint"], measure="beta"
        )

        assert subcommunity_beta.shape == test_case["subcommunity_beta"].shape
        assert allclose(subcommunity_beta, test_case["subcommunity_beta"])

    def test_subcommunity_gamma(self, test_case):
        metacommunity = SimilarityInsensitiveMetacommunity(
            abundance=test_case["abundance"],
            subcommunity_ordering=test_case["subcommunity_ordering"],
        )
        subcommunity_gamma = metacommunity.subcommunity_diversity(
            test_case["viewpoint"], measure="gamma"
        )

        assert subcommunity_gamma.shape == test_case["subcommunity_gamma"].shape
        assert allclose(subcommunity_gamma, test_case["subcommunity_gamma"])

    def test_normalized_subcommunity_alpha(self, test_case):
        metacommunity = SimilarityInsensitiveMetacommunity(
            abundance=test_case["abundance"],
            subcommunity_ordering=test_case["subcommunity_ordering"],
        )
        normalized_subcommunity_alpha = metacommunity.subcommunity_diversity(
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
        metacommunity = SimilarityInsensitiveMetacommunity(
            abundance=test_case["abundance"],
            subcommunity_ordering=test_case["subcommunity_ordering"],
        )
        normalized_subcommunity_rho = metacommunity.subcommunity_diversity(
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
        metacommunity = SimilarityInsensitiveMetacommunity(
            abundance=test_case["abundance"],
            subcommunity_ordering=test_case["subcommunity_ordering"],
        )
        normalized_subcommunity_beta = metacommunity.subcommunity_diversity(
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
        metacommunity = SimilarityInsensitiveMetacommunity(
            abundance=test_case["abundance"],
            subcommunity_ordering=test_case["subcommunity_ordering"],
        )
        metacommunity_alpha = metacommunity.metacommunity_diversity(
            test_case["viewpoint"], "alpha"
        )
        assert isclose(metacommunity_alpha, test_case["metacommunity_alpha"])

    def test_metacommunity_rho(self, test_case):
        metacommunity = SimilarityInsensitiveMetacommunity(
            abundance=test_case["abundance"],
            subcommunity_ordering=test_case["subcommunity_ordering"],
        )
        metacommunity_rho = metacommunity.metacommunity_diversity(
            test_case["viewpoint"], "rho"
        )
        assert isclose(metacommunity_rho, test_case["metacommunity_rho"])

    def test_metacommunity_beta(self, test_case):
        metacommunity = SimilarityInsensitiveMetacommunity(
            abundance=test_case["abundance"],
            subcommunity_ordering=test_case["subcommunity_ordering"],
        )
        metacommunity_beta = metacommunity.metacommunity_diversity(
            test_case["viewpoint"], "beta"
        )
        assert isclose(metacommunity_beta, test_case["metacommunity_beta"])

    def test_metacommunity_gamma(self, test_case):
        metacommunity = SimilarityInsensitiveMetacommunity(
            abundance=test_case["abundance"],
            subcommunity_ordering=test_case["subcommunity_ordering"],
        )
        metacommunity_gamma = metacommunity.metacommunity_diversity(
            test_case["viewpoint"], "gamma"
        )
        assert isclose(metacommunity_gamma, test_case["metacommunity_gamma"])

    def test_metacommunity_normalized_alpha(self, test_case):
        metacommunity = SimilarityInsensitiveMetacommunity(
            abundance=test_case["abundance"],
            subcommunity_ordering=test_case["subcommunity_ordering"],
        )
        metacommunity_normalized_alpha = metacommunity.metacommunity_diversity(
            test_case["viewpoint"], "normalized_alpha"
        )
        assert isclose(
            metacommunity_normalized_alpha, test_case["metacommunity_normalized_alpha"]
        )

    def test_metacommunity_normalized_rho(self, test_case):
        metacommunity = SimilarityInsensitiveMetacommunity(
            abundance=test_case["abundance"],
            subcommunity_ordering=test_case["subcommunity_ordering"],
        )
        metacommunity_normalized_rho = metacommunity.metacommunity_diversity(
            test_case["viewpoint"], "normalized_rho"
        )
        assert isclose(
            metacommunity_normalized_rho, test_case["metacommunity_normalized_rho"]
        )

    def test_metacommunity_normalized_beta(self, test_case):
        metacommunity = SimilarityInsensitiveMetacommunity(
            abundance=test_case["abundance"],
            subcommunity_ordering=test_case["subcommunity_ordering"],
        )
        metacommunity_normalized_beta = metacommunity.metacommunity_diversity(
            test_case["viewpoint"], "normalized_beta"
        )
        assert isclose(
            metacommunity_normalized_beta, test_case["metacommunity_normalized_beta"]
        )

    def test_subcommunities_to_dataframe(self, test_case):
        metacommunity = SimilarityInsensitiveMetacommunity(
            abundance=test_case["abundance"],
            subcommunity_ordering=test_case["subcommunity_ordering"],
        )
        subcommunity_dataframe = metacommunity.subcommunities_to_dataframe(
            viewpoint=test_case["viewpoint"]
        )
        expected_subcommunity_dataframe = test_case["subcommunity_dataframe"][
            subcommunity_dataframe.columns
        ]
        assert_frame_equal(subcommunity_dataframe, expected_subcommunity_dataframe)

    def test_metacommunities_to_dataframe(self, test_case):
        metacommunity = SimilarityInsensitiveMetacommunity(
            abundance=test_case["abundance"],
            subcommunity_ordering=test_case["subcommunity_ordering"],
        )
        metacommunity_dataframe = metacommunity.metacommunity_to_dataframe(
            viewpoint=test_case["viewpoint"]
        )
        expected_metacommunity_dataframe = test_case["metacommunity_dataframe"][
            metacommunity_dataframe.columns
        ]
        assert_frame_equal(metacommunity_dataframe, expected_metacommunity_dataframe)


SIMILARITY_SENSITIVE_METACOMMUNITY_TEST_CASES = [
    {
        "description": "disjoint communities; uniform counts; uniform inter-community similarities; viewpoint 0.",
        "abundance": Abundance(
            counts=array([[1, 0], [1, 0], [1, 0], [0, 1], [0, 1], [0, 1]])
        ),
        "subcommunity_ordering": array(["subcommunity_1", "subcommunity_2"]),
        "similarity": SimilarityFromMemory(
            similarity=DataFrame(
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
        "subcommunity_dataframe": DataFrame(
            {
                "alpha": array([3.0, 3.0]),
                "rho": array([2.05, 2.05]),
                "beta": array([0.487805, 0.487805]),
                "gamma": array([1.463415, 1.463415]),
                "normalized_alpha": array([1.5, 1.5]),
                "normalized_rho": array([1.025, 1.025]),
                "normalized_beta": array([0.97561, 0.97561]),
                "viewpoint": [0, 0],
                "community": array(["subcommunity_1", "subcommunity_2"]),
            }
        ),
        "metacommunity_dataframe": DataFrame(
            {
                "alpha": [3.0],
                "rho": [2.05],
                "beta": [0.487805],
                "gamma": [1.463415],
                "normalized_alpha": [1.5],
                "normalized_rho": [1.025],
                "normalized_beta": [0.97561],
                "viewpoint": 0,
                "community": ["metacommunity"],
            },
            index=range(1),
        ),
    },
    {
        "description": "overlapping communities; non-uniform counts; non-uniform inter-community similarities; viewpoint 2.",
        "abundance": {"counts": array([[1, 5], [3, 0], [0, 1]], dtype=dtype("f8"))},
        "subcommunity_ordering": array(["subcommunity_1", "subcommunity_2"]),
        "similarity": SimilarityFromMemory(
            similarity=DataFrame(
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
        "subcommunity_dataframe": DataFrame(
            {
                "alpha": array([3.07692308, 2.22222222]),
                "rho": array([1.97775446, 1.48622222]),
                "beta": array([0.50562394, 0.67284689]),
                "gamma": array([1.52671756, 1.49253731]),
                "normalized_alpha": array([1.23076923, 1.33333333]),
                "normalized_rho": array([0.79110178, 0.89173333]),
                "normalized_beta": array([1.26405985, 1.12141148]),
                "viewpoint": [2, 2],
                "community": array(["subcommunity_1", "subcommunity_2"]),
            }
        ),
        "metacommunity_dataframe": DataFrame(
            {
                "alpha": [2.5],
                "rho": [1.6502801833927663],
                "beta": [0.5942352817544037],
                "gamma": [1.5060240963855422],
                "normalized_alpha": [1.2903225806451613],
                "normalized_rho": [0.8485572790897555],
                "normalized_beta": [1.1744247216675028],
                "viewpoint": [2],
                "community": ["metacommunity"],
            },
            index=range(1),
        ),
    },
    {
        "description": "similarity from function; default num_processors.",
        "abundance": Abundance(
            counts=array([[1, 5], [3, 0], [0, 1]], dtype=dtype("f8"))
        ),
        "subcommunity_ordering": array(["subcommunity_1", "subcommunity_2"]),
        "similarity": {
            "similarity_function": sim_func,
            "features": array(
                [
                    ["species_1"],
                    ["species_2"],
                    ["species_3"],
                ]
            ),
            "species_ordering": Index(
                [
                    "species_1",
                    "species_2",
                    "species_3",
                ]
            ),
            "num_processors": None,
        },
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
        "subcommunity_dataframe": DataFrame(
            {
                "alpha": array([3.07692308, 2.22222222]),
                "rho": array([1.97775446, 1.48622222]),
                "beta": array([0.50562394, 0.67284689]),
                "gamma": array([1.52671756, 1.49253731]),
                "normalized_alpha": array([1.23076923, 1.33333333]),
                "normalized_rho": array([0.79110178, 0.89173333]),
                "normalized_beta": array([1.26405985, 1.12141148]),
                "viewpoint": [2, 2],
                "community": array(["subcommunity_1", "subcommunity_2"]),
            }
        ),
        "metacommunity_dataframe": DataFrame(
            {
                "alpha": [2.5],
                "rho": [1.6502801833927663],
                "beta": [0.5942352817544037],
                "gamma": [1.5060240963855422],
                "normalized_alpha": [1.2903225806451613],
                "normalized_rho": [0.8485572790897555],
                "normalized_beta": [1.1744247216675028],
                "viewpoint": [2],
                "community": ["metacommunity"],
            },
            index=range(1),
        ),
    },
]


class TestSimilaritySensitiveMetacommunity:
    @fixture(params=SIMILARITY_SENSITIVE_METACOMMUNITY_TEST_CASES, scope="class")
    def test_case(self, request):
        test_case_ = deepcopy(request.param)
        with SharedArrayManager() as shared_array_manager:
            if isinstance(request.param["abundance"], dict):
                counts = shared_array_manager.from_array(
                    request.param["abundance"]["counts"]
                )
                abundance = SharedAbundance(
                    counts=counts, shared_array_manager=shared_array_manager
                )
                test_case_["abundance"] = abundance

            if isinstance(request.param["similarity"], dict):
                kwargs = {
                    key: request.param["similarity"][key]
                    for key in [
                        "similarity_function",
                        "species_ordering",
                        "num_processors",
                    ]
                }
                features = shared_array_manager.from_array(
                    request.param["similarity"]["features"]
                )
                kwargs["shared_array_manager"] = shared_array_manager
                kwargs["features"] = features
                similarity = SimilarityFromFunction(**kwargs)
                test_case_["similarity"] = similarity

            yield test_case_

    def test_metacommunity_similarity(self, test_case):
        metacommunity = SimilaritySensitiveMetacommunity(
            abundance=test_case["abundance"],
            subcommunity_ordering=test_case["subcommunity_ordering"],
            similarity=test_case["similarity"],
        )
        metacommunity_similarity = metacommunity.metacommunity_similarity()
        assert (
            metacommunity_similarity.shape
            == test_case["metacommunity_similarity"].shape
        )
        assert allclose(metacommunity_similarity, test_case["metacommunity_similarity"])

    def test_subcommunity_similarity(self, test_case):
        metacommunity = SimilaritySensitiveMetacommunity(
            abundance=test_case["abundance"],
            subcommunity_ordering=test_case["subcommunity_ordering"],
            similarity=test_case["similarity"],
        )
        subcommunity_similarity = metacommunity.subcommunity_similarity()

        assert (
            subcommunity_similarity.shape == test_case["subcommunity_similarity"].shape
        )
        assert allclose(subcommunity_similarity, test_case["subcommunity_similarity"])

    def test_normalized_subcommunity_similarity(self, test_case):
        metacommunity = SimilaritySensitiveMetacommunity(
            abundance=test_case["abundance"],
            subcommunity_ordering=test_case["subcommunity_ordering"],
            similarity=test_case["similarity"],
        )
        normalized_subcommunity_similarity = (
            metacommunity.normalized_subcommunity_similarity()
        )

        assert (
            normalized_subcommunity_similarity.shape
            == test_case["normalized_subcommunity_similarity"].shape
        )
        assert allclose(
            normalized_subcommunity_similarity,
            test_case["normalized_subcommunity_similarity"],
        )

    def test_subcommunity_alpha(self, test_case):
        metacommunity = SimilaritySensitiveMetacommunity(
            abundance=test_case["abundance"],
            subcommunity_ordering=test_case["subcommunity_ordering"],
            similarity=test_case["similarity"],
        )
        subcommunity_alpha = metacommunity.subcommunity_diversity(
            test_case["viewpoint"], measure="alpha"
        )

        assert subcommunity_alpha.shape == test_case["subcommunity_alpha"].shape
        assert allclose(subcommunity_alpha, test_case["subcommunity_alpha"])

    def test_subcommunity_rho(self, test_case):
        metacommunity = SimilaritySensitiveMetacommunity(
            abundance=test_case["abundance"],
            subcommunity_ordering=test_case["subcommunity_ordering"],
            similarity=test_case["similarity"],
        )
        subcommunity_rho = metacommunity.subcommunity_diversity(
            test_case["viewpoint"], measure="rho"
        )

        assert subcommunity_rho.shape == test_case["subcommunity_rho"].shape
        assert allclose(subcommunity_rho, test_case["subcommunity_rho"])

    def test_subcommunity_beta(self, test_case):
        metacommunity = SimilaritySensitiveMetacommunity(
            abundance=test_case["abundance"],
            subcommunity_ordering=test_case["subcommunity_ordering"],
            similarity=test_case["similarity"],
        )
        subcommunity_beta = metacommunity.subcommunity_diversity(
            test_case["viewpoint"], measure="beta"
        )

        assert subcommunity_beta.shape == test_case["subcommunity_beta"].shape
        assert allclose(subcommunity_beta, test_case["subcommunity_beta"])

    def test_subcommunity_gamma(self, test_case):
        metacommunity = SimilaritySensitiveMetacommunity(
            abundance=test_case["abundance"],
            subcommunity_ordering=test_case["subcommunity_ordering"],
            similarity=test_case["similarity"],
        )
        subcommunity_gamma = metacommunity.subcommunity_diversity(
            test_case["viewpoint"], measure="gamma"
        )

        assert subcommunity_gamma.shape == test_case["subcommunity_gamma"].shape
        assert allclose(subcommunity_gamma, test_case["subcommunity_gamma"])

    def test_normalized_subcommunity_alpha(self, test_case):
        metacommunity = SimilaritySensitiveMetacommunity(
            abundance=test_case["abundance"],
            subcommunity_ordering=test_case["subcommunity_ordering"],
            similarity=test_case["similarity"],
        )
        normalized_subcommunity_alpha = metacommunity.subcommunity_diversity(
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
        metacommunity = SimilaritySensitiveMetacommunity(
            abundance=test_case["abundance"],
            subcommunity_ordering=test_case["subcommunity_ordering"],
            similarity=test_case["similarity"],
        )
        normalized_subcommunity_rho = metacommunity.subcommunity_diversity(
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
        metacommunity = SimilaritySensitiveMetacommunity(
            abundance=test_case["abundance"],
            subcommunity_ordering=test_case["subcommunity_ordering"],
            similarity=test_case["similarity"],
        )
        normalized_subcommunity_beta = metacommunity.subcommunity_diversity(
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
        metacommunity = SimilaritySensitiveMetacommunity(
            abundance=test_case["abundance"],
            subcommunity_ordering=test_case["subcommunity_ordering"],
            similarity=test_case["similarity"],
        )
        metacommunity_alpha = metacommunity.metacommunity_diversity(
            test_case["viewpoint"], "alpha"
        )
        assert isclose(metacommunity_alpha, test_case["metacommunity_alpha"])

    def test_metacommunity_rho(self, test_case):
        metacommunity = SimilaritySensitiveMetacommunity(
            abundance=test_case["abundance"],
            subcommunity_ordering=test_case["subcommunity_ordering"],
            similarity=test_case["similarity"],
        )
        metacommunity_rho = metacommunity.metacommunity_diversity(
            test_case["viewpoint"], "rho"
        )
        assert isclose(metacommunity_rho, test_case["metacommunity_rho"])

    def test_metacommunity_beta(self, test_case):
        metacommunity = SimilaritySensitiveMetacommunity(
            abundance=test_case["abundance"],
            subcommunity_ordering=test_case["subcommunity_ordering"],
            similarity=test_case["similarity"],
        )
        metacommunity_beta = metacommunity.metacommunity_diversity(
            test_case["viewpoint"], "beta"
        )
        assert isclose(metacommunity_beta, test_case["metacommunity_beta"])

    def test_metacommunity_gamma(self, test_case):
        metacommunity = SimilaritySensitiveMetacommunity(
            abundance=test_case["abundance"],
            subcommunity_ordering=test_case["subcommunity_ordering"],
            similarity=test_case["similarity"],
        )
        metacommunity_gamma = metacommunity.metacommunity_diversity(
            test_case["viewpoint"], "gamma"
        )
        assert isclose(metacommunity_gamma, test_case["metacommunity_gamma"])

    def test_metacommunity_normalized_alpha(self, test_case):
        metacommunity = SimilaritySensitiveMetacommunity(
            abundance=test_case["abundance"],
            subcommunity_ordering=test_case["subcommunity_ordering"],
            similarity=test_case["similarity"],
        )
        metacommunity_normalized_alpha = metacommunity.metacommunity_diversity(
            test_case["viewpoint"], "normalized_alpha"
        )
        assert isclose(
            metacommunity_normalized_alpha, test_case["metacommunity_normalized_alpha"]
        )

    def test_metacommunity_normalized_rho(self, test_case):
        metacommunity = SimilaritySensitiveMetacommunity(
            abundance=test_case["abundance"],
            subcommunity_ordering=test_case["subcommunity_ordering"],
            similarity=test_case["similarity"],
        )
        metacommunity_normalized_rho = metacommunity.metacommunity_diversity(
            test_case["viewpoint"], "normalized_rho"
        )
        assert isclose(
            metacommunity_normalized_rho, test_case["metacommunity_normalized_rho"]
        )

    def test_metacommunity_normalized_beta(self, test_case):
        metacommunity = SimilaritySensitiveMetacommunity(
            abundance=test_case["abundance"],
            subcommunity_ordering=test_case["subcommunity_ordering"],
            similarity=test_case["similarity"],
        )
        metacommunity_normalized_beta = metacommunity.metacommunity_diversity(
            test_case["viewpoint"], "normalized_beta"
        )
        assert isclose(
            metacommunity_normalized_beta, test_case["metacommunity_normalized_beta"]
        )

    def test_subcommunities_to_dataframe(self, test_case):
        metacommunity = SimilaritySensitiveMetacommunity(
            abundance=test_case["abundance"],
            subcommunity_ordering=test_case["subcommunity_ordering"],
            similarity=test_case["similarity"],
        )
        subcommunity_dataframe = metacommunity.subcommunities_to_dataframe(
            viewpoint=test_case["viewpoint"]
        )
        expected_subcommunity_dataframe = test_case["subcommunity_dataframe"][
            subcommunity_dataframe.columns
        ]
        assert_frame_equal(subcommunity_dataframe, expected_subcommunity_dataframe)

    def test_metacommunities_to_dataframe(self, test_case):
        metacommunity = SimilaritySensitiveMetacommunity(
            abundance=test_case["abundance"],
            subcommunity_ordering=test_case["subcommunity_ordering"],
            similarity=test_case["similarity"],
        )
        metacommunity_dataframe = metacommunity.metacommunity_to_dataframe(
            viewpoint=test_case["viewpoint"]
        )
        expected_metacommunity_dataframe = test_case["metacommunity_dataframe"][
            metacommunity_dataframe.columns
        ]
        assert_frame_equal(metacommunity_dataframe, expected_metacommunity_dataframe)


class TestSharedSimilaritySensitiveMetacommunity:
    @fixture(params=SIMILARITY_SENSITIVE_METACOMMUNITY_TEST_CASES, scope="class")
    def test_case(self, request):
        test_case_ = deepcopy(request.param)
        with SharedArrayManager() as shared_array_manager:
            if isinstance(request.param["abundance"], dict):
                counts = shared_array_manager.from_array(
                    request.param["abundance"]["counts"]
                )
                LOGGER.debug("counts: %s", counts, counts.data)
                abundance = SharedAbundance(
                    counts=counts, shared_array_manager=shared_array_manager
                )
                test_case_["abundance"] = abundance

            if isinstance(test_case_["similarity"], dict):
                kwargs = {
                    key: request.param["similarity"][key]
                    for key in [
                        "similarity_function",
                        "species_ordering",
                        "num_processors",
                    ]
                }
                features = shared_array_manager.from_array(
                    request.param["similarity"]["features"]
                )
                kwargs["shared_array_manager"] = shared_array_manager
                kwargs["features"] = features
                similarity = SimilarityFromFunction(**kwargs)
                test_case_["similarity"] = similarity
            test_case_["shared_array_manager"] = shared_array_manager

            yield test_case_

    def test_metacommunity_similarity(self, test_case):
        metacommunity = SharedSimilaritySensitiveMetacommunity(
            abundance=test_case["abundance"],
            subcommunity_ordering=test_case["subcommunity_ordering"],
            similarity=test_case["similarity"],
            shared_array_manager=test_case["shared_array_manager"],
        )
        metacommunity_similarity = metacommunity.metacommunity_similarity()
        assert (
            metacommunity_similarity.shape
            == test_case["metacommunity_similarity"].shape
        )
        assert allclose(metacommunity_similarity, test_case["metacommunity_similarity"])

    def test_subcommunity_similarity(self, test_case):
        metacommunity = SharedSimilaritySensitiveMetacommunity(
            abundance=test_case["abundance"],
            subcommunity_ordering=test_case["subcommunity_ordering"],
            similarity=test_case["similarity"],
            shared_array_manager=test_case["shared_array_manager"],
        )
        subcommunity_similarity = metacommunity.subcommunity_similarity()

        assert (
            subcommunity_similarity.shape == test_case["subcommunity_similarity"].shape
        )
        assert allclose(subcommunity_similarity, test_case["subcommunity_similarity"])

    def test_normalized_subcommunity_similarity(self, test_case):
        metacommunity = SharedSimilaritySensitiveMetacommunity(
            abundance=test_case["abundance"],
            subcommunity_ordering=test_case["subcommunity_ordering"],
            similarity=test_case["similarity"],
            shared_array_manager=test_case["shared_array_manager"],
        )
        normalized_subcommunity_similarity = (
            metacommunity.normalized_subcommunity_similarity()
        )

        assert (
            normalized_subcommunity_similarity.shape
            == test_case["normalized_subcommunity_similarity"].shape
        )
        assert allclose(
            normalized_subcommunity_similarity,
            test_case["normalized_subcommunity_similarity"],
        )

    def test_subcommunity_alpha(self, test_case):
        metacommunity = SharedSimilaritySensitiveMetacommunity(
            abundance=test_case["abundance"],
            subcommunity_ordering=test_case["subcommunity_ordering"],
            similarity=test_case["similarity"],
            shared_array_manager=test_case["shared_array_manager"],
        )
        subcommunity_alpha = metacommunity.subcommunity_diversity(
            test_case["viewpoint"], measure="alpha"
        )

        assert subcommunity_alpha.shape == test_case["subcommunity_alpha"].shape
        assert allclose(subcommunity_alpha, test_case["subcommunity_alpha"])

    def test_subcommunity_rho(self, test_case):
        metacommunity = SharedSimilaritySensitiveMetacommunity(
            abundance=test_case["abundance"],
            subcommunity_ordering=test_case["subcommunity_ordering"],
            similarity=test_case["similarity"],
            shared_array_manager=test_case["shared_array_manager"],
        )
        subcommunity_rho = metacommunity.subcommunity_diversity(
            test_case["viewpoint"], measure="rho"
        )

        assert subcommunity_rho.shape == test_case["subcommunity_rho"].shape
        assert allclose(subcommunity_rho, test_case["subcommunity_rho"])

    def test_subcommunity_beta(self, test_case):
        metacommunity = SharedSimilaritySensitiveMetacommunity(
            abundance=test_case["abundance"],
            subcommunity_ordering=test_case["subcommunity_ordering"],
            similarity=test_case["similarity"],
            shared_array_manager=test_case["shared_array_manager"],
        )
        subcommunity_beta = metacommunity.subcommunity_diversity(
            test_case["viewpoint"], measure="beta"
        )

        assert subcommunity_beta.shape == test_case["subcommunity_beta"].shape
        assert allclose(subcommunity_beta, test_case["subcommunity_beta"])

    def test_subcommunity_gamma(self, test_case):
        metacommunity = SharedSimilaritySensitiveMetacommunity(
            abundance=test_case["abundance"],
            subcommunity_ordering=test_case["subcommunity_ordering"],
            similarity=test_case["similarity"],
            shared_array_manager=test_case["shared_array_manager"],
        )
        subcommunity_gamma = metacommunity.subcommunity_diversity(
            test_case["viewpoint"], measure="gamma"
        )

        assert subcommunity_gamma.shape == test_case["subcommunity_gamma"].shape
        assert allclose(subcommunity_gamma, test_case["subcommunity_gamma"])

    def test_normalized_subcommunity_alpha(self, test_case):
        metacommunity = SharedSimilaritySensitiveMetacommunity(
            abundance=test_case["abundance"],
            subcommunity_ordering=test_case["subcommunity_ordering"],
            similarity=test_case["similarity"],
            shared_array_manager=test_case["shared_array_manager"],
        )
        normalized_subcommunity_alpha = metacommunity.subcommunity_diversity(
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
        metacommunity = SharedSimilaritySensitiveMetacommunity(
            abundance=test_case["abundance"],
            subcommunity_ordering=test_case["subcommunity_ordering"],
            similarity=test_case["similarity"],
            shared_array_manager=test_case["shared_array_manager"],
        )
        normalized_subcommunity_rho = metacommunity.subcommunity_diversity(
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
        metacommunity = SharedSimilaritySensitiveMetacommunity(
            abundance=test_case["abundance"],
            subcommunity_ordering=test_case["subcommunity_ordering"],
            similarity=test_case["similarity"],
            shared_array_manager=test_case["shared_array_manager"],
        )
        normalized_subcommunity_beta = metacommunity.subcommunity_diversity(
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
        metacommunity = SharedSimilaritySensitiveMetacommunity(
            abundance=test_case["abundance"],
            subcommunity_ordering=test_case["subcommunity_ordering"],
            similarity=test_case["similarity"],
            shared_array_manager=test_case["shared_array_manager"],
        )
        metacommunity_alpha = metacommunity.metacommunity_diversity(
            test_case["viewpoint"], "alpha"
        )
        assert isclose(metacommunity_alpha, test_case["metacommunity_alpha"])

    def test_metacommunity_rho(self, test_case):
        metacommunity = SharedSimilaritySensitiveMetacommunity(
            abundance=test_case["abundance"],
            subcommunity_ordering=test_case["subcommunity_ordering"],
            similarity=test_case["similarity"],
            shared_array_manager=test_case["shared_array_manager"],
        )
        metacommunity_rho = metacommunity.metacommunity_diversity(
            test_case["viewpoint"], "rho"
        )
        assert isclose(metacommunity_rho, test_case["metacommunity_rho"])

    def test_metacommunity_beta(self, test_case):
        metacommunity = SharedSimilaritySensitiveMetacommunity(
            abundance=test_case["abundance"],
            subcommunity_ordering=test_case["subcommunity_ordering"],
            similarity=test_case["similarity"],
            shared_array_manager=test_case["shared_array_manager"],
        )
        metacommunity_beta = metacommunity.metacommunity_diversity(
            test_case["viewpoint"], "beta"
        )
        assert isclose(metacommunity_beta, test_case["metacommunity_beta"])

    def test_metacommunity_gamma(self, test_case):
        metacommunity = SharedSimilaritySensitiveMetacommunity(
            abundance=test_case["abundance"],
            subcommunity_ordering=test_case["subcommunity_ordering"],
            similarity=test_case["similarity"],
            shared_array_manager=test_case["shared_array_manager"],
        )
        metacommunity_gamma = metacommunity.metacommunity_diversity(
            test_case["viewpoint"], "gamma"
        )
        assert isclose(metacommunity_gamma, test_case["metacommunity_gamma"])

    def test_metacommunity_normalized_alpha(self, test_case):
        metacommunity = SharedSimilaritySensitiveMetacommunity(
            abundance=test_case["abundance"],
            subcommunity_ordering=test_case["subcommunity_ordering"],
            similarity=test_case["similarity"],
            shared_array_manager=test_case["shared_array_manager"],
        )
        metacommunity_normalized_alpha = metacommunity.metacommunity_diversity(
            test_case["viewpoint"], "normalized_alpha"
        )
        assert isclose(
            metacommunity_normalized_alpha, test_case["metacommunity_normalized_alpha"]
        )

    def test_metacommunity_normalized_rho(self, test_case):
        metacommunity = SharedSimilaritySensitiveMetacommunity(
            abundance=test_case["abundance"],
            subcommunity_ordering=test_case["subcommunity_ordering"],
            similarity=test_case["similarity"],
            shared_array_manager=test_case["shared_array_manager"],
        )
        metacommunity_normalized_rho = metacommunity.metacommunity_diversity(
            test_case["viewpoint"], "normalized_rho"
        )
        assert isclose(
            metacommunity_normalized_rho, test_case["metacommunity_normalized_rho"]
        )

    def test_metacommunity_normalized_beta(self, test_case):
        metacommunity = SharedSimilaritySensitiveMetacommunity(
            abundance=test_case["abundance"],
            subcommunity_ordering=test_case["subcommunity_ordering"],
            similarity=test_case["similarity"],
            shared_array_manager=test_case["shared_array_manager"],
        )
        metacommunity_normalized_beta = metacommunity.metacommunity_diversity(
            test_case["viewpoint"], "normalized_beta"
        )
        assert isclose(
            metacommunity_normalized_beta, test_case["metacommunity_normalized_beta"]
        )

    def test_subcommunities_to_dataframe(self, test_case):
        metacommunity = SharedSimilaritySensitiveMetacommunity(
            abundance=test_case["abundance"],
            subcommunity_ordering=test_case["subcommunity_ordering"],
            similarity=test_case["similarity"],
            shared_array_manager=test_case["shared_array_manager"],
        )
        subcommunity_dataframe = metacommunity.subcommunities_to_dataframe(
            viewpoint=test_case["viewpoint"]
        )
        expected_subcommunity_dataframe = test_case["subcommunity_dataframe"][
            subcommunity_dataframe.columns
        ]
        assert_frame_equal(subcommunity_dataframe, expected_subcommunity_dataframe)

    def test_metacommunities_to_dataframe(self, test_case):
        metacommunity = SharedSimilaritySensitiveMetacommunity(
            abundance=test_case["abundance"],
            subcommunity_ordering=test_case["subcommunity_ordering"],
            similarity=test_case["similarity"],
            shared_array_manager=test_case["shared_array_manager"],
        )
        metacommunity_dataframe = metacommunity.metacommunity_to_dataframe(
            viewpoint=test_case["viewpoint"]
        )
        expected_metacommunity_dataframe = test_case["metacommunity_dataframe"][
            metacommunity_dataframe.columns
        ]
        assert_frame_equal(metacommunity_dataframe, expected_metacommunity_dataframe)
