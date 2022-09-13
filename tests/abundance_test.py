"""Tests for diversity.abundance."""
from numpy import allclose, array, array_equal, dtype
from pytest import fixture, mark, raises

# from diversity import abundance
from diversity.abundance import Abundance, make_abundance
from diversity.exceptions import InvalidArgumentError


class MockClass:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class MockAbundance(MockClass):
    pass


MAKE_ABUNDANCE_TEST_CASES = [
    {
        "description": "Abundance",
        "counts": array([[2, 4], [3, 0], [0, 1]], dtype=dtype("f8")),
        "expect_raise": False,
        "expected_return_type": MockAbundance,
    },
]


class TestMakeAbundance:
    @fixture(params=MAKE_ABUNDANCE_TEST_CASES)
    def test_case(self, request, monkeypatch):
        with monkeypatch.context() as patched_context:
            for target, mock_class in [
                ("diversity.abundance.Abundance", MockAbundance),
            ]:
                patched_context.setattr(target, mock_class)
            test_case_ = {
                key: request.param[key]
                for key in [
                    "counts",
                    "expect_raise",
                    "expected_return_type",
                ]
            }
            if request.param["expected_return_type"] == MockAbundance:
                test_case_["expected_init_kwargs"] = {"counts": request.param["counts"]}
            else:
                test_case_["expected_init_kwargs"] = {
                    "counts": request.param["counts"],
                }
            yield test_case_

    def test_make_abundance(self, test_case):
        if test_case["expect_raise"]:
            with raises(InvalidArgumentError):
                make_abundance(counts=test_case["counts"])
        else:
            abundance = make_abundance(test_case["counts"])
            assert isinstance(abundance, test_case["expected_return_type"])
            for key, arg in test_case["expected_init_kwargs"].items():
                assert abundance.kwargs[key] is arg


ABUNDANCE_TEST_CASES = [
    {
        "description": "default parameters; 2 communities; both contain exclusive species",
        "counts": array([[2, 4], [3, 0], [0, 1]], dtype=dtype("f8")),
        "subcommunity_abundance": array(
            [[2 / 10, 4 / 10], [3 / 10, 0 / 10], [0 / 10, 1 / 10]]
        ),
        "metacommunity_abundance": array([[6 / 10], [3 / 10], [1 / 10]]),
        "subcommunity_normalizing_constants": array([5 / 10, 5 / 10]),
        "normalized_subcommunity_abundance": array(
            [[2 / 5, 4 / 5], [3 / 5, 0 / 5], [0 / 5, 1 / 5]]
        ),
    },
    {
        "description": "default parameters; 2 communities; one contains exclusive species",
        "counts": array([[2, 4], [3, 0], [5, 1]], dtype=dtype("f8")),
        "subcommunity_abundance": array(
            [[2 / 15, 4 / 15], [3 / 15, 0 / 15], [5 / 15, 1 / 15]]
        ),
        "metacommunity_abundance": array(
            [
                [6 / 15],
                [3 / 15],
                [6 / 15],
            ]
        ),
        "subcommunity_normalizing_constants": array([10 / 15, 5 / 15]),
        "normalized_subcommunity_abundance": array(
            [[2 / 10, 4 / 5], [3 / 10, 0 / 5], [5 / 10, 1 / 5]]
        ),
    },
    {
        "description": "default parameters; 2 communities; neither contain exclusive species",
        "counts": array(
            [
                [2, 4],
                [3, 1],
                [1, 5],
            ],
            dtype=dtype("f8"),
        ),
        "subcommunity_abundance": array(
            [
                [2 / 16, 4 / 16],
                [3 / 16, 1 / 16],
                [1 / 16, 5 / 16],
            ]
        ),
        "metacommunity_abundance": array([[6 / 16], [4 / 16], [6 / 16]]),
        "subcommunity_normalizing_constants": array([6 / 16, 10 / 16]),
        "normalized_subcommunity_abundance": array(
            [
                [2 / 6, 4 / 10],
                [3 / 6, 1 / 10],
                [1 / 6, 5 / 10],
            ]
        ),
    },
    {
        "description": "default parameters; 2 mutually exclusive communities",
        "counts": array(
            [
                [2, 0],
                [3, 0],
                [0, 1],
                [0, 4],
            ],
            dtype=dtype("f8"),
        ),
        "subcommunity_abundance": array(
            [
                [2 / 10, 0 / 10],
                [3 / 10, 0 / 10],
                [0 / 10, 1 / 10],
                [0 / 10, 4 / 10],
            ]
        ),
        "metacommunity_abundance": array([[2 / 10], [3 / 10], [1 / 10], [4 / 10]]),
        "subcommunity_normalizing_constants": array([5 / 10, 5 / 10]),
        "normalized_subcommunity_abundance": array(
            [
                [2 / 5, 0 / 5],
                [3 / 5, 0 / 5],
                [0 / 5, 1 / 5],
                [0 / 5, 4 / 5],
            ]
        ),
    },
    {
        "description": "default parameters; 1 community",
        "counts": array(
            [
                [2],
                [5],
                [3],
            ],
            dtype=dtype("f8"),
        ),
        "subcommunity_abundance": array(
            [
                [2 / 10],
                [5 / 10],
                [3 / 10],
            ]
        ),
        "metacommunity_abundance": array(
            [
                [2 / 10],
                [5 / 10],
                [3 / 10],
            ]
        ),
        "subcommunity_normalizing_constants": array([10 / 10]),
        "normalized_subcommunity_abundance": array(
            [
                [2 / 10],
                [5 / 10],
                [3 / 10],
            ]
        ),
    },
]


class TestAbundance:
    """Tests metacommunity.Abundance."""

    @mark.parametrize("test_case", ABUNDANCE_TEST_CASES)
    def test_init(self, test_case):
        """Tests initializer."""
        abundance = Abundance(counts=test_case["counts"])
        array_equal(abundance.counts, test_case["counts"])

    @mark.parametrize("test_case", ABUNDANCE_TEST_CASES)
    def test_subcommunity_abundance(self, test_case):
        """Tests .subcommunity_abundance."""
        abundance = Abundance(counts=test_case["counts"])
        subcommunity_abundance = abundance.subcommunity_abundance()
        assert allclose(subcommunity_abundance, test_case["subcommunity_abundance"])

    @mark.parametrize("test_case", ABUNDANCE_TEST_CASES)
    def test_metacommunity_abundance(self, test_case):
        """Tests .metacommunity_abundance."""
        abundance = Abundance(counts=test_case["counts"])
        metacommunity_abundance = abundance.metacommunity_abundance()
        assert allclose(metacommunity_abundance, test_case["metacommunity_abundance"])

    @mark.parametrize("test_case", ABUNDANCE_TEST_CASES)
    def test_subcommunity_normalizing_constants(self, test_case):
        """Tests .subcommunity_normalizing_constants."""
        abundance = Abundance(counts=test_case["counts"])
        subcommunity_normalizing_constants = (
            abundance.subcommunity_normalizing_constants()
        )
        assert allclose(
            subcommunity_normalizing_constants,
            test_case["subcommunity_normalizing_constants"],
        )

    @mark.parametrize("test_case", ABUNDANCE_TEST_CASES)
    def test_normalized_subcommunity_abundance(self, test_case):
        """Tests .normalized_subcommunity_abundance."""
        abundance = Abundance(counts=test_case["counts"])
        normalized_subcommunity_abundance = (
            abundance.normalized_subcommunity_abundance()
        )
        assert allclose(
            normalized_subcommunity_abundance,
            test_case["normalized_subcommunity_abundance"],
        )
