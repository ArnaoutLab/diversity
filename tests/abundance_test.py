"""Tests for diversity.abundance."""
from numpy import allclose, array, array_equal, dtype
from pytest import fixture, mark, raises

# from diversity import abundance
from diversity.abundance import Abundance, make_abundance, SharedAbundance
from diversity.exceptions import InvalidArgumentError
from diversity.shared import SharedArrayManager, SharedArraySpec, SharedArrayView


class MockClass:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class MockAbundance(MockClass):
    pass


class MockSharedAbundance(MockClass):
    pass


MAKE_ABUNDANCE_TEST_CASES = [
    {
        "description": "Abundance",
        "counts": array([[2, 4], [3, 0], [0, 1]], dtype=dtype("f8")),
        "shared_array_manager": False,
        "expect_raise": False,
        "expected_return_type": MockAbundance,
    },
    {
        "description": "SharedAbundance",
        "counts": SharedArrayView(
            spec=SharedArraySpec(
                name="fake_name",
                shape=(3, 2),
                dtype=dtype("f8"),
            ),
            memory_view=array([[2, 4], [3, 0], [0, 1]], dtype=dtype("f8")).data,
        ),
        "shared_array_manager": True,
        "expect_raise": False,
        "expected_return_type": MockSharedAbundance,
    },
    {
        "description": "Shared counts, but no manager",
        "counts": SharedArrayView(
            spec=SharedArraySpec(
                name="fake_name",
                shape=(3, 2),
                dtype=dtype("f8"),
            ),
            memory_view=array([[2, 4], [3, 0], [0, 1]], dtype=dtype("f8")).data,
        ),
        "shared_array_manager": False,
        "expect_raise": True,
        "expected_return_type": None,
    },
    {
        "description": "numpy counts, with manager",
        "counts": array([[2, 4], [3, 0], [0, 1]], dtype=dtype("f8")),
        "shared_array_manager": True,
        "expect_raise": True,
        "expected_return_type": None,
    },
]


class TestMakeAbundance:
    @fixture(params=MAKE_ABUNDANCE_TEST_CASES)
    def test_case(self, request, monkeypatch, shared_array_manager):
        with monkeypatch.context() as patched_context:
            for target, new_class in [
                ("diversity.abundance.Abundance", MockAbundance),
                ("diversity.abundance.SharedAbundance", MockSharedAbundance),
            ]:
                patched_context.setattr(target, new_class)
            test_case_ = {
                key: request.param[key]
                for key in [
                    "counts",
                    "expect_raise",
                    "expected_return_type",
                ]
            }
            if request.param["shared_array_manager"]:
                test_case_["shared_array_manager"] = shared_array_manager
            else:
                test_case_["shared_array_manager"] = None

            if request.param["expected_return_type"] == MockAbundance:
                test_case_["expected_init_kwargs"] = {"counts": request.param["counts"]}
                yield test_case_
            else:
                test_case_["expected_init_kwargs"] = {
                    "counts": request.param["counts"],
                    "shared_array_manager": shared_array_manager,
                }
                yield test_case_

    def test_make_abundance(self, test_case):
        if test_case["expect_raise"]:
            with raises(InvalidArgumentError):
                make_abundance(
                    counts=test_case["counts"],
                    shared_array_manager=test_case["shared_array_manager"],
                )
        else:
            abundance = make_abundance(
                counts=test_case["counts"],
                shared_array_manager=test_case["shared_array_manager"],
            )
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


class TestSharedAbundance:
    """Tests metacommunity.Abundance."""

    @mark.parametrize("test_case", ABUNDANCE_TEST_CASES)
    def test_subcommunity_abundance(self, test_case, shared_array_manager):
        """Tests .subcommunity_abundance."""
        counts = shared_array_manager.from_array(test_case["counts"])
        abundance = SharedAbundance(
            counts=counts, shared_array_manager=shared_array_manager
        )
        subcommunity_abundance = abundance.subcommunity_abundance()
        assert allclose(subcommunity_abundance, test_case["subcommunity_abundance"])

    @mark.parametrize("test_case", ABUNDANCE_TEST_CASES)
    def test_shared_subcommunity_abundance(self, test_case, shared_array_manager):
        """Tests .shared_subcommunity_abundance."""
        counts = shared_array_manager.from_array(test_case["counts"])
        abundance = SharedAbundance(
            counts=counts, shared_array_manager=shared_array_manager
        )
        shared_subcommunity_abundance = abundance.shared_subcommunity_abundance()
        assert allclose(
            shared_subcommunity_abundance.data, test_case["subcommunity_abundance"]
        )
        assert shared_subcommunity_abundance is counts

    @mark.parametrize("test_case", ABUNDANCE_TEST_CASES)
    def test_metacommunity_abundance(self, test_case, shared_array_manager):
        """Tests .metacommunity_abundance."""
        counts = shared_array_manager.from_array(test_case["counts"])
        abundance = SharedAbundance(
            counts=counts, shared_array_manager=shared_array_manager
        )
        metacommunity_abundance = abundance.metacommunity_abundance()
        assert allclose(metacommunity_abundance, test_case["metacommunity_abundance"])

    @mark.parametrize("test_case", ABUNDANCE_TEST_CASES)
    def test_shared_metacommunity_abundance(self, test_case, shared_array_manager):
        """Tests .shared_metacommunity_abundance."""
        counts = shared_array_manager.from_array(test_case["counts"])
        abundance = SharedAbundance(
            counts=counts, shared_array_manager=shared_array_manager
        )
        shared_metacommunity_abundance = abundance.shared_metacommunity_abundance()
        assert allclose(
            shared_metacommunity_abundance.data, test_case["metacommunity_abundance"]
        )
        assert isinstance(shared_metacommunity_abundance, SharedArrayView)

    @mark.parametrize("test_case", ABUNDANCE_TEST_CASES)
    def test_subcommunity_normalizing_constants(self, test_case, shared_array_manager):
        """Tests .subcommunity_normalizing_constants."""
        counts = shared_array_manager.from_array(test_case["counts"])
        abundance = SharedAbundance(
            counts=counts, shared_array_manager=shared_array_manager
        )
        subcommunity_normalizing_constants = (
            abundance.subcommunity_normalizing_constants()
        )
        assert allclose(
            subcommunity_normalizing_constants,
            test_case["subcommunity_normalizing_constants"],
        )

    @mark.parametrize("test_case", ABUNDANCE_TEST_CASES)
    def test_normalized_subcommunity_abundance(self, test_case, shared_array_manager):
        """Tests .normalized_subcommunity_abundance."""
        counts = shared_array_manager.from_array(test_case["counts"])
        abundance = SharedAbundance(
            counts=counts, shared_array_manager=shared_array_manager
        )
        normalized_subcommunity_abundance = (
            abundance.normalized_subcommunity_abundance()
        )
        assert allclose(
            normalized_subcommunity_abundance,
            test_case["normalized_subcommunity_abundance"],
        )

    @mark.parametrize("test_case", ABUNDANCE_TEST_CASES)
    def test_shared_normalized_subcommunity_abundance(
        self, test_case, shared_array_manager
    ):
        """Tests .shared_normalized_subcommunity_abundance."""
        counts = shared_array_manager.from_array(test_case["counts"])
        abundance = SharedAbundance(
            counts=counts, shared_array_manager=shared_array_manager
        )
        shared_normalized_subcommunity_abundance = (
            abundance.shared_normalized_subcommunity_abundance()
        )
        assert allclose(
            shared_normalized_subcommunity_abundance.data,
            test_case["normalized_subcommunity_abundance"],
        )
        assert shared_normalized_subcommunity_abundance is counts
