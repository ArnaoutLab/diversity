"""Tests for diversity.shared."""

from multiprocessing.shared_memory import SharedMemory

from numpy import allclose, array, array_equal, dtype, ndarray, vectorize
from pytest import fixture, mark

from diversity.log import LOGGER
from diversity.shared import (
    extract_data_if_shared,
    LoadSharedArray,
    SharedArrayManager,
    SharedArraySpec,
    SharedArrayView,
)

EXTRACT_DATA_IF_SHARED_TEST_CASES = [
    {
        "description": "list input",
        "args": ([1, 2, 3],),
        "expect_data_attribute": False,
    },
    {
        "description": "array input",
        "args": (array([[1, 2], [3, 4]]),),
        "expect_data_attribute": False,
    },
    {
        "description": "SharedArrayView input",
        "args": (
            SharedArrayView(
                spec=SharedArraySpec(
                    name="fake_name",
                    shape=(2, 3),
                    dtype=dtype("f8"),
                ),
                memory_view=array([[1.1, 1.2, 1.3], [2.1, 2.2, 2.3]]).data,
            ),
        ),
        "expect_data_attribute": True,
    },
    {
        "description": "multiple non-shared inputs",
        "args": (
            [1, 2, 3],
            array([[1, 2], [3, 4]]),
        ),
        "expect_data_attribute": (False, False),
    },
    {
        "description": "multiple shared inputs",
        "args": (
            SharedArrayView(
                spec=SharedArraySpec(
                    name="fake_name1",
                    shape=(2, 3),
                    dtype=dtype("f8"),
                ),
                memory_view=array([[1.1, 1.2, 1.3], [2.1, 2.2, 2.3]]).data,
            ),
            SharedArrayView(
                spec=SharedArraySpec(
                    name="fake_name2",
                    shape=(3,),
                    dtype=dtype("f8"),
                ),
                memory_view=array([11.1, 12.2, 13.3]).data,
            ),
        ),
        "expect_data_attribute": (True, True),
    },
    {
        "description": "multiple mixed inputs",
        "args": (
            SharedArrayView(
                spec=SharedArraySpec(
                    name="fake_name1",
                    shape=(2, 3),
                    dtype=dtype("f8"),
                ),
                memory_view=array([[1.1, 1.2, 1.3], [2.1, 2.2, 2.3]]).data,
            ),
            [1, 2, 3],
            SharedArrayView(
                spec=SharedArraySpec(
                    name="fake_name2",
                    shape=(3,),
                    dtype=dtype("f8"),
                ),
                memory_view=array([11.1, 12.2, 13.3]).data,
            ),
        ),
        "expect_data_attribute": (True, False, True),
    },
]


class TestExtractDataIfShared:
    @mark.parametrize("test_case", EXTRACT_DATA_IF_SHARED_TEST_CASES)
    def test_extract_data_if_shared(self, test_case):
        extracted = extract_data_if_shared(*test_case["args"])
        if type(test_case["expect_data_attribute"]) == bool:
            if test_case["expect_data_attribute"]:
                assert extracted is test_case["args"][0].data
            else:
                assert extracted is test_case["args"][0]
        else:
            for actual, expect_data_attribute, arg in zip(
                extracted, test_case["expect_data_attribute"], test_case["args"]
            ):
                if expect_data_attribute:
                    assert actual is arg.data
                else:
                    assert actual is arg


LOAD_SHARED_ARRAY_TEST_CASES = [
    {
        "description": "Random array data",
        "data": array(
            [
                [0.2863447, 0.62799787, 0.71223763, 0.67280038],
                [0.15417271, 0.41962604, 0.85460662, 0.93927894],
                [0.40719875, 0.70349337, 0.91718212, 0.14033922],
            ]
        ),
        "mutation_operation": lambda arr: arr + 1,
        "mutated_data": array(
            [
                [1.2863447, 1.62799787, 1.71223763, 1.67280038],
                [1.15417271, 1.41962604, 1.85460662, 1.93927894],
                [1.40719875, 1.70349337, 1.91718212, 1.14033922],
            ]
        ),
    },
    {
        "description": "object array data",
        "data": array([["foo", "bar", "baz"], ["zip", "zap", "zippidi"]], dtype=object),
        "mutation_operation": vectorize(lambda arr: arr + "_mutated"),
        "mutated_data": array(
            [
                ["foo_mutated", "bar_mutated", "baz_mutated"],
                ["zip_mutated", "zap_mutated", "zippidi_mutated"],
            ],
            dtype=object,
        ),
    },
]


class TestLoadSharedArray:
    @fixture(params=LOAD_SHARED_ARRAY_TEST_CASES)
    def test_case(self, request):
        memory_size = request.param["data"].size * request.param["data"].itemsize
        shared_memory = SharedMemory(create=True, size=memory_size)
        name = shared_memory.name

        shape = request.param["data"].shape
        dtype_ = request.param["data"].dtype
        ndarray(shape=shape, dtype=dtype_, buffer=shared_memory.buf)[:] = request.param[
            "data"
        ]
        request.param["spec"] = SharedArraySpec(name=name, shape=shape, dtype=dtype_)
        yield request.param
        shared_memory.close()
        shared_memory.unlink()

    def test_load_shared_array(self, test_case):
        with LoadSharedArray(test_case["spec"]) as shared_array_view:
            assert shared_array_view.data.shape == test_case["data"].shape
            assert array_equal(shared_array_view.data, test_case["data"])
            shared_array_view.data[:] = test_case["mutation_operation"](
                shared_array_view.data
            )
            if shared_array_view.spec.dtype == object:
                assert array_equal(shared_array_view.data, test_case["mutated_data"])
            else:
                assert allclose(shared_array_view.data, test_case["mutated_data"])


SHARED_ARRAY_MANAGER_TEST_CASES = [
    {
        "description": "Random array data",
        "shape": (3, 4),
        "data_type": dtype("f8"),
        "data": array(
            [
                [0.2863447, 0.62799787, 0.71223763, 0.67280038],
                [0.15417271, 0.41962604, 0.85460662, 0.93927894],
                [0.40719875, 0.70349337, 0.91718212, 0.14033922],
            ]
        ),
        "mutation_operation": lambda arr: arr + 1,
        "mutated_data": array(
            [
                [1.2863447, 1.62799787, 1.71223763, 1.67280038],
                [1.15417271, 1.41962604, 1.85460662, 1.93927894],
                [1.40719875, 1.70349337, 1.91718212, 1.14033922],
            ]
        ),
    },
    {
        "description": "object array data",
        "shape": (2, 3),
        "data_type": object,
        "data": array([["foo", "bar", "baz"], ["zip", "zap", "zippidi"]], dtype=object),
        "mutation_operation": vectorize(lambda arr: arr + "_mutated"),
        "mutated_data": array(
            [
                ["foo_mutated", "bar_mutated", "baz_mutated"],
                ["zip_mutated", "zap_mutated", "zippidi_mutated"],
            ],
            dtype=object,
        ),
    },
]


class TestSharedArrayManager:
    @mark.parametrize("test_case", SHARED_ARRAY_MANAGER_TEST_CASES)
    def test_empty(self, test_case):
        with SharedArrayManager() as manager:
            shared_array_view = manager.empty(
                shape=test_case["shape"], data_type=test_case["data_type"]
            )
            assert shared_array_view.data.shape == test_case["data"].shape
            shared_array_view.data[:] = test_case["data"]
            assert array_equal(shared_array_view.data, test_case["data"])
            shared_array_view.data[:] = test_case["mutation_operation"](
                shared_array_view.data
            )
            if shared_array_view.spec.dtype == object:
                assert array_equal(shared_array_view.data, test_case["mutated_data"])
            else:
                assert allclose(shared_array_view.data, test_case["mutated_data"])


SHARED_ARRAY_VIEW_TEST_CASES = [
    {
        "description": "Random array data",
        "spec": SharedArraySpec(name="fake_name", shape=(3, 4), dtype=dtype("f8")),
        "data": array(
            [
                [0.2863447, 0.62799787, 0.71223763, 0.67280038],
                [0.15417271, 0.41962604, 0.85460662, 0.93927894],
                [0.40719875, 0.70349337, 0.91718212, 0.14033922],
            ]
        ),
        "mutation_operation": lambda arr: arr + 1,
        "mutated_data": array(
            [
                [1.2863447, 1.62799787, 1.71223763, 1.67280038],
                [1.15417271, 1.41962604, 1.85460662, 1.93927894],
                [1.40719875, 1.70349337, 1.91718212, 1.14033922],
            ]
        ),
    },
    {
        "description": "object array data",
        "spec": SharedArraySpec(name="fake_name", shape=(2, 3), dtype=object),
        "data": array([["foo", "bar", "baz"], ["zip", "zap", "zippidi"]], dtype=object),
        "mutation_operation": vectorize(lambda arr: arr + "_mutated"),
        "mutated_data": array(
            [
                ["foo_mutated", "bar_mutated", "baz_mutated"],
                ["zip_mutated", "zap_mutated", "zippidi_mutated"],
            ],
            dtype=object,
        ),
    },
    {
        "description": "Empty array data",
        "spec": SharedArraySpec(name="fake_name", shape=(1, 0), dtype=dtype("f8")),
        "data": array([[]]),
        "mutation_operation": lambda arr: arr + 1,
        "mutated_data": array([[]]),
    },
]


class TestSharedArrayView:
    @fixture(params=SHARED_ARRAY_VIEW_TEST_CASES)
    def test_case(self, request):
        request.param["shared_data"] = request.param["data"].copy()
        request.param["memory_view"] = request.param["shared_data"].data
        return request.param

    def test_shared_array_view(self, test_case):
        shared_array_view = SharedArrayView(
            spec=test_case["spec"], memory_view=test_case["memory_view"]
        )

        assert shared_array_view.data.shape == test_case["shared_data"].shape
        assert array_equal(shared_array_view.data, test_case["shared_data"])
        shared_array_view.data[:] = test_case["mutation_operation"](
            shared_array_view.data
        )
        assert array_equal(shared_array_view.data, test_case["shared_data"])
        if shared_array_view.spec.dtype == object:
            assert array_equal(shared_array_view.data, test_case["mutated_data"])
        else:
            assert allclose(shared_array_view.data, test_case["mutated_data"])
