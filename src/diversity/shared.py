"""Helper functions for parallel computations.

Classes
-------
LoadSharedArray
    Manages view of a shared memory block.
SharedArrayManager
    Manages shared arrays.
SharedArraySpec
    Description of how to locate and interpret shared array.
SharedArrayView
    Views a managed memory block as numpy.ndarray.

Functions
---------
extract_data_if_shared
    Returns data of shared array as numpy array.
"""
from contextlib import AbstractContextManager
from dataclasses import dataclass
from math import prod
from multiprocessing.shared_memory import SharedMemory

from numpy import dtype, ndarray

from diversity.log import LOGGER
from diversity.exceptions import LogicError


def extract_data_if_shared(*args):
    """Returns .data attribute of a SharedArrayView, and arg otherwise."""
    if len(args) > 1:
        return tuple(
            arg.data if isinstance(arg, SharedArrayView) else arg for arg in args
        )
    else:
        return args[0].data if isinstance(args[0], SharedArrayView) else args[0]


class LoadSharedArray(AbstractContextManager):
    """Context manager to view shared memory block as numpy.ndarray."""

    def __init__(self, spec):
        """Initializes object from existing shared memory block.

        Parameters
        ----------
        spec: diversity.shared.SharedArraySpec
            Specification of shared array and its memory location.
        """
        LOGGER.debug("LoadSharedArray(%s)", spec)
        self.__spec = spec
        self.__shared_memory = None
        self.__shared_array_view = None

    def __enter__(self):
        LOGGER.debug("LoadSharedArray.__enter__()")
        self.__shared_memory = SharedMemory(name=self.__spec.name)
        self.__shared_array_view = SharedArrayView(
            spec=self.__spec,
            memory_view=self.__shared_memory.buf,
        )
        return self.__shared_array_view

    def __exit__(self, exc_type, exc_value, traceback):
        LOGGER.debug(
            "LoadSharedArray.__exit__(%s, %s, %s)", exc_type, exc_value, traceback
        )
        self.__shared_array_view.data = None
        self.__shared_memory.close()


class SharedArrayManager(AbstractContextManager):
    """Manages numpy.ndarray interpretable shared memory blocks."""

    def __init__(self):
        LOGGER.debug("SharedArrayManager()")
        self.__shared_array_views = None
        self.__shared_memory_blocks = None

    def __enter__(self):
        LOGGER.debug("SharedArrayManager.__enter__()")
        self.__shared_array_views = []
        self.__shared_memory_blocks = []
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        LOGGER.debug(
            "SharedArrayManager.__exit__(%s, %s, %s)", exc_type, exc_value, traceback
        )
        for view in self.__shared_array_views:
            view.data = None
        for shared_memory in self.__shared_memory_blocks:
            shared_memory.close()
            shared_memory.unlink()
        self.__shared_data = None

    def assert_active(self):
        LOGGER.debug("SharedArrayManager.__assert_active()")
        if self.__shared_memory_blocks is None:
            raise LogicError("SharedArrayManager object is inactive.")

    def empty(self, shape, data_type):
        """Allocates shared memory block for a numpy.ndarray.

        Parameters
        ----------
        shape: tuple
            Shape of desired numpy.ndarray.
        data_type: numpy.dtype
            Data type of the numpy.ndarray.

        Returns
        -------
        A diversity.shared.SharedArrayManager.SharedArrayView object
        viewing the allocated memory block.
        """
        LOGGER.debug(
            "SharedArrayManager.empty(shape=%s, data_type=%s)", shape, data_type
        )
        self.assert_active()
        itemsize = dtype(data_type).itemsize
        size = prod([*shape, itemsize])
        shared_memory = SharedMemory(create=True, size=size)
        spec = SharedArraySpec(name=shared_memory.name, shape=shape, dtype=data_type)
        view = SharedArrayView(spec=spec, memory_view=shared_memory.buf)
        self.__shared_array_views.append(view)
        self.__shared_memory_blocks.append(shared_memory)
        return view

    def from_array(self, arr):
        """Stores data from numpy.ndarray in shared memory block."""
        LOGGER.debug("SharedArrayManager.from_array(arr=%s)", arr)
        view = self.empty(shape=arr.shape, data_type=arr.dtype)
        view.data[:] = arr
        return view


@dataclass
class SharedArraySpec:
    """Describes the shape and memory location of a shared array.

    Attributes
    ----------
    name: str
        The name of the shared memory containing the array data.
    shape: tuple[int]
        The shape of the array.
    dtype: numpy.dtype
        The data type of the array.
    """

    name: str
    shape: tuple[int]
    dtype: dtype


class SharedArrayView:
    """Views a managed memory block as numpy.ndarray.

    Attributes
    ----------
    spec: diversity.shared.SharedArraySpec
        Specification of the shared array and its underlying memory
        block.
    data: numpy.ndarray
        The numpy.ndarray whose underlying memory block is managed.
    """

    def __init__(self, spec, memory_view):
        LOGGER.debug("SharedArrayView(%s, %s)", spec, memory_view)
        self.spec = spec
        self.data = ndarray(shape=spec.shape, dtype=spec.dtype, buffer=memory_view)
