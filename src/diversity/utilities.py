"""Miscellaneous helper module for the metacommunity package.

Classes
-------
SharedArraySpec
    Description of how to locate and interpret shared array.
ISharedArray
    Abstract base class for shared arrays.
SharedArray
    Shared array owning the corresponding memory block.
SharedArrayView
    Shared array, not owning the corresponding memory block.

Functions
---------
get_file_delimiter
    Determines delimiter in datafile from file extension.
partition_range
    Splits range_ into evenly sized consecutive subranges.
power_mean
    Calculates weighted power means.
unique_correspondence
    Describes correspondence between a sequence of items and the set of
    their unique elements.

Exceptions
----------
MetacommunityError
    Base class for all custom metacommunity exceptions.
InvalidArgumentError
    Raised when invalid argument is passed to a function.
ArgumentWarning
    Used for warnings of problematic argument choices.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from warnings import warn

from numpy import (
    array,
    dtype,
    isclose,
    prod,
    amin,
    amax,
    multiply,
    sum as numpy_sum,
    ndarray,
    inf,
    power,
    unique,
)


class MetacommunityError(Exception):
    """Base class for all custom metacommunity exceptions."""

    pass


class InvalidArgumentError(MetacommunityError):
    """Raised when a function receives an invalid argument."""

    pass


class ArgumentWarning(Warning):
    """Used for warnings related to problematic argument choices."""

    pass


def get_file_delimiter(filepath):
    """Determines delimiter of file from filepath ending.

    Parameters
    ----------
    filepath: str
        The filepath whose delimiter is to be determine.

    Returns
    -------
    "," if the file has the csv extension and "\t" otherwise.

    Notes
    -----
    Emits a warning if filepath has neither .csv nor .tsv ending.
    """
    suffix = Path(filepath).suffix
    if suffix == ".csv":
        return ","
    elif suffix == ".tsv":
        return "\t"
    else:
        warn(
            f"File extension for {filepath} not recognized. Assuming"
            " tab-delimited file.",
            category=ArgumentWarning,
        )
        return "\t"


def partition_range(range_, num_chunks):
    """Splits range_ into evenly sized consecutive subranges.

    Parameters
    ----------
    range_: range
        The overall range to split into chunks.
    num_chunks: int
        The number of subranges to split range_ into. Must be positive.

    Returns
    -------
    A list of range objects. The ranges are evenly sized. When range_
    doesn't divide evenly by num_chunks, the ranges near the end of the
    returned list are 1 larger than ranges near the beginning.

    Raises
    ------
    InvalidArgumentError when num_chunks is not positive.
    """
    if num_chunks <= 0:
        raise InvalidArgumentError(
            f"Range can not be split into a non-positive number of chunks."
        )
    small_chunk_size, num_big_chunks = divmod(len(range_), num_chunks)
    num_small_chunks = num_chunks - num_big_chunks
    small_chunks_start = range_.start
    small_chunks = [
        range(
            small_chunks_start + (i * small_chunk_size),
            small_chunks_start + ((i + 1) * small_chunk_size),
        )
        for i in range(num_small_chunks)
    ]
    big_chunks_start = small_chunks_start + (num_small_chunks * small_chunk_size)
    big_chunk_size = small_chunk_size + 1
    big_chunks = [
        range(
            big_chunks_start + (i * big_chunk_size),
            big_chunks_start + ((i + 1) * big_chunk_size),
        )
        for i in range(num_big_chunks)
    ]
    return [*small_chunks, *big_chunks]


def __validate_power_mean_args(weights, items, atol, weight_is_nonzero):
    """Validates arguments for power_mean.

    Parameters
    ----------
    weights, items, atol
        Same as for power_mean.
    weight_is_nonzero
        Boolean array of same shape as weights, indicating those whose
        absolute value meets or exceeds atol.

    Raises
    ------
    InvalidArgumentError when:
    - weights has more than 2 axes.
    - weights and items do not have the same shape
    - any column in weights contains 0s only when weights is 2d and when
      weights is all 0.
    """
    if len(weights.shape) > 2:
        raise InvalidArgumentError(
            f"Invalid weights shape for power_mean: {weights.shape}."
        )
    if weights.shape != items.shape:
        raise InvalidArgumentError(
            f"Shape of weights ({weights.shape}) must be the same as"
            f" shape fo items ({items.shape})."
        )
    all_0_column = None
    if len(weights.shape) == 1:
        all_0_column = [not (weight_is_nonzero).any()]
    elif len(weights.shape) == 2:
        all_0_column = [
            not (weight_is_nonzero[:, col]).any() for col in range(weights.shape[1])
        ]
    if any(all_0_column):
        raise InvalidArgumentError(
            "power_mean expects at least one zero weight. Weights are"
            " considered 0, if absolute value does not meet or exceed"
            f" configurable minimum threshold: {atol:.2e}."
        )


def power_mean(order, weights, items, atol=1e-9):
    """Calculates weighted power means.

    Parameters
    ----------
    order: numeric
        Exponent used for the power mean.
    weights: numpy.ndarray
        The weights corresponding to items. Must be 1-d or 2-d. If 2-d,
        each column must contain at least one wheight whose absolute
        value meets or exceeds atol.
    items: numpy.ndarray
        The elements for which the weighted power mean is computed. Must
        have same shape as weights.
    atol: float
        Threshold below which weights are considered to be 0.

    Returns
    -------
    A numpy.ndarray of the power means of items along axis 0 using order
    as exponent, weighing by weights. The array shape is the same as
    that of weights, or items except with the 0-axis removed. In the
    case of 1-d weights and items, the result has shape (1,). When order
    is close to 0 (absolute value less than atol), less than -100, or
    greater than 100 analytical formulas for the limits at 0, -infinity,
    or infinity are used respectively. An exception is raised if all
    weights or a column of weights (in the 2-d case) are all too close
    to 0.
    """
    print(f"order: {order}")
    print(f"weights: {weights}")
    print(f"items: {items}")
    print(f"atol: {atol}")
    weight_is_nonzero = abs(weights) >= atol
    __validate_power_mean_args(weights, items, atol, weight_is_nonzero)
    if isclose(order, 0):
        return prod(
            power(items, weights, where=weight_is_nonzero),
            axis=0,
            where=weight_is_nonzero,
        )
    elif order < -100:
        return amin(items, axis=0, where=weight_is_nonzero, initial=inf)
    elif order > 100:
        return amax(items, axis=0, where=weight_is_nonzero, initial=-inf)
    items_power = power(items, order, where=weight_is_nonzero)
    items_product = multiply(items_power, weights, where=weight_is_nonzero)
    items_sum = numpy_sum(items_product, axis=0, where=weight_is_nonzero)
    return power(items_sum, 1 / order)


def unique_correspondence(items, ordered_unique_items=None):
    """Returns uniqued items and a mapping from items to uniqued items.

    Parameters
    ----------
    items: numpy.ndarray
        Array of items to unique and/or obtain mapping for.
    ordered_unique_items: Iterable
        Unique items in desired order. If None, ordering will be
        established according to numpy.unique.

    Returns
    -------
    A tuple with coordinates:
    0 - numpy.ndarray
        The ordered unique items.
    1 - numpy.ndarray
        The position in the unique items iterable for each item in
        items.
    """
    if ordered_unique_items is None:
        ordered_unique_items_, item_positions = unique(items, return_inverse=True)
    else:
        ordered_unique_items_ = array(ordered_unique_items)
        item_to_position = {
            str(item): pos for pos, item in enumerate(ordered_unique_items_)
        }
        if len(item_to_position) != len(ordered_unique_items):
            raise InvalidArgumentError(f"Expected ordered_unique_items to be uniqued.")
        item_positions = array([item_to_position[str(item)] for item in items])
    return (ordered_unique_items_, item_positions)


@dataclass
class SharedArraySpec:
    """Describes the shape and memory location of a shared array.

    Attributes
    ----------
    name: str
        The name of the multiprocessing.shared_memory.SharedMemory
        object containing the array data.
    shape: tuple[int]
        The shape of the shared array.
    dtype: numpy.dtype
        The data type of the shared array.
    """

    name: str
    shape: tuple[int]
    dtype: dtype

    @classmethod
    def from_shared_array(cls, shared_array):
        """Creates object from attributes in shared_array.

        Parameters
        ----------
        shared_array: ISharedArray
            The array whose shape and memory location to describe.
        """
        return cls(
            name=shared_array.name, shape=shared_array.shape, dtype=shared_array.dtype
        )


class ISharedArray(ABC):
    """A block in memory that can be viewed as a numpy.ndarray.

    The memory block is either owned, which means that the object is
    responsible for releasing the memory when appropriate, or the memory
    block is accessed without ownership of it and the object is not
    responsible for its deallocation.

    Attributes
    ----------
    shared_memory: multiprocessing.shared_memory.SharedMemory
        The shared memory block in which the array data is stored.
    array: numpy.ndarray
        The numpy.ndarray object whose data resides in the shared memory
        block.
    name: str
        Same as .shared_memory.name attribute for convenience.
    shape: tuple of ints
        Same as .data.shape attribute for convenience.
    dtype: numpy.dtype
        Same as .data.dtype attribute for convenience.
    """

    @abstractmethod
    def __init__(self):
        """Gets memory block and views it as numpy.ndarray."""
        pass

    @property
    def name(self):
        return self.shared_memory.name

    @property
    def shape(self):
        return self.data.shape

    @property
    def dtype(self):
        return self.data.dtype

    @abstractmethod
    def __del__(self):
        """Closes view on memory block, and deallocates if owning."""
        pass


class SharedArray(ISharedArray):
    """Implementation of ISharedArray which owns its memory block.

    Example
    -------
    fill_row.py:
    ```
    from diversity.utilities import SharedArrayView

    def fill_row(row, value, shared_array_spec):
        shared_array = SharedArrayView(shared_array_spec)
        shared_array.data[row] = value
    ```
    in interpreter:
    ```
    >>> from multiprocessing import cpu_count, Pool
    >>> import numpy as np
    >>> from diversity.utilities import SharedArray, SharedArraySpec
    >>> from fill_row import fill_row
    >>>
    >>> shared_array = SharedArray(shape=(3, 4), dtype=np.dtype("f8"))
    >>> args = [
    ...     (i, -i, SharedArraySpec.from_shared_array(shared_array))
    ...     for i in range(shared_array.shape[0])
    ... ]
    >>> with Pool(cpu_count()) as pool:
    ...     pool.starmap(fill_row, args)
    >>>
    >>> shared_array.data
    array([[ 0.,  0.,  0.,  0.],
           [-1., -1., -1., -1.],
           [-2., -2., -2., -2.]])
    >>>
    >>> # If not explicitly deleted a warning is emitted (see Notes).
    >>> del shared_array
    ```

    Notes
    -----
    - If an object of this type has not been destroyed before end of
      program execution, a warning may be thrown that resources are
      leaking. The desctructor of this class takes care of that,
      automatically, but to avoid the warning, explicity delete the
      object.
    - This class is designed for contiguous memory arrays. For more
      complicated arrays with custom classes, this class may break.
    """

    @classmethod
    def from_array(cls, arr):
        """Initializes SharedArray object with array data.

        Parameters
        ----------
        arr: numpy.ndarray
            Data to initialize the object with.

        Returns
        -------
        A SharedArray object whose .data attribute contains the
        same data as arr.
        """
        shared_array = cls(arr.shape, arr.dtype)
        shared_array.data[:] = arr
        return shared_array

    def __init__(self, shape, dtype):
        """Creates empty shared numpy.ndarray wrapper.

        Parameters
        ----------
        shape: tuple of ints
            The shape of the wrapped array.
        dtype: numpy.dtype
            The data type of the wrapped array.
        """
        itemsize = dtype.itemsize
        size = itemsize * prod(shape)
        self.shared_memory = SharedMemory(create=True, size=size)
        self.data = ndarray(shape=shape, dtype=dtype, buffer=self.shared_memory.buf)

    def __del__(self):
        """Releases shared memory block."""
        self.shared_memory.close()
        self.shared_memory.unlink()


class SharedArrayView(SharedArray):
    """Implementation of ISharedArray which doesn't own its memory block.

    See documentation for diversity.utilities.ISharedArray for
    descriptions of attributes. See documentation for
    diversity.utilities.SharedArray for usage example.
    """

    def __init__(self, spec):
        """Initializes object from existing shared memory block.

        Parameters
        ----------
        spec: diversity.utilities.SharedArraySpec
            Specification of shared array and its memory location.
        """
        self.shared_memory = SharedMemory(name=spec.name)
        self.data = ndarray(
            shape=spec.shape, dtype=spec.dtype, buffer=self.shared_memory.buf
        )

    def __del__(self):
        """Closes access to shared memory."""
        self.shared_memory.close()
