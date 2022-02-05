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
        self.shared_memory = None
        self.data = None

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
        LOGGER.debug("SharedArraySpec.from_shared_array(%s, %s)", cls, shared_array)
        return cls(
            name=shared_array.name, shape=shared_array.shape, dtype=shared_array.dtype
        )


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


class SharedArrayView(SIharedArray):
    """Implementation of ISharedArray which doesn't own its memory block.

    See diversity.utilities.ISharedArray for descriptions of attributes.
    See documentation for diversity.utilities.SharedArray for usage
    example.
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
