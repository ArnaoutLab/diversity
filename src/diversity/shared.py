"""Helper functions for parallel computations.

Classes
-------
ISharedArray
    Abstract base class for shared arrays.
SharedArraySpec
    Description of how to locate and interpret shared array.
# SharedArray
#     Shared array owning the corresponding memory block.
SharedArrayView
    Shared array, not owning the corresponding memory block.
SharedArrayManager
    Manages Shared arrays.

Functions
---------
func, func_
    Functions for performing portion of large matrix multiplication.
"""
from abc import ABC, abstractmethod

from contextlib import AbstractContextManager
from dataclasses import dataclass
from math import prod
from multiprocessing.shared_memory import SharedMemory

from numpy import dtype, ndarray
from pandas import read_csv

from diversity.log import LOGGER
from diversity.utilities import get_file_delimiter, LogicError, pivot_table


class SharedAbundance:
    count_property = "count"

    def __init__(
        self,
        counts_filepath,
        manager,
        species_ordering=None,
        subcommunity_ordering=None,
        subcommunity_column="subcommunity",
        species_column="species",
        count_column="count",
    ):
        LOGGER.debug(
            "SharedAbundance(%s, %s, species_ordering=%s,"
            " subcommunity_ordering=%s, subcommunity_column=%s,"
            " species_column=%s, count_column=%s",
            counts_filepath,
            manager,
            species_ordering,
            subcommunity_ordering,
            subcommunity_column,
            species_column,
            count_column,
        )
        self.__shared_data = read_shared_counts(
            filepath=counts_filepath,
            manager=manager,
            subcommunity_column=subcommunity_column,
            species_column=species_column,
            count_column=count_column,
            species_ordering=species_ordering,
        )
        self.__shared_data.data.flags.writable = False
        self.__cached_property = "count"
        self.__total_abundance = self.__shared_data.data.sum()
        self.__subcommunity_normalizing_constants = (
            self.__shared_data.data.sum(axis=0) / self.__total_abundance
        )

    @cached_property
    def __spec(self):
        return SharedArraySpec(
            name=self.__shared_data.name,
            shape=self.__shared_data.data.shape,
            dtype=self.__shared_data.data.dtype,
        )

    @property
    def subcommunity_abundance(self):
        """Calculates the relative abundances in subcommunities.

        Returns
        -------
        A numpy.ndarray of shape (n_species, n_subcommunities), where
        rows correspond to unique species, columns correspond to
        subcommunities and each element is the abundance of the species
        in the subcommunity relative to the total metacommunity size.
        The row ordering is established by the species_to_row attribute.
        """
        self.__shared_data.data.flags.writable = True
        if self.__cached_property == "count":
            self.__shared_data.data /= self.__total_abundance
        elif self.__cached_property == "normalized_subcommunity_abundance":
            self.__shared_data.data *= self.__subcommunity_normalizing_constants
        self.__shared_data.data.flags.writable = False
        self.__cached_property = "subcommunity_abundance"
        return self.__shared_data.data

    @property
    def subcommunity_abundance_spec(self):
        """Memory block of data after storing relative subcommunity abundances.

        Returns
        -------
        A diversity.shared.SharedArraySpec object desribing the memory
        block at which the data is stored.
        """
        self.subcommunity_abundance
        return self.__spec

    @cached_property
    def metacommunity_abundance(self):
        """Calculates the relative abundances in metacommunity.

        Returns
        -------
        A numpy.ndarray of shape (n_species, 1), where rows correspond
        to unique species and each row contains the relative abundance
        of the species in the metacommunity. The row ordering is
        established by the species_to_row attribute.
        """
        return self.subcommunity_abundance.sum(axis=1, keepdims=True)

    @property
    def normalized_subcommunity_abundance(self):
        """Calculates the relative abundances in subcommunities.

        Returns
        -------
        A numpy.ndarray of shape (n_species, n_subcommunities), where
        rows correspond to unique species, columns correspond to
        subcommunities and each element is the abundance of the species
        in the subcommunity relative to the total metacommunity size.
        The row ordering is established by the species_to_row attribute.
        """
        self.__shared_data.data.flags.writable = True
        if self.__cached_property == "count":
            self.__shared_data.data /= (
                self.__total_abundance * self.__subcommunity_normalizing_constants
            )
        elif self.__cached_property == "subcommunity_abundance":
            self.__shared_data.data /= self.__subcommunity_normalizing_constants
        self.__shared_data.data.flags.writable = False
        self.__cached_property = "subcommunity_abundance"
        return self.__shared_data.data

    @property
    def normalized_subcommunity_abundance_spec(self):
        """Memory block of data after storing relative normalized subcommunity abundances.

        Returns
        -------
        A diversity.shared.SharedArraySpec object desribing the memory
        block at which the data is stored.
        """
        self.normalized_subcommunity_abundance
        return self.__spec


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
        LOGGER.debug(
            "SharedArrayView(%s, %s, %s)",
            spec,
            memory_view,
        )
        self.data = ndarray(shape=spec.shape, dtype=spec.dtype, buffer=memory_view)


class LoadSharedArray(AbstractContextManager):
    """Context manager to view shared memory block that is not owned."""

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

    @dataclass
    class _SharedData:
        memory_blocks = []
        array_views = []

    def __init__(self):
        LOGGER.debug("SharedArrayManager()")
        self.__shared_data = None

    def __enter__(self):
        LOGGER.debug("SharedArrayManager.__enter__()")
        self.__shared_data = self._SharedData()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        LOGGER.debug(
            "SharedArrayManager.__exit__(%s, %s, %s)", exc_type, exc_value, traceback
        )
        for view in self.__shared_data.array_views:
            view.data = None
        for shared_memory in self.__shared_data.memory_blocks:
            shared_memory.unlink()
            shared_memory.close()
        self.__shared_data = None

    def __assert_active(self):
        LOGGER.debug("SharedArrayManager.__assert_active()")
        if self.__shared_data is None:
            raise LogicError("Resource allocation using inactive object.")

    def empty(self, shape, dtype):
        """Allocates shared memory block for a numpy.ndarray.

        Parameters
        ----------
        shape: tuple
            Shape of desired numpy.ndarray.
        dtype: numpy.dtype
            Data type of the numpy.ndarray.

        Returns
        -------
        A diversity.shared.SharedArrayManager.SharedArrayView object
        viewing the allocated memory block.
        """
        LOGGER.debug("SharedArrayManager.empty(%s, %s)", shape, dtype)
        self.__assert_active()
        itemsize = dtype.itemsize
        size = prod([*shape, itemsize])
        shared_memory = SharedMemory(create=True, size=size)
        spec = SharedArraySpec(name=shared_memory.name, shape=shape, dtype=dtype)
        view = SharedArrayView(spec=spec, memory_view=shared_memory.buf)
        self.__shared_data.array_views.append(view)
        self.__shared_data.memory_blocks.append(shared_memory)
        return view
