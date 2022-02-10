"""Module for calculating relative sub- and metacomunity abundances.

Classes
-------
IAbundance
    Abstract base class for relative species abundances in (meta-/sub-)
    communities.
Abundance
    Implements IAbundance for fast, but memory-heavy calculations.
SharedAbundance
    Implements IAbundance using shared memory.

Functions
---------
make_abundance
    Chooses and creates instance of concrete IAbundance implementation.
"""
from abc import ABC, abstractmethod
from functools import cache

from numpy import dtype, empty, ndarray

from diversity.exceptions import InvalidArgumentError
from diversity.log import LOGGER
from diversity.shared import SharedArrayManager, SharedArrayView


def make_abundance(counts, shared_array_manager=None):
    """Initializes a concrete subclass of IAbundance.

    Parameters
    ----------
    counts: numpy.ndarray or diversity.shared.SharedArrayView
        If numpy.ndarray, see diversity.abundance.Abundance.
        If diversity.shared.SharedArrayView, see
        diversity.abundance.SharedAbundance.
    shared_array_manager: diversity.shared.SharedArrayManager
        See diversity.abundance.SharedAbundance. Only relevant if a
        shared array is passed as argument for counts.

    Returns
    -------
    An instance of a concrete subclass of IAbundance.

    Notes
    -----
    Valid parameter combinations are:
    - counts: numpy.ndarray
      shared_array_manager: None
    - counts: diversity.shared.SharedArrayView
      shared_array_manager: diversity.shared.SharedArrayManager
    """
    LOGGER.debug(
        "make_abundance(counts=%s, shared_array_manager=%s)",
        counts,
        shared_array_manager,
    )
    if isinstance(counts, ndarray) and shared_array_manager is None:
        abundance = Abundance(counts=counts)
    elif isinstance(counts, SharedArrayView) and isinstance(
        shared_array_manager, SharedArrayManager
    ):
        abundance = SharedAbundance(
            counts=counts, shared_array_manager=shared_array_manager
        )
    else:
        raise InvalidArgumentError(
            "Invalid argument types for make_abundance; counts: %s,"
            " shared_array_manager: %s",
            type(counts),
            type(shared_array_manager),
        )
    return abundance


class IAbundance(ABC):
    """Interface for relative abundances of species in a metacommunity.

    A community consists of a set of species, each of which may appear
    any (non-negative) number of times. A metacommunity consists of one
    or more subcommunities and can be represented by the number of
    appearances of each species in each of the subcommunities that the
    species appears in.
    """

    @abstractmethod
    def subcommunity_abundance(self):
        """Calculates the relative abundances in subcommunities.

        Returns
        -------
        A numpy.ndarray of shape (n_species, n_subcommunities), where
        rows correspond to unique species, columns correspond to
        subcommunities and each element is the abundance of the species
        in the subcommunity relative to the total metacommunity size.
        """
        pass

    @abstractmethod
    def metacommunity_abundance(self):
        """Calculates the relative abundances in metacommunity.

        Returns
        -------
        A numpy.ndarray of shape (n_species, 1), where rows correspond
        to unique species and each row contains the relative abundance
        of the species in the metacommunity.
        """
        pass

    @abstractmethod
    def subcommunity_normalizing_constants(self):
        """Calculates subcommunity normalizing constants.

        Returns
        -------
        A numpy.ndarray of shape (n_subcommunities,), with the fraction
        of each subcommunity's size of the metacommunity.
        """
        pass

    @abstractmethod
    def normalized_subcommunity_abundance(self):
        """Calculates normalized relative abundances in subcommunities.

        Returns
        -------
        A numpy.ndarray of shape (n_species, n_subcommunities), where
        rows correspond to unique species, columns correspond to
        subcommunities and each element is the abundance of the species
        in the subcommunity relative to the subcommunity size.
        """
        pass


class Abundance(IAbundance):
    """Implements IAbundance for fast, but memory-heavy calculations.

    Caches counts and (normalized) relative meta- and subcommunity
    abundances at the same time.
    """

    def __init__(self, counts):
        """Initializes object.

        Parameters
        ----------
        counts: numpy.ndarray
            A 2-d numpy.ndarray with one column per subcommunity, one
            row per species, containing the count of each species in the
            corresponding subcommunities.

        """
        LOGGER.debug("Abundance(counts=%s", counts)
        self.counts = counts

    @cache
    def subcommunity_abundance(self):
        total_abundance = self.counts.sum()
        relative_abundances = empty(shape=self.counts.shape, dtype=dtype("f8"))
        relative_abundances[:] = self.counts / total_abundance
        return relative_abundances

    @cache
    def metacommunity_abundance(self):
        return self.subcommunity_abundance().sum(axis=1, keepdims=True)

    @cache
    def subcommunity_normalizing_constants(self):
        return self.subcommunity_abundance().sum(axis=0)

    @cache
    def normalized_subcommunity_abundance(self):
        return self.subcommunity_abundance() / self.subcommunity_normalizing_constants()


class SharedAbundance(IAbundance):
    """Implements IAbundance using shared memory.

    Caches only one of relative subcommunity abundances and normalized
    relative subcommunity abundances at a time. All relative abundances
    are stored in shared arrays, which can be passed to other processors
    without copying.
    """

    def __init__(self, counts, shared_array_manager):
        """Initializes object.

        Parameters
        ----------
        counts: diversity.shared.SharedArrayView
            A 2-d shared array with one column per subcommunity, one
            row per species, containing the count of each species in the
            corresponding subcommunities. Array must be an array of
            floats.
        shared_array_manager: diversity.shared.SharedArrayManager
            An active manager for creating shared arrays.

        Notes
        -----
        Object will break once shared_array_manager or counts become
        inactive.
        """
        LOGGER.debug(
            "SharedAbundance(counts=%s, shared_array_manager=%s)",
            counts,
            shared_array_manager,
        )
        self.__shared_data = counts
        self.__shared_data.data /= self.__shared_data.data.sum()
        self.__storing_normalized_abundances = False
        self.__subcommunity_normalizing_constants = self.__shared_data.data.sum(axis=0)
        self.__metacommunity_abundance = shared_array_manager.empty(
            shape=(self.__shared_data.data.shape[0], 1),
            data_type=self.__shared_data.data.dtype,
        )
        self.__shared_data.data.sum(
            axis=1, keepdims=True, out=self.__metacommunity_abundance.data
        )

    def subcommunity_abundance(self):
        """Same as in diversity.metacommunity.IAbundance, except a shared array is returned."""
        if self.__storing_normalized_abundances:
            self.__shared_data.data *= self.__subcommunity_normalizing_constants
        self.__storing_normalized_abundances = False
        return self.__shared_data.data

    def shared_subcommunity_abundance(self):
        """Returns diversity.shared.SharedArrayView of subcommunity abundances."""
        self.subcommunity_abundance()
        return self.__shared_data

    def metacommunity_abundance(self):
        """Same as in diversity.metacommunity.IAbundance, except a shared array is returned."""
        return self.__metacommunity_abundance.data

    def shared_metacommunity_abundance(self):
        """Returns diversity.shared.SharedView of metacommunity abundances."""
        return self.__metacommunity_abundance

    def subcommunity_normalizing_constants(self):
        return self.__subcommunity_normalizing_constants

    def normalized_subcommunity_abundance(self):
        """Same as in diversity.metacommunity.IAbundance, except a shared array is returned."""
        if not self.__storing_normalized_abundances:
            self.__shared_data.data /= self.__subcommunity_normalizing_constants
        self.__storing_normalized_abundances = True
        return self.__shared_data.data

    def shared_normalized_subcommunity_abundance(self):
        """Returns diversity.shared.SharedArrayView of normalized subcommunity abundances."""
        self.normalized_subcommunity_abundance()
        return self.__shared_data
