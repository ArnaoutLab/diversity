"""Module for calculating relative sub- and metacomunity abundances.

Classes
-------
Abundance
    Abstract base class for relative species abundances in (meta-/sub-)
    communities.
Abundance
    Implements Abundance for fast, but memory-heavy calculations.

Functions
---------
make_abundance
    Chooses and creates instance of concrete Abundance implementation.
"""
from abc import ABC, abstractmethod
from functools import cached_property
from typing import Iterable, Union

from numpy import arange, ndarray
from pandas import DataFrame, RangeIndex

from diversity.log import LOGGER


class Abundance(ABC):
    """Interface for relative abundances of species in a metacommunity.

    A community consists of a set of species, each of which may appear
    any (non-negative) number of times. A metacommunity consists of one
    or more subcommunities and can be represented by the number of
    appearances of each species in each of the subcommunities that the
    species appears in.
    """

    def __init__(self, counts: Union[ndarray, DataFrame]) -> None:
        """
        Parameters
        ----------
        counts:
            2-d array with one column per subcommunity, one row per species,
            containing the count of each species in the corresponding subcommunities.
        """
        LOGGER.debug("Abundance(counts=%s", counts)
        self.subcommunities_names = self.get_subcommunity_names(counts=counts)
        self.subcommunity_abundance = self.make_subcommunity_abundance(counts=counts)

    @abstractmethod
    def get_subcommunity_names(self, counts: Union[ndarray, DataFrame]) -> Iterable:
        """Creates or accesses subcommunity column names then returns them

        Parameters
        ----------
        counts
            2-d array with one column per subcommunity, one row per species,
            containing the count of each species in the corresponding subcommunities.

        Returns
        -------
        The names of the subcommunities in the order they appear in the counts matrix
        """
        pass

    @abstractmethod
    def make_subcommunity_abundance(self, counts: Union[ndarray, DataFrame]) -> ndarray:
        """Calculates the relative abundances in subcommunities.

        Parameters
        ----------
        counts
            2-d array with one column per subcommunity, one row per species,
            containing the count of each species in the corresponding subcommunities.

        Returns
        -------
        A numpy.ndarray of shape (n_species, n_subcommunities), where
        rows correspond to unique species, columns correspond to
        subcommunities and each element is the abundance of the species
        in the subcommunity relative to the total metacommunity size.
        """
        pass

    @cached_property
    def metacommunity_abundance(self) -> ndarray:
        """Calculates the relative abundances in metacommunity.

        Returns
        -------
        A numpy.ndarray of shape (n_species, 1), where rows correspond
        to unique species and each row contains the relative abundance
        of the species in the metacommunity.
        """
        return self.subcommunity_abundance.sum(axis=1, keepdims=True)

    @cached_property
    def subcommunity_normalizing_constants(self) -> ndarray:
        """Calculates subcommunity normalizing constants.

        Returns
        -------
        A numpy.ndarray of shape (n_subcommunities,), with the fraction
        of each subcommunity's size of the metacommunity.
        """
        return self.subcommunity_abundance.sum(axis=0)

    @cached_property
    def normalized_subcommunity_abundance(self) -> ndarray:
        """Calculates normalized relative abundances in subcommunities.

        Returns
        -------
        A numpy.ndarray of shape (n_species, n_subcommunities), where
        rows correspond to unique species, columns correspond to
        subcommunities and each element is the abundance of the species
        in the subcommunity relative to the subcommunity size.
        """
        return self.subcommunity_abundance / self.subcommunity_normalizing_constants


class AbundanceFromArray(Abundance):
    """Implements Abundance for fast, but memory-heavy calculations.

    Caches relative meta- and subcommunity abundances.
    """

    def __init__(self, counts: ndarray) -> None:
        super().__init__(counts)

    def get_subcommunity_names(self, counts: ndarray) -> ndarray:
        return arange(len(counts))

    def make_subcommunity_abundance(self, counts: ndarray) -> ndarray:
        return counts / counts.sum()


class AbundanceFromDataFrame(Abundance):
    """Implements Abundance for fast, but memory-heavy calculations.

    Caches relative meta- and subcommunity abundances.
    """

    def __init__(self, counts: DataFrame) -> None:
        super().__init__(counts)

    def get_subcommunity_names(self, counts: DataFrame) -> RangeIndex:
        return counts.columns

    def make_subcommunity_abundance(self, counts: DataFrame) -> ndarray:
        counts = counts.to_numpy()
        return counts / counts.sum()


def make_abundance(counts: Union[DataFrame, ndarray]) -> Abundance:
    """Initializes a concrete subclass of Abundance.

    Parameters
    ----------
    counts:
        2-d array with one column per subcommunity, one row per species,
        where the elements are the species counts

    Returns
    -------
    An instance of a concrete subclass of Abundance.
    """
    LOGGER.debug("make_abundance(counts=%s)", counts)
    if isinstance(counts, DataFrame):
        return AbundanceFromDataFrame(counts=counts)
    elif isinstance(counts, ndarray):
        return AbundanceFromArray(counts=counts)
    else:
        raise NotImplementedError(
            f"Type {type(counts)} is not supported for argument 'counts'."
            "Valid types include pandas.DataFrame or numpy.ndarray"
        )
