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
from typing import Union

from numpy import float64, ndarray, empty
from pandas import DataFrame

from diversity.log import LOGGER


class Abundance(ABC):
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


class AbundanceFromArray(Abundance):
    """Implements Abundance for fast, but memory-heavy calculations.

    Caches counts and (normalized) relative meta- and subcommunity
    abundances at the same time.
    """

    def __init__(self, counts: Union[DataFrame, ndarray]) -> None:
        """
        Parameters
        ----------
        counts:
            2-d array with one column per subcommunity, one row per species,
            containing the count of each species in the corresponding subcommunities.
        """
        LOGGER.debug("Abundance(counts=%s", counts)
        self.counts = DataFrame(counts)

    @cached_property
    def subcommunity_abundance(self) -> ndarray:
        total_abundance = self.counts.sum().sum()
        relative_abundances = empty(shape=self.counts.shape, dtype=float64)
        relative_abundances[:] = self.counts / total_abundance
        return relative_abundances

    @cached_property
    def metacommunity_abundance(self) -> ndarray:
        return self.subcommunity_abundance.sum(axis=1, keepdims=True)

    @cached_property
    def subcommunity_normalizing_constants(self) -> ndarray:
        return self.subcommunity_abundance.sum(axis=0)

    @cached_property
    def normalized_subcommunity_abundance(self) -> ndarray:
        return self.subcommunity_abundance / self.subcommunity_normalizing_constants


def make_abundance(counts: Union[DataFrame, ndarray]) -> Abundance:
    """Initializes a concrete subclass of Abundance.

    Parameters
    ----------
    counts:
        2-d array with one column per subcommunity, one row per species,
        containing the count of each species in the correspon

    Returns
    -------
    An instance of a concrete subclass of Abundance.
    """
    LOGGER.debug("make_abundance(counts=%s)", counts)
    if isinstance(counts, (DataFrame, ndarray)):
        return AbundanceFromArray(counts=counts)
    else:
        raise NotImplementedError(
            f"Type {type(counts)} is not supported for argument 'counts'."
            "Valid types include pandas.DataFrame or numpy.ndarray"
        )
