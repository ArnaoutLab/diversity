"""Module for calculating relative sub- and metacomunity abundances.

Classes
-------
Abundance
    Abstract base class for relative species abundances in (meta-/sub-)
    communities.
AbundanceFromArray
    Implements Abundance for fast, but memory-heavy calculations.

Functions
---------
make_abundance
    Chooses and creates instance of concrete Abundance implementation.
"""
from functools import cached_property
from typing import Iterable, Union

from numpy import arange, ndarray
from pandas import DataFrame, RangeIndex
from scipy.sparse import spmatrix, diags, issparse


class Abundance:
    """Calculates metacommuntiy and subcommunity relative abundance
    components from a numpy.ndarray containing species counts
    """

    def __init__(self, counts: ndarray) -> None:
        """
        Parameters
        ----------
        counts:
            2-d array with one column per subcommunity, one row per
            species, containing the count of each species in the
            corresponding subcommunities.
        """
        self.subcommunities_names = self.get_subcommunity_names(counts=counts)
        self.subcommunity_abundance = self.make_subcommunity_abundance(counts=counts)

    def get_subcommunity_names(self, counts: ndarray) -> Iterable:
        """Creates or accesses subcommunity column names then returns
        them

        Parameters
        ----------
        counts
            2-d array with one column per subcommunity, one row per
            species, containing the count of each species in the
            corresponding subcommunities.

        Returns
        -------
        The names of the subcommunities in the order they appear in the
        counts matrix
        """
        return arange(counts.shape[1])

    def make_subcommunity_abundance(self, counts: ndarray) -> ndarray:
        """Calculates the relative abundances in subcommunities.

        Parameters
        ----------
        counts
            2-d array with one column per subcommunity, one row per
            species, containing the count of each species in the
            corresponding subcommunities.

        Returns
        -------
        A numpy.ndarray of shape (n_species, n_subcommunities), where
        rows correspond to unique species, columns correspond to
        subcommunities and each element is the abundance of the species
        in the subcommunity relative to the total metacommunity size.
        """
        return counts / counts.sum()

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


class AbundanceFromDataFrame(Abundance):
    """Calculates metacommuntiy and subcommunity relative abundance
    components from a pandas.DataFrame containing species counts
    """

    def get_subcommunity_names(self, counts: DataFrame) -> RangeIndex:
        return counts.columns

    def make_subcommunity_abundance(self, counts: DataFrame) -> ndarray:
        counts = counts.to_numpy()
        return counts / counts.sum()


class AbundanceFromSparseArray(Abundance):
    """Calculates metacommuntiy and subcommunity relative abundance
    components from a pandas.DataFrame containing species counts
    """

    @cached_property
    def metacommunity_abundance(self) -> ndarray:
        return self.subcommunity_abundance.sum(axis=1)

    @cached_property
    def normalized_subcommunity_abundance(self) -> ndarray:
        return self.subcommunity_abundance @ diags(
            1 / self.subcommunity_normalizing_constants
        )


def make_abundance(counts: Union[DataFrame, spmatrix, ndarray]) -> Abundance:
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
    if isinstance(counts, DataFrame):
        return AbundanceFromDataFrame(counts=counts)
    elif hasattr(counts, 'shape'):
        if issparse(counts):
            return AbundanceFromSparseArray(counts=counts)
        else:
            return Abundance(counts=counts)
    else:
        raise NotImplementedError(
            f"Type {type(counts)} is not supported for argument "
            "'counts'. Valid types include pandas.DataFrame, "
            "numpy.ndarray, or scipy.sparse.spmatrix"
        )
