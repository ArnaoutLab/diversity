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

from numpy import arange, ndarray, concatenate
from pandas import DataFrame, RangeIndex
from scipy.sparse import spmatrix, diags, issparse, hstack, csc_array


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
        self.num_subcommunities = counts.shape[1]

        self.subcommunity_abundance = self.make_subcommunity_abundance(counts=counts)
        self.metacommunity_abundance = self.make_metacommunity_abundance()
        self.normalized_subcommunity_abundance = (
            self.make_normalized_subcommunity_abundance()
        )
        self.unify_abundance_array()

    def unify_abundance_array(self):
        """Creates one matrix containing all the abundance matrices:
        metacommunity, subcommunity, and normalized subcommunity.
        These matrices are still available as views on the unified
        data structure. (Because we are using basic slicing here, only
        one copy of the data will exist after garbage collection.)

        This allows for a major computational improvement in efficiency:
        see components.SimilaritySensitiveComponents.
        """
        self.unified_abundance_array = concatenate(
            (
                self.metacommunity_abundance,
                self.subcommunity_abundance,
                self.normalized_subcommunity_abundance,
            ),
            axis=1,
        )
        self.metacommunity_abundance = self.unified_abundance_array[:, [0]]
        self.subcommunity_abundance = self.unified_abundance_array[
            :, 1 : (1 + self.num_subcommunities)
        ]
        self.normalized_subcommunity_abundance = self.unified_abundance_array[
            :, (1 + self.num_subcommunities) :
        ]

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

    def make_metacommunity_abundance(self) -> ndarray:
        """Calculates the relative abundances in metacommunity.

        Returns
        -------
        A numpy.ndarray of shape (n_species, 1), where rows correspond
        to unique species and each row contains the relative abundance
        of the species in the metacommunity.
        """
        return self.subcommunity_abundance.sum(axis=1, keepdims=True)

    def make_subcommunity_normalizing_constants(self) -> ndarray:
        """Calculates subcommunity normalizing constants.

        Returns
        -------
        A numpy.ndarray of shape (n_subcommunities,), with the fraction
        of each subcommunity's size of the metacommunity.
        """
        return self.subcommunity_abundance.sum(axis=0)

    def make_normalized_subcommunity_abundance(self) -> ndarray:
        """Calculates normalized relative abundances in subcommunities.

        Returns
        -------
        A numpy.ndarray of shape (n_species, n_subcommunities), where
        rows correspond to unique species, columns correspond to
        subcommunities and each element is the abundance of the species
        in the subcommunity relative to the subcommunity size.
        """
        self.subcommunity_normalizing_constants = (
            self.make_subcommunity_normalizing_constants()
        )
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

    def unify_abundance_array(self):
        self.unified_abundance_array = hstack(
            (
                self.metacommunity_abundance,
                self.subcommunity_abundance,
                self.normalized_subcommunity_abundance,
            )
        )
        # Convert to a type that supports slicing:
        self.unified_abundance_array = csc_array(self.unified_abundance_array)
        self.metacommunity_abundance = self.unified_abundance_array[:, [0]]
        self.subcommunity_abundance = self.unified_abundance_array[
            :, 1 : (1 + self.num_subcommunities)
        ]
        self.normalized_subcommunity_abundance = self.unified_abundance_array[
            :, (1 + self.num_subcommunities) :
        ]

    def make_metacommunity_abundance(self) -> ndarray:
        sparse_type = type(self.subcommunity_abundance)
        return sparse_type(self.subcommunity_abundance.sum(axis=1)[:, None])

    def make_normalized_subcommunity_abundance(self) -> ndarray:
        self.subcommunity_normalizing_constants = (
            self.make_subcommunity_normalizing_constants()
        )
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
    elif hasattr(counts, "shape"):
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
