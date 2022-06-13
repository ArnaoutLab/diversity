"""Module for metacommunity and subcommunity diversity measures.

Classes
-------
Metacommunity
    Represents a metacommunity made up of subcommunities and computes
    metacommunity subcommunity diversity measures.

Functions
---------
make_metacommunity
    Builds diversity.metacommunity.Metacommunity object according to
    parameter specification.
"""

from abc import ABC, abstractmethod
from functools import cache

from pandas import DataFrame, Index, unique
from numpy import broadcast_to, divide, dtype, empty, zeros

from diversity.abundance import make_abundance
from diversity.exceptions import InvalidArgumentError
from diversity.log import LOGGER
from diversity.shared import SharedArrayManager
from diversity.similarity import ISimilarity, make_similarity
from diversity.utilities import (
    pivot_table,
    power_mean,
    subset_by_column,
)


def make_metacommunity(
    counts,
    subcommunities=None,
    similarity=None,
    subcommunity_column="subcommunity",
    species_column="species",
    count_column="count",
    shared_array_manager=None,
    abundance_kwargs={},
    similarity_kwargs={},
):
    """Initializes a concrete subclass of IMetacommunity.

    Parameters
    ----------
    counts: pandas.DataFrame
        Table with 3 columns: one column lists subcommunity identifiers,
        one lists species identifiers, and the last lists counts of
        species in corresponding subcommunities. Subcommunity-species
        identifier pairs are assumed to be unique. Column headers are
        specified by the subcommunity_column, species_column, and
        count_column arguments. Its data is passed to
        diversity.abundance.make_abundance along with abundance_kwargs.
    subcommunities: numpy.ndarray
        Names of subcommunities to include. Their union is the
        metacommunity, and data for all other subcommunities is ignored.
    similarity: pandas.DataFrame, str, or Callable
        For similarity-sensitive diversity measures. Passed to
        diversity.similarity.make_similarity along with
        similarity_kwargs and a species_subset argument determined by
        subcommunities.
    subcommunity_column, species_column, count_column: str
        Column headers for subcommunity names, species names and
        corresponding counts in counts table.
    shared_array_manager: diversity.shared.SharedArrayManager
        See diversity.metacommunity.SharedSimilaritySensitiveMetacommunity.
    abundance_kwargs: dict[str, Any]
        Additional keyword arguments for diversity.abundance.make_abundance.
    similarity_kwargs: dict[str, Any]
        Additional keyword arguments for diversity.similarity.make_similarity.

    Returns
    -------
    An instance of a concrete subclass of IMetacommunity.
    """
    LOGGER.debug(
        "make_metacommunity(counts=%s, subcommunities=%s,"
        " similarity=%s, subcommunity_column=%s, species_column=%s,"
        " count_column=%s shared_array_manager=%s, abundance_kwargs=%s,"
        " similarity_kwargs=%s",
        counts,
        subcommunities,
        similarity,
        subcommunity_column,
        species_column,
        count_column,
        shared_array_manager,
        abundance_kwargs,
        similarity_kwargs,
    )
    counts_subset = subset_by_column(
        data_frame=counts, column=subcommunity_column, subset=subcommunities
    )
    if subcommunities is None:
        subcommunities = counts_subset[subcommunity_column].unique()
    species_subset = unique(counts_subset[species_column])

    if similarity is None:
        species_ordering = species_subset
    else:
        similarity = make_similarity(
            similarity=similarity,
            species_subset=species_subset,
            **similarity_kwargs,
        )
        species_ordering = similarity.species_ordering

    if "shared_array_manager" in abundance_kwargs:
        pivotted_counts = abundance_kwargs["shared_array_manager"].empty(
            shape=(len(species_subset), len(subcommunities)), data_type=dtype("f8")
        )
    else:
        pivotted_counts = empty(
            shape=(len(species_subset), len(subcommunities)), dtype=dtype("f8")
        )
    pivot_table(
        data_frame=counts_subset,
        pivot_column=subcommunity_column,
        index_column=species_column,
        value_columns=[count_column],
        pivot_ordering=subcommunities,
        index_ordering=species_ordering,
        out=pivotted_counts,
    )
    abundance = make_abundance(counts=pivotted_counts, **abundance_kwargs)
    kwargs = {
        "abundance": abundance,
        "subcommunities": subcommunities,
        "similarity": similarity,
        "shared_array_manager": shared_array_manager,
    }
    kwargs = {kw: arg for kw, arg in kwargs.items() if arg is not None}
    strategies = {
        (abundance, subcommunities): FrequencySensitiveMetacommunity,
        (abundance, subcommunities, similarity): SimilaritySensitiveMetacommunity,
        (
            abundance,
            subcommunities,
            similarity,
            shared_array_manager,
        ): SharedSimilaritySensitiveMetacommunity,
    }
    metacommunity_class, args = strategies[kwargs]
    metacommunity = metacommunity_class(args)
    return metacommunity


class IMetacommunity(ABC):
    """Interface for metacommunities and calculating their diversity."""

    @abstractmethod
    def __init__(self, abundance, subcommunity_ordering):
        self.abundance = abundance
        self.subcommunity_ordering = subcommunity_ordering
        self.measure_components = None

    @cache
    def subcommunity_diversity(self, viewpoint, measure):
        """Calculates subcommunity diversity measures.

        Parameters
        ----------
        viewpoint: numeric
            Viewpoint parameter for diversity measure.
        measure: str
            Name of the diversity measure.

        Returns
        -------
        A numpy array with a diversity value per subcommunity.

        Notes
        -----
        Valid measure identifiers are: "alpha", "rho", "beta", "gamma",
        "normalized_alpha", "normalized_rho", and "normalized_beta".
        """
        numerator, denominator = self.measure_components[measure]
        if callable(numerator):
            numerator = numerator()
        denominator = denominator()
        if measure == "gamma":
            denominator = broadcast_to(
                denominator,
                self.abundance.normalized_subcommunity_abundance().shape,
            )
        community_ratio = divide(
            numerator, denominator, out=zeros(denominator.shape), where=denominator != 0
        )
        result = power_mean(
            1 - viewpoint,
            self.abundance.normalized_subcommunity_abundance(),
            community_ratio,
        )
        if measure in ["beta", "normalized_beta"]:
            return 1 / result
        return result

    @cache
    def metacommunity_diversity(self, viewpoint, measure):
        """Calculates metcommunity diversity measures."""
        subcommunity_diversity = self.subcommunity_diversity(viewpoint, measure)
        return power_mean(
            1 - viewpoint,
            self.abundance.subcommunity_normalizing_constants(),
            subcommunity_diversity,
        )

    def subcommunities_to_dataframe(self, viewpoint):
        """Table containing all subcommunity diversity values.

        Parameters
        ----------
        viewpoint: numeric
            Non-negative number. Can be interpreted as the degree of
            ignorance towards rare species, where 0 treats rare species
            the same as frequent species, and infinity considers only the
            most frequent species.
        """
        df = DataFrame(
            {
                key: self.subcommunity_diversity(viewpoint, key)
                for key in self.measure_components.keys()
            }
        )
        df.insert(0, "viewpoint", viewpoint)
        df.insert(0, "community", self.subcommunity_ordering)
        return df

    def metacommunity_to_dataframe(self, viewpoint):
        """Table containing all metacommunity diversity values.

        Parameters
        ----------
        viewpoint: numeric
            Non-negative number. Can be interpreted as the degree of
            ignorance towards rare species, where 0 treats rare species
            the same as frequent species, and infinity considers only the
            most frequent species.
        """
        df = DataFrame(
            {
                key: self.metacommunity_diversity(viewpoint, key)
                for key in self.measure_components.keys()
            },
            index=Index(["metacommunity"], name="community"),
        )
        df.insert(0, "viewpoint", viewpoint)
        df.reset_index(inplace=True)
        return df


class ISimilaritySensitiveMetacommunity(IMetacommunity):
    """Interface for calculating similarity-sensitive diversity."""

    def __init__(self, abundance, subcommunity_ordering, similarity):
        """Initializes object.

        Parameters
        ----------
        abundance: diversity.abundance.IAbundance
            Object whose (sub-/meta-)community species abundances are
            used.
        subcommunity_ordering: numpy.ndarray
            Ordered subcommunity identifiers. Ordering must correspond
            to the ordering used by abundance.
        similarity: diversity.similarity.ISimilarity
            Object for calculating abundance-weighted similarities.
        """
        super().__init__(
            abundance=abundance, subcommunity_ordering=subcommunity_ordering
        )
        self.similarity = similarity
        self.measure_components = {
            "alpha": (1, self.subcommunity_similarity),
            "rho": (self.metacommunity_similarity, self.subcommunity_similarity),
            "beta": (self.metacommunity_similarity, self.subcommunity_similarity),
            "gamma": (1, self.metacommunity_similarity),
            "normalized_alpha": (1, self.normalized_subcommunity_similarity),
            "normalized_rho": (
                self.metacommunity_similarity,
                self.normalized_subcommunity_similarity,
            ),
            "normalized_beta": (
                self.metacommunity_similarity,
                self.normalized_subcommunity_similarity,
            ),
        }

    @abstractmethod
    def metacommunity_similarity(self):
        """Sums of similarities weighted by metacommunity abundances."""
        pass

    @abstractmethod
    def subcommunity_similarity(self):
        """Sums of similarities weighted by subcommunity abundances."""
        pass

    @abstractmethod
    def normalized_subcommunity_similarity(self):
        """Sums of similarities weighted by the normalized subcommunity abundances."""
        pass


class FrequencySensitiveMetacommunity(IMetacommunity):
    """Implements IMetacommunity for similarity-insensitive diversity."""

    def __init__(self, abundance, subcommunity_ordering):
        """Initializes object.

        Parameters
        ----------
        abundance: diversity.abundance.IAbundance
            Object whose (sub-/meta-)community species abundances are
            used.
        """
        super().__init__(
            abundance=abundance, subcommunity_ordering=subcommunity_ordering
        )
        self.measure_components = {
            "alpha": (1, self.abundance.subcommunity_abundance),
            "rho": (
                self.abundance.metacommunity_abundance,
                self.abundance.subcommunity_abundance,
            ),
            "beta": (
                self.abundance.metacommunity_abundance,
                self.abundance.subcommunity_abundance,
            ),
            "gamma": (1, self.abundance.metacommunity_abundance),
            "normalized_alpha": (
                1,
                self.abundance.normalized_subcommunity_abundance,
            ),
            "normalized_rho": (
                self.abundance.metacommunity_abundance,
                self.abundance.normalized_subcommunity_abundance,
            ),
            "normalized_beta": (
                self.abundance.metacommunity_abundance,
                self.abundance.normalized_subcommunity_abundance,
            ),
        }


class SimilaritySensitiveMetacommunity(ISimilaritySensitiveMetacommunity):
    """Implements ISimilaritySensitiveMetacommunity for fast but memory heavy calculations."""

    @cache
    def metacommunity_similarity(self):
        return self.similarity.calculate_weighted_similarities(
            self.abundance.metacommunity_abundance()
        )

    @cache
    def subcommunity_similarity(self):
        return self.similarity.calculate_weighted_similarities(
            self.abundance.subcommunity_abundance()
        )

    @cache
    def normalized_subcommunity_similarity(self):
        return self.similarity.calculate_weighted_similarities(
            self.abundance.normalized_subcommunity_abundance()
        )


class SharedSimilaritySensitiveMetacommunity(ISimilaritySensitiveMetacommunity):
    """Implements ISimilaritySensitiveMetacommunity using shared memory.

    Caches only one of weighted subcommunity similarities and normalized
    weighted subcommunity similarities at a time. All weighted similarities
    are stored in shared arrays, which can be passed to other processors
    without copying.
    """

    def __init__(
        self, abundance, subcommunity_ordering, similarity, shared_array_manager
    ):
        """Initializes object.

        Parameters
        ----------
        abundance, subcommunity_ordering, similarity
            See diversity.metacommunity.ISimilaritySensitiveMetacommunity.
        shared_memory_manager: diversity.shared.SharedMemoryManager
            Active manager for obtaining shared arrays.

        Notes
        -----
        - Object will break once shared_array_manager becomes inactive.
        - If a diversity.similarity.SimilarityFromFunction object is
          chosen as argument for simlarity, it must be paired with
          diversity.abundance.SharedAbundance object as argument for
          abundance.
        """
        super().__init__(
            abundance=abundance,
            subcommunity_ordering=subcommunity_ordering,
            similarity=similarity,
        )
        self.__storing_normalized_similarities = None
        self.__shared_array_manager = shared_array_manager
        self.__shared_similarity = self.__shared_array_manager.empty(
            shape=self.abundance.subcommunity_abundance().shape,
            data_type=self.abundance.subcommunity_abundance().dtype,
        )
        self.__metacommunity_similarity = None

    def metacommunity_similarity(self):
        if self.__metacommunity_similarity is None:
            self.__metacommunity_similarity = self.__shared_array_manager.empty(
                shape=self.abundance.metacommunity_abundance().shape,
                data_type=self.abundance.metacommunity_abundance().dtype,
            )
            self.similarity.calculate_weighted_similarities(
                self.abundance.metacommunity_abundance(),
                out=self.__metacommunity_similarity,
            )
        return self.__metacommunity_similarity.data

    @cache
    def subcommunity_similarity(self):
        if self.__storing_normalized_similarities is None:
            self.similarity.calculate_weighted_similarities(
                self.abundance.subcommunity_abundance(), out=self.__shared_similarity
            )
        elif self.__storing_normalized_similarities:
            self.__shared_similarity.data *= (
                self.abundance.subcommunity_normalizing_constants()
            )
        self.__storing_normalized_similarities = False
        return self.__shared_similarity.data

    @cache
    def normalized_subcommunity_similarity(self):
        if self.__storing_normalized_similarities is None:
            self.similarity.calculate_weighted_similarities(
                self.abundance.normalized_subcommunity_abundance(),
                out=self.__shared_similarity,
            )
        elif not self.__storing_normalized_similarities:
            self.__shared_similarity.data /= (
                self.abundance.subcommunity_normalizing_constants()
            )
        self.__storing_normalized_similarities = True
        return self.__shared_similarity.data
