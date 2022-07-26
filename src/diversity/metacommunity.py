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

from pandas import CategoricalDtype, DataFrame, Index, pivot_table
from numpy import atleast_1d, broadcast_to, divide, zeros

from diversity.abundance import make_abundance
from diversity.log import LOGGER
from diversity.similarity import ISimilarity, make_similarity
from diversity.utilities import power_mean


def make_metacommunity(
    counts,
    similarity=None,
    subcommunities=None,
    species=None,
    subcommunity_column="subcommunity",
    species_column="species",
    count_column="count",
    chunk_size=100,
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
        count_column arguments.
    similarity: pandas.DataFrame, str, Callable, numpy.ndarray, or numpy.memmap
        For similarity-sensitive diversity measures. When numpy.ndarray or
        numpy.memmap is used, the ordering of species in the species argument for
        diversity.similarity.make_similarity corresponds to the ordering
        of species in counts.
    subcommunities: numpy.ndarray
        Names of subcommunities to include. Their union is the
        metacommunity, and data for all other subcommunities is ignored.
    species: Sequence
        Names of species to include. Data for all species not listed here is ignored.
    subcommunity_column, species_column, count_column: str
        Column headers for subcommunity names, species names and
        corresponding counts in counts table.

    Returns
    -------
    An instance of a concrete subclass of IMetacommunity.
    """
    LOGGER.debug(
        "make_metacommunity(counts=%s, similarity=%s,"
        " subcommunities=%s, species=%s, subcommunity_column=%s, species_column=%s,"
        " count_column=%s, chunk_size=%s",
        counts,
        similarity,
        subcommunities,
        species,
        subcommunity_column,
        species_column,
        count_column,
        chunk_size,
    )
    unique_species = counts[species_column].unique()
    if similarity is None:
        species_ordering = unique_species
    else:
        similarity = make_similarity(
            similarity=similarity,
            species_order=species,
            chunk_size=chunk_size,
        )
        species_ordering = similarity.species_ordering
    counts[subcommunity_column] = counts[subcommunity_column].astype("category")
    counts[species_column] = counts[species_column].astype(
        CategoricalDtype(categories=species_ordering)
    )
    if subcommunities is not None:
        counts = counts[counts[subcommunity_column].isin(atleast_1d(subcommunities))]
    if species is not None:
        counts = counts[counts[species_column].isin(atleast_1d(species))]

    pivotted_counts = DataFrame(0, index=species_ordering, columns=subcommunities)
    pivotted_counts.loc[species_ordering, subcommunities] = pivot_table(
        counts,
        columns=subcommunity_column,
        index=species_column,
        values=count_column,
        fill_value=0,
        dropna=False,
    )
    abundance = make_abundance(counts=pivotted_counts)
    frequency_sensitive = (
        FrequencySensitiveMetacommunity,
        {
            "abundance": abundance,
            "subcommunity_ordering": subcommunities,
        },
    )
    similarity_sensitive = (
        SimilaritySensitiveMetacommunity,
        {
            "abundance": abundance,
            "subcommunity_ordering": subcommunities,
            "similarity": similarity,
        },
    )
    strategies = {
        False: frequency_sensitive,
        True: similarity_sensitive,
    }
    strategy_choice = strategies[issubclass(type(similarity), ISimilarity)]
    metacommunity_class, kwargs = strategy_choice
    metacommunity = metacommunity_class(**kwargs)
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
