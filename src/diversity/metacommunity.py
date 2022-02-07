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

from functools import cache

from pandas import DataFrame, concat, unique
from numpy import zeros, broadcast_to, divide

from diversity.abundance import Abundance
from diversity.log import LOGGER
from diversity.shared import extract_data_if_shared
from diversity.similarity import make_similarity
from diversity.utilities import (
    pivot_table,
    power_mean,
    subset_by_column,
)


def make_metacommunity(
    counts,
    similarity_method,
    subcommunities=None,
    chunk_size=1,
    features_filepath=None,
    num_processors=None,
    subcommunity_column="subcommunity",
    species_column="species",
    count_column="count",
):
    """Builds a Metacommunity object from specified parameters.

    Depending on the chosen counts argument, the subcommunity and
    normalized subcommunity relative abundances are either
    simultaneously stored in memory for memory-heavy but fast
    computations (when using a pandas.DataFrame), or stored one at a
    time in a shared memory block for large data sets (when providing a
    filepath).

    The similarity_method parameter value determines whether:
        1. An in-memory similarity matrix is used, or
        2. Similarities are read from a file, or
        3. Similarities are computed on the fly without storing the
           entire species similarity matrix in memory or on disk.

    Parameters
    ----------
    counts: pandas.DataFrame, or str
        Table or path to file containing table with 3 columns: one
        column lists subcommunity identifiers, one lists species
        identifiers, and the last lists counts of species in
        corresponding subcommunities. Subcommunity-species identifier
        pairs are assumed to be unique. If using Callable as
        similarity_method, this must be a filepath. Both file and data
        frame are assumed to have column headers as specified by the
        subcommunity_column, species_column, and count_column arguments.
    similarity_method: pandas.DataFrame, str, or Callable
        For in-memory data frame, see diversity.similarity.SimilarityFromMemory,
        for str filepath to similarity matrix, see
        diversity.similarity.SimilarityFromFile, and for callable which
        calculates similarities on the fly,
        see diversity.similarity.SimilarityFromFunction.
    subcommunities: collection of str objects supporting membership test
        Names of subcommunities to include. Their union is the
        metacommunity, and data for all other subcommunities is ignored.
    chunk_size: int
        See diversity.similarity.SimilarityFromFile. Only
        relevant when using in-file similarity matrix (str filepath as
        similarity_method).
    features_filepath: str
        Path to .tsv, or .csv file containing species features. Assumed
        to have a header row, with species identifiers in a column with
        header species_column. All other columns are assumed to be
        features. This parameter is only relevant when similarities are
        computed on the fly (when a Callable is used as the
        similarity_method argument).
    num_processors: int
        See diversity.similarity.SimilarityFromFunction. Only relevant
        when calculating similarities on the fly (when a Callable is
        used as the similarity_method argument).
    subcommunity_column, species_column, count_column: str
        Used to specify non-default column headers in counts table.

    Returns
    -------
    A diversity.metacommunity.Metacommunity object built according to
    parameter specification.
    """
    LOGGER.debug(
        "make_metacommunity(%s, %s, subcommunities=%s, chunk_size=%s,"
        " subcommunity_column=%s, species_column=%s, count_column=%s"
        % (
            counts,
            similarity_matrix,
            subcommunities,
            chunk_size,
            subcommunity_column,
            species_column,
            count_column,
        )
    )

    counts_subset = subset_by_column(counts, subcommunities, subcommunity_column)
    species_subset = unique(counts_subset[species_column])
    similarity = make_similarity(similarity_matrix, species_subset, chunk_size)
    abundance = Abundance(
        pivot_table(
            data_frame=counts_subset,
            pivot_column=subcommunity_column,
            index_column=species_column,
            value_columns=[count_column],
            index_ordering=similarity.species_order,
        )
    )
    return Metacommunity(similarity, abundance)


def make_pairwise_metacommunities(
    counts, similarity_matrix, subcommunity_column, **kwargs
):
    subcommunties_groups = counts.groupby(subcommunity_column)
    pairwise_metacommunities = []
    for i, (_, group_i) in enumerate(subcommunties_groups):
        for j, (_, group_j) in enumerate(subcommunties_groups):
            if j > i:
                counts = concat([group_i, group_j])
                pair_ij = make_metacommunity(
                    counts,
                    similarity_matrix,
                    subcommunity_column=subcommunity_column,
                    **kwargs,
                )
                pairwise_metacommunities.append(pair_ij)
    return pairwise_metacommunities


class Metacommunity:
    """Class for metacommunities and calculating their diversity.

    All diversities computed by objects of this class are
    similarity-sensitive. See https://arxiv.org/abs/1404.6520 for
    precise definitions of the various diversity measures.
    """

    def __init__(self, similarity, abundance):
        """Initializes object.

        Parameters
        ----------
        similarity: diversity.similarity.ISimilarity
            Object for calculating abundance-weighted similarities.
        abundance: diversity.abundance.IAbundance
            Object whose (sub-/meta-)community species abundances are
            used.
        """
        self.__similarity = similarity
        self.__abundance = abundance
        self.__measure_components = {
            "sensitive": {
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
            },
            "insensitive": {
                "alpha": (1, self.__abundance.subcommunity_abundance),
                "rho": (
                    self.__abundance.metacommunity_abundance,
                    self.__abundance.subcommunity_abundance,
                ),
                "beta": (
                    self.__abundance.metacommunity_abundance,
                    self.__abundance.subcommunity_abundance,
                ),
                "gamma": (1, self.__abundance.metacommunity_abundance),
                "normalized_alpha": (
                    1,
                    self.__abundance.normalized_subcommunity_abundance,
                ),
                "normalized_rho": (
                    self.__abundance.metacommunity_abundance,
                    self.__abundance.normalized_subcommunity_abundance,
                ),
                "normalized_beta": (
                    self.__abundance.metacommunity_abundance,
                    self.__abundance.normalized_subcommunity_abundance,
                ),
            },
        }

    @cache
    def metacommunity_similarity(self):
        """Sums of similarities weighted by metacommunity abundances."""
        return self.__similarity.calculate_weighted_similarities(
            self.__abundance.metacommunity_abundance
        )

    @cache
    def subcommunity_similarity(self):
        """Sums of similarities weighted by subcommunity abundances."""
        return self.__similarity.calculate_weighted_similarities(
            self.__abundance.subcommunity_abundance
        )

    @cache
    def normalized_subcommunity_similarity(self):
        """Sums of similarities weighted by the normalized subcommunity abundances."""
        return self.__similarity.calculate_weighted_similarities(
            self.__abundance.normalized_subcommunity_abundance
        )

    @cache
    def subcommunity_measure(self, viewpoint, measure, similarity="sensitive"):
        """Calculates subcommunity diversity measures."""
        numerator, denominator = self.__measure_components[similarity][measure]
        if callable(numerator):
            numerator = numerator()
        denominator = denominator()
        numerator, denominator = map(extract_data_if_shared, (numerator, denominator))
        if measure == "gamma":
            denominator = broadcast_to(
                denominator,
                extract_data_if_shared(self.__abundance.subcommunity_abundance).shape,
            )
        community_ratio = divide(
            numerator, denominator, out=zeros(denominator.shape), where=denominator != 0
        )
        result = power_mean(
            1 - viewpoint,
            extract_data_if_shared(self.__abundance.normalized_subcommunity_abundance),
            community_ratio,
        )
        if measure in ["beta", "normalized_beta"]:
            return 1 / result
        return result

    def metacommunity_measure(self, viewpoint, measure, similarity="sensitive"):
        """Calculates metcommunity diversity measures."""
        subcommunity_measure = self.subcommunity_measure(viewpoint, measure, similarity)
        return power_mean(
            1 - viewpoint,
            extract_data_if_shared(self.__abundance.subcommunity_normalizing_constants),
            subcommunity_measure,
        )

    def subcommunities_to_dataframe(self, viewpoint, similarity="sensitive"):
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
                key: self.subcommunity_measure(viewpoint, key, similarity)
                for key in self.__measure_components["sensitve"].keys()
            }
        )
        df.insert(0, "viewpoint", viewpoint)
        df.insert(0, "community", self.__abundance.subcommunity_order)
        return df

    def metacommunity_to_dataframe(self, viewpoint, similarity="sensitive"):
        """Table containing all metacommunity diversity values.

        Parameters
        ----------
        viewpoint: numeric
            Non-negative number. Can be interpreted as the degree of
            ignorance towards rare species, where 0 treats rare species
            the same as frequent species, and infinity considers only the
            most frequent species."""
        df = DataFrame(
            {
                key: self.metacommunity_measure(viewpoint, key, similarity)
                for key in self.__measure_components["sensitve"].keys()
            }
        )
        df.insert(0, "viewpoint", viewpoint)
        df.insert(0, "community", "metacommunity")
        return df
