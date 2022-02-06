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

from functools import cached_property

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
        similarity: diversity.metacommunity.Similarity
            Object for calculating abundance-weighted similarities.
        abundance: diversity.metacommunity.Abundance
            Object whose (sub-/meta-)community species abundances are
            used.
        """
        self.similarity = similarity
        self.abundance = abundance
        self.measures_components = {
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
            },
        }

    @cached_property
    def metacommunity_similarity(self):
        """Sums of similarities weighted by metacommunity abundances."""
        return self.similarity.calculate_weighted_similarities(
            self.abundance.metacommunity_abundance
        )

    @cached_property
    def subcommunity_similarity(self):
        """Sums of similarities weighted by subcommunity abundances."""
        return self.similarity.calculate_weighted_similarities(
            self.abundance.subcommunity_abundance
        )

    @cached_property
    def normalized_subcommunity_similarity(self):
        """Sums of similarities weighted by the normalized subcommunity abundances."""
        return self.similarity.calculate_weighted_similarities(
            self.abundance.normalized_subcommunity_abundance
        )

    def subcommunity_measure(self, viewpoint, measure, similarity="sensitive"):
        """Calculates subcommunity diversity measures."""
        numerator, denominator = map(
            extract_data_if_shared, self.measures_components[similarity][measure]
        )
        if measure == "gamma":
            denominator = broadcast_to(
                denominator, self.abundance.subcommunity_abundance.shape
            )
        community_ratio = divide(
            numerator, denominator, out=zeros(denominator.shape), where=denominator != 0
        )
        result = power_mean(
            1 - viewpoint,
            extract_data_if_shared(self.abundance.normalized_subcommunity_abundance),
            community_ratio,
        )
        if measure in ["beta", "normalized_beta"]:
            return 1 / result
        return result

    def metacommunity_measure(self, viewpoint, measure):
        """Calculates metcommunity diversity measures."""
        subcommunity_measure = self.subcommunity_measure(viewpoint, measure)
        return power_mean(
            1 - viewpoint,
            extract_data_if_shared(self.abundance.subcommunity_normalizing_constants),
            subcommunity_measure,
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
        return DataFrame(
            {
                "community": self.abundance.subcommunity_order,
                "viewpoint": viewpoint,
                "alpha": self.subcommunity_alpha(viewpoint),
                "rho": self.subcommunity_rho(viewpoint),
                "beta": self.subcommunity_beta(viewpoint),
                "gamma": self.subcommunity_gamma(viewpoint),
                "normalized_alpha": self.normalized_subcommunity_alpha(viewpoint),
                "normalized_rho": self.normalized_subcommunity_rho(viewpoint),
                "normalized_beta": self.normalized_subcommunity_beta(viewpoint),
            }
        )

    def metacommunity_to_dataframe(self, viewpoint):
        """Table containing all metacommunity diversity values.

        Parameters
        ----------
        viewpoint: numeric
            Non-negative number. Can be interpreted as the degree of
            ignorance towards rare species, where 0 treats rare species
            the same as frequent species, and infinity considers only the
            most frequent species."""
        return DataFrame(
            {
                "community": "metacommunity",
                "viewpoint": viewpoint,
                "alpha": self.metacommunity_alpha(viewpoint),
                "rho": self.metacommunity_rho(viewpoint),
                "beta": self.metacommunity_beta(viewpoint),
                "gamma": self.metacommunity_gamma(viewpoint),
                "normalized_alpha": self.normalized_metacommunity_alpha(viewpoint),
                "normalized_rho": self.normalized_metacommunity_rho(viewpoint),
                "normalized_beta": self.normalized_metacommunity_beta(viewpoint),
            },
            index=[0],
        )


def make_metacommunity(
    counts,
    similarity_matrix,
    subcommunities=None,
    chunk_size=1,
    subcommunity_column="subcommunity",
    species_column="species",
    count_column="count",
):
    """Builds a Metacommunity object from specified parameters.

    Parameters
    ----------
    counts: numpy.ndarray or pandas.DataFrame
        See diversity.metacommunity.Abundance. If the object is a
        pandas.DataFrame, its to_numpy method should return the expected
        numpy.ndarray.
    similarity_matrix: pandas.DataFrame, or str
        For data frame, see diversity.metacommunity.SimilarityFromMemory,
        and for str, see diversity.metacommunity.SimilarityFromFile.
    subcommunities: Set
        Names of subcommunities to include. Their union is the
        metacommunity, and data for all other subcommunities is ignored.
    chunk_size: int
        Optional. See diversity.metacommunity.SimilarityFromFile.
    subcommunity_column, species_column, count_column: str
        Used to specify non-default column headers. See
        diversity.metacommunity.Abundance.

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
