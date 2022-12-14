"""Module for metacommunity and subcommunity diversity measures.

Classes
-------
Metacommunity
    Represents a metacommunity made up of subcommunities and computes
    metacommunity and subcommunity diversity measures.
"""

from functools import cache
from typing import Callable

from pandas import DataFrame, Index
from numpy import broadcast_to, divide, zeros, ndarray, memmap

from diversity.log import LOGGER
from diversity.abundance import make_abundance, Abundance
from diversity.similarity import make_similarity, Similarity
from diversity.utilities import power_mean


class Metacommunity:
    """Creates diversity components and calculates diversity measures."""

    def __init__(
        self,
        counts: DataFrame | ndarray,
        similarity: DataFrame | ndarray | memmap | str | Callable | None = None,
        X: ndarray | None = None,
        chunk_size: int | None = 100,
    ) -> None:
        """
        Parameters
        ----------
        counts:
            2-d array with one column per subcommunity, one row per species,
            containing the count of each species in the corresponding subcommunities.
        similarity:
            For similarity-sensitive diversity measures. Use DataFrame or ndarray if the
            similarity matrix fits in memory. Use a numpy memmap or a filepath if the similarity matrix
            fits on the hard drive disk. Use a Callable function if the similarity matrix does not fit
            on the disk.
        X:
            An array, where each pair of rows will be passed to 'similarity' if it is Callable.
        chunk_size:
            The number of file lines to process at a time when the similarity matrix
            is read from a file. Larger chunk sizes are faster, but take more memory.
        """
        LOGGER.debug(
            "make_metacommunity(counts=%s, similarity=%s, X=%s, chunk_size=%s",
            counts,
            similarity,
            X,
            chunk_size,
        )
        self.abundance: Abundance = make_abundance(counts=counts)
        self.similarity: Similarity = make_similarity(
            similarity=similarity, X=X, chunk_size=chunk_size
        )
        self.measure_components: dict = self.make_measure_components()

    @cache
    def metacommunity_similarity(self):
        return self.similarity.weighted_similarities(
            self.abundance.metacommunity_abundance()
        )

    @cache
    def subcommunity_similarity(self):
        return self.similarity.weighted_similarities(
            self.abundance.subcommunity_abundance()
        )

    @cache
    def normalized_subcommunity_similarity(self):
        return self.similarity.weighted_similarities(
            self.abundance.normalized_subcommunity_abundance()
        )

    def make_measure_components(self) -> dict:
        if self.similarity is None:
            return {
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
        else:
            return {
                "alpha": (1, self.subcommunity_similarity),
                "rho": (
                    self.metacommunity_similarity,
                    self.subcommunity_similarity,
                ),
                "beta": (
                    self.metacommunity_similarity,
                    self.subcommunity_similarity,
                ),
                "gamma": (1, self.metacommunity_similarity),
                "normalized_alpha": (
                    1,
                    self.normalized_subcommunity_similarity,
                ),
                "normalized_rho": (
                    self.metacommunity_similarity,
                    self.normalized_subcommunity_similarity,
                ),
                "normalized_beta": (
                    self.metacommunity_similarity,
                    self.normalized_subcommunity_similarity,
                ),
            }

    def subcommunity_diversity(self, viewpoint: float, measure: str) -> ndarray:
        """Calculates subcommunity diversity measures.

        Parameters
        ----------
        viewpoint: numeric
            Viewpoint parameter for diversity measure. Valid measure identifiers
            are: "alpha", "rho", "beta", "gamma", "normalized_alpha",
            "normalized_rho", and "normalized_beta"
        measure: str
            Name of the diversity measure.

        Returns
        -------
        A numpy array with a diversity value for each subcommunity.

        Notes
        -----
        .
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

    def metacommunity_diversity(self, viewpoint: float, measure: str) -> ndarray:
        """Calculates metcommunity diversity measures."""
        subcommunity_diversity = self.subcommunity_diversity(viewpoint, measure)
        return power_mean(
            1 - viewpoint,
            self.abundance.subcommunity_normalizing_constants(),
            subcommunity_diversity,
        )

    def subcommunities_to_dataframe(self, viewpoint: float):
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
        df.insert(0, "community", self.abundance.counts.columns)
        return df

    def metacommunity_to_dataframe(self, viewpoint: float):
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
