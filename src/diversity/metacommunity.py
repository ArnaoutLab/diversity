"""Module for metacommunity and subcommunity diversity measures.

Classes
-------
Metacommunity
    Represents a metacommunity made up of subcommunities and computes
    metacommunity and subcommunity diversity measures.
"""

from functools import cached_property
from typing import Callable, Iterable

from pandas import DataFrame, Index, concat
from numpy import atleast_1d, broadcast_to, divide, zeros, ndarray, memmap

from diversity.log import LOGGER
from diversity.abundance import make_abundance, Abundance
from diversity.similarity import make_similarity, Similarity
from diversity.utilities import power_mean


class Metacommunity:
    """Creates diversity components and calculates diversity measures."""

    MEASURES = (
        "alpha",
        "rho",
        "beta",
        "gamma",
        "normalized_alpha",
        "normalized_rho",
        "normalized_beta",
    )

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

    @cached_property
    def metacommunity_similarity(self):
        return self.similarity.weighted_similarities(
            self.abundance.metacommunity_abundance
        )

    @cached_property
    def subcommunity_similarity(self):
        return self.similarity.weighted_similarities(
            self.abundance.subcommunity_abundance
        )

    @cached_property
    def normalized_subcommunity_similarity(self):
        return self.similarity.weighted_similarities(
            self.abundance.normalized_subcommunity_abundance
        )

    def get_measure_components(self, measure):
        match measure, self.similarity:
            case "alpha" | "gamma" | "normalized_alpha", _:
                numerator = 1
            case "beta" | "rho" | "normalized_beta" | "normalized_rho", None:
                numerator = self.abundance.metacommunity_abundance
            case "beta" | "rho" | "normalized_beta" | "normalized_rho", Similarity():
                numerator = self.metacommunity_similarity
        match measure, self.similarity:
            case "alpha" | "beta" | "rho", None:
                denominator = self.abundance.subcommunity_abundance
            case "gamma", None:
                denominator = self.abundance.metacommunity_abundance
            case "normalized_alpha" | "normalized_beta" | "normalized_rho", None:
                denominator = self.abundance.normalized_subcommunity_abundance
            case "alpha" | "beta" | "rho", Similarity():
                denominator = self.subcommunity_similarity
            case "gamma", Similarity():
                denominator = self.metacommunity_similarity
            case "normalized_alpha" | "normalized_beta" | "normalized_rho", Similarity():
                denominator = self.normalized_subcommunity_similarity
        if measure == "gamma":
            denominator = broadcast_to(
                denominator,
                self.abundance.normalized_subcommunity_abundance.shape,
            )
        return numerator, denominator

    def subcommunity_diversity(self, viewpoint: float, measure: str) -> ndarray:
        """Calculates subcommunity diversity measures.

        Parameters
        ----------
        viewpoint:
            Viewpoint parameter for diversity measure.
        measure:
            Name of the diversity measure. Valid measures
            include: "alpha", "rho", "beta", "gamma", "normalized_alpha",
            "normalized_rho", and "normalized_beta"

        Returns
        -------
        A numpy array with a diversity value for each subcommunity.
        """
        if measure not in self.MEASURES:
            raise (
                ValueError(
                    f"Invalid measure '{measure}'. Argument 'measure' must be one of: {', '.join(self.MEASURES)}"
                )
            )
        numerator, denominator = self.get_measure_components(measure)
        community_ratio = divide(
            numerator,
            denominator,
            out=zeros(denominator.shape),
            where=denominator != 0,
        )
        diversity_measure = power_mean(
            1 - viewpoint,
            self.abundance.normalized_subcommunity_abundance,
            community_ratio,
        )
        if measure in ["beta", "normalized_beta"]:
            return 1 / diversity_measure
        return diversity_measure

    def metacommunity_diversity(self, viewpoint: float, measure: str) -> ndarray:
        """Calculates metcommunity diversity measures.

        Parameters
        ----------
        viewpoint:
            Viewpoint parameter for diversity measure.
        measure:
            Name of the diversity measure. Valid measures
            include: "alpha", "rho", "beta", "gamma", "normalized_alpha",
            "normalized_rho", and "normalized_beta"

        Returns
        -------
        A numpy array containing the metacommunity diversity measure.
        """
        subcommunity_diversity = self.subcommunity_diversity(viewpoint, measure)
        return power_mean(
            1 - viewpoint,
            self.abundance.subcommunity_normalizing_constants,
            subcommunity_diversity,
        )

    def subcommunities_to_dataframe(self, viewpoint: float):
        """Table containing all subcommunity diversity values.

        Parameters
        ----------
        viewpoint:
            Non-negative number. Can be interpreted as the degree of
            ignorance towards rare species, where 0 treats rare species
            the same as frequent species, and infinity considers only the
            most frequent species.
        """
        df = DataFrame(
            {
                measure: self.subcommunity_diversity(viewpoint, measure)
                for measure in self.MEASURES
            }
        )
        df.insert(0, "viewpoint", viewpoint)
        df.insert(0, "community", self.abundance.counts.columns)
        return df

    def metacommunity_to_dataframe(self, viewpoint: float):
        """Table containing all metacommunity diversity values.

        Parameters
        ----------
        viewpoint:
            Non-negative number. Can be interpreted as the degree of
            ignorance towards rare species, where 0 treats rare species
            the same as frequent species, and infinity considers only the
            most frequent species.
        """
        df = DataFrame(
            {
                measure: self.metacommunity_diversity(viewpoint, measure)
                for measure in self.MEASURES
            },
            index=Index(["metacommunity"], name="community"),
        )
        df.insert(0, "viewpoint", viewpoint)
        df.reset_index(inplace=True)
        return df

    def to_dataframe(self, viewpoint: float | Iterable[float]):
        dataframes = []
        for q in atleast_1d(viewpoint):
            dataframes.append(self.metacommunity_to_dataframe(viewpoint=q))
            dataframes.append(self.subcommunities_to_dataframe(viewpoint=q))
        return concat(dataframes).reset_index(drop=True)
