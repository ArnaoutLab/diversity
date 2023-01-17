"""Module for metacommunity and subcommunity diversity measures.

Classes
-------
Metacommunity
    Represents a metacommunity made up of subcommunities and computes
    metacommunity and subcommunity diversity measures.
"""

from functools import cached_property
from typing import Callable, Iterable, Optional, Union

from pandas import DataFrame, Index, concat
from numpy import atleast_1d, broadcast_to, divide, zeros, ndarray

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
        counts: Union[DataFrame, ndarray],
        similarity: Union[DataFrame, ndarray, str, Callable, None] = None,
        X: Optional[ndarray] = None,
        chunk_size: Optional[int] = 100,
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

    def subcommunity_diversity(self, viewpoint: float, measure: str) -> ndarray:
        """Calculates subcommunity diversity measures.

        Parameters
        ----------
        viewpoint:
            Viewpoint parameter for diversity measure.
        measure:
            Name of the diversity measure. Valid measures include:
            "alpha", "rho", "beta", "gamma", "normalized_alpha",
            "normalized_rho", and "normalized_beta"

        Returns
        -------
        A numpy array with a diversity value for each subcommunity.
        """
        if measure not in self.MEASURES:
            raise (
                ValueError(
                    f"Invalid measure '{measure}'. "
                    f"Argument 'measure' must be one of: {', '.join(self.MEASURES)}"
                )
            )
        if measure in ("alpha", "gamma", "normalized_alpha"):
            numerator = 1
        if self.similarity is None:
            if measure in ("beta", "rho", "normalized_beta", "normalized_rho"):
                numerator = self.abundance.metacommunity_abundance
            if measure in ("alpha", "beta", "rho"):
                denominator = self.abundance.subcommunity_abundance
            elif measure == "gamma":
                denominator = self.abundance.metacommunity_abundance
            elif measure in ("normalized_alpha", "normalized_beta", "normalized_rho"):
                denominator = self.abundance.normalized_subcommunity_abundance
        elif isinstance(self.similarity, Similarity):
            if measure in ("beta", "rho", "normalized_beta", "normalized_rho"):
                numerator = self.metacommunity_similarity
            if measure in ("alpha", "beta", "rho"):
                denominator = self.subcommunity_similarity
            elif measure == "gamma":
                denominator = self.metacommunity_similarity
            elif measure in ("normalized_alpha", "normalized_beta", "normalized_rho"):
                denominator = self.normalized_subcommunity_similarity
        if measure == "gamma":
            denominator = broadcast_to(
                denominator,
                self.abundance.normalized_subcommunity_abundance.shape,
            )
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
            Affects the contribution of rare species to the diversity measure.
            When viewpoint = 0 all species (rare or frequent) contribute equally.
            When viewpoint = infinity, only the single most frequent species contribute.
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
            Affects the contribution of rare species to the diversity measure.
            When viewpoint = 0 all species (rare or frequent) contribute equally.
            When viewpoint = infinity, only the single most frequent species contribute.
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

    def to_dataframe(self, viewpoint: Union[float, Iterable[float]]):
        """Table containing all metacommunity and subcommunity diversity values.

        Parameters
        ----------
        viewpoint:
            Affects the contribution of rare species to the diversity measure.
            When viewpoint = 0 all species (rare or frequent) contribute equally.
            When viewpoint = infinity, only the single most frequent species contribute.
        """
        dataframes = []
        for q in atleast_1d(viewpoint):
            dataframes.append(self.metacommunity_to_dataframe(viewpoint=q))
            dataframes.append(self.subcommunities_to_dataframe(viewpoint=q))
        return concat(dataframes).reset_index(drop=True)
