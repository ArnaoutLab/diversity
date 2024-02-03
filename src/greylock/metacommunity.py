"""Module for metacommunity and subcommunity diversity measures.

Classes
-------
Metacommunity
    Represents a metacommunity made up of subcommunities and computes
    metacommunity and subcommunity diversity measures.
"""

from typing import Callable, Iterable, Optional, Union

from pandas import DataFrame, Index, concat
from numpy import atleast_1d, broadcast_to, divide, zeros, ndarray
from greylock.exceptions import InvalidArgumentError

from greylock.abundance import make_abundance
from greylock.similarity import make_similarity
from greylock.components import make_components
from greylock.utilities import power_mean


class Metacommunity:
    """Creates diversity components and calculates diversity measures.

    A community consists of a set of species, each of which may appear
    any (non-negative) number of times. A metacommunity consists of one
    or more subcommunities and can be represented by the number of
    appearances of each species in each of the subcommunities that the
    species appears in.
    """

    MEASURES = (
        "alpha",
        "rho",
        "beta",
        "gamma",
        "normalized_alpha",
        "normalized_rho",
        "normalized_beta",
        "rho_hat",
        "beta_hat",
    )

    def __init__(
        self,
        counts: Union[DataFrame, ndarray],
        similarity: Union[DataFrame, ndarray, str, Callable, None] = None,
        X: Optional[ndarray] = None,
        chunk_size: Optional[int] = 100,
        symmetric: Optional[bool] = False,
        max_inflight_tasks: Optional[int] = 64,
    ) -> None:
        """
        Parameters
        ----------
        counts:
            2-d array with one column per subcommunity, one row per
            species, containing the count of each species in the
            corresponding subcommunities.
        similarity:
            For similarity-sensitive diversity measures. Use DataFrame
            or ndarray if the similarity matrix fits in memory. Use a
            numpy memmap or a filepath if the similarity matrix fits on
            the hard drive disk. Use a Callable function if the
            similarity matrix does not fit on the disk.
        X:
            An array, where each pair of rows will be passed to
            'similarity' if it is Callable.
        chunk_size:
            The number of file lines to process at a time when the
            similarity matrix is read from a file. Larger chunk sizes
            are faster, but take more memory.
        """
        self.counts = counts
        self.abundance = make_abundance(counts=counts)
        self.similarity = make_similarity(
            similarity=similarity,
            X=X,
            chunk_size=chunk_size,
            symmetric=symmetric,
            max_inflight_tasks=max_inflight_tasks,
        )
        self.components = make_components(
            abundance=self.abundance, similarity=self.similarity
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
        A numpy.ndarray with a diversity measure for each subcommunity.
        """
        if measure not in self.MEASURES:
            raise (
                InvalidArgumentError(
                    f"Invalid measure '{measure}'. "
                    "Argument 'measure' must be one of: "
                    f"{', '.join(self.MEASURES)}"
                )
            )
        numerator = self.components.numerators[measure]
        denominator = self.components.denominators[measure]
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
            order=1 - viewpoint,
            weights=self.abundance.normalized_subcommunity_abundance,
            items=community_ratio,
        )
        if measure in {"beta", "normalized_beta"}:
            return 1 / diversity_measure

        if measure in {"rho_hat"} and self.counts.shape[1] > 1:
            N = self.counts.shape[1]
            return (diversity_measure - 1) / (N - 1)

        if measure in {"beta_hat"} and self.counts.shape[1] > 1:
            N = self.counts.shape[1]
            return ((N / diversity_measure) - 1) / (N - 1)

        return diversity_measure

    def metacommunity_diversity(self, viewpoint: float, measure: str) -> ndarray:
        """Calculates metacommunity diversity measures.

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
        A numpy.ndarray containing the metacommunity diversity measure.
        """
        subcommunity_diversity = self.subcommunity_diversity(viewpoint, measure)
        return power_mean(
            1 - viewpoint,
            self.abundance.subcommunity_normalizing_constants,
            subcommunity_diversity,
        )

    def subcommunities_to_dataframe(self, viewpoint: float, measures=MEASURES):
        """Table containing all subcommunity diversity values.

        Parameters
        ----------
        viewpoint:
            Affects the contribution of rare species to the diversity
            measure. When viewpoint = 0 all species (rare or frequent)
            contribute equally. When viewpoint = infinity, only the
            single most frequent species contribute.

        Returns
        -------
        A pandas.DataFrame containing all subcommunity diversity
        measures for a given viewpoint
        """
        df = DataFrame(
            {
                measure: self.subcommunity_diversity(viewpoint, measure)
                for measure in measures
            }
        )
        df.insert(0, "viewpoint", viewpoint)
        df.insert(0, "community", self.abundance.subcommunities_names)
        return df

    def metacommunity_to_dataframe(self, viewpoint: float, measures=MEASURES):
        """Table containing all metacommunity diversity values.

        Parameters
        ----------
        viewpoint:
            Affects the contribution of rare species to the diversity
            measure. When viewpoint = 0 all species (rare or frequent)
            contribute equally. When viewpoint = infinity, only the
            single most frequent species contributes.

        Returns
        -------
        A pandas.DataFrame containing all metacommunity diversity
        measures for a given viewpoint
        """
        df = DataFrame(
            {
                measure: self.metacommunity_diversity(viewpoint, measure)
                for measure in measures
            },
            index=Index(["metacommunity"], name="community"),
        )
        df.insert(0, "viewpoint", viewpoint)
        df.reset_index(inplace=True)
        return df

    def to_dataframe(self, viewpoint: Union[float, Iterable[float]], measures=MEASURES):
        """Table containing all metacommunity and subcommunity diversity
        values.

        Parameters
        ----------
        viewpoint:
            Affects the contribution of rare species to the diversity
            measure. When viewpoint = 0 all species (rare or frequent)
            contribute equally. When viewpoint = infinity, only the
            single most frequent species contributes.

        Returns
        -------
        A pandas.DataFrame containing all metacommunity and subcommunity
        diversity measures for a given viewpoint
        """
        dataframes = []
        for q in atleast_1d(viewpoint):
            dataframes.append(
                self.metacommunity_to_dataframe(viewpoint=q, measures=measures)
            )
            dataframes.append(
                self.subcommunities_to_dataframe(viewpoint=q, measures=measures)
            )
        return concat(dataframes).reset_index(drop=True)
