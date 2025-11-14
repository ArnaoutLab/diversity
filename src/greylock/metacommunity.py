"""Module for metacommunity and subcommunity diversity measures.

Classes
-------
Metacommunity
    Represents a metacommunity made up of subcommunities and computes
    metacommunity and subcommunity diversity measures.
"""

from typing import Callable, Iterable, Optional, Union

from pandas import DataFrame, Index, Series, concat
from numpy import array, atleast_1d, broadcast_to, divide, zeros, ndarray, power, prod, sum as np_sum, \
identity as np_identity
from greylock.exceptions import InvalidArgumentError

from greylock.abundance import make_abundance
from greylock.similarity import Similarity, SimilarityFromArray, SimilarityIdentity, SimilarityFromFunction
from greylock.components import Components
from greylock.powermean import power_mean


class Metacommunity:
    similarity: Similarity
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
        similarity: Union[ndarray, Similarity, None] = None,
    ) -> None:
        """
        Parameters
        ----------
        counts:
            2-d array with one column per subcommunity, one row per
            species, containing the count of each species in the
            corresponding subcommunities.
        similarity:
            For small datasets this can be the similarity matrix as
            an n-by-n numpy.ndarray.
            For larger datasets, various subclasses of Similarity
            provide the similarity matrix in various memory- and compute-
            efficient way.
            If None is given here, the diversity measures calculated will
            be frequency-sensitive only, not similarity-sensitive.
        """
        self.counts = counts
        self.abundance = make_abundance(counts=counts)
        if similarity is None:
            self.similarity = SimilarityIdentity()
        elif isinstance(similarity, ndarray):
            self.similarity = SimilarityFromArray(similarity=similarity)
        else:
            self.similarity = similarity
        self.components = Components(
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
            atol=self.abundance.min_count,
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
        df.insert(0, "community", Series(self.abundance.subcommunities_names))
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
        for q in atleast_1d(array(viewpoint)):
            dataframes.append(
                self.metacommunity_to_dataframe(viewpoint=q, measures=measures)
            )
            dataframes.append(
                self.subcommunities_to_dataframe(viewpoint=q, measures=measures)
            )
        return concat(dataframes).reset_index(drop=True)

    def get_exp_renyi_div_from_ords(self, P, P_ord, Q_ord, viewpoint):
        ord_ratio = P_ord/Q_ord
        if viewpoint != 1:
            exp_renyi_div = power_mean(
                order=viewpoint-1,
                weights=P,
                items=ord_ratio,
                atol=self.abundance.min_count,
            )
        else:
            exp_renyi_div = prod(power(ord_ratio, P))
        return exp_renyi_div

    def get_exp_renyi_div_with(self, Q_abundance: Union[DataFrame, ndarray], viewpoint: float) -> float:
        """Calculate the exponentiated similarity-sensitive Renyi divergence, according to the formula
        exp-Renyi-divergence^{Z}_{q}(P || Q) = [ sum_{i} P_{i} ((ZP)_{i} / (ZQ)_{i})^{q-1}]^{1/(q-1)}
        for q not equal to 1, and
        exp-KL-divergence^{Z}(P || Q) = prod_{i} [ (ZP)_{i} / (ZQ)_{i} ]^{P_i}
        for q=1
        
        Parameters
        ----------
        Q_abundance: the abundance vector of the other metacommunity (called Q in the formula above)
        viewpoint: the viewpoint parameter.

        Returns
        -------
        A float.
        """
        if type(Q_abundance) == DataFrame:
            Q_abundance = Q_abundance.to_numpy()
        P_meta_ab = self.abundance.metacommunity_abundance
        P_norm_subcom_ab = self.abundance.normalized_subcommunity_abundance
        Q_meta_ab = np_sum(Q_abundance,axis=1, keepdims=True)
        Q_meta_ab = Q_meta_ab/np_sum(Q_meta_ab, axis=0)
        Q_norm_subcom_ab = Q_abundance/np_sum(Q_abundance,axis=0)

        P_meta_ord = self.components.metacommunity_ordinariness
        P_norm_subcom_ord = self.components.normalized_subcommunity_ordinariness
        Q_meta_ord = self.similarity.weighted_abundances(Q_meta_ab)
        Q_norm_subcom_ord = self.similarity.weighted_abundances(Q_norm_subcom_ab)

        exp_renyi_div_meta = self.get_exp_renyi_div_from_ords(P_meta_ab, P_meta_ord, Q_meta_ord, viewpoint)
        
        exp_renyi_divs_subcom = np_identity(self.abundance.num_subcommunities)
        for i in range(self.abundance.num_subcommunities):
            for j in range(self.abundance.num_subcommunities):
                P = P_norm_subcom_ab[:,i]
                P_ord = P_norm_subcom_ord[:,i]
                Q_ord = Q_norm_subcom_ord[:,j]
                exp_renyi_div = self.get_exp_renyi_div_from_ords(P, P_ord, Q_ord, viewpoint)
                exp_renyi_divs_subcom[i,j] = exp_renyi_div

        exp_renyi_divs_subcom = DataFrame(exp_renyi_divs_subcom, columns=self.abundance.subcommunities_names, \
            index=self.abundance.subcommunities_names)

        return exp_renyi_div_meta, exp_renyi_divs_subcom
