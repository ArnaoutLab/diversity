from functools import cached_property
from numpy import ndarray

from greylock.abundance import Abundance
from greylock.similarity import Similarity


class Components:
    """Dispatches diversity components based on specified measure"""

    def __init__(self, abundance) -> None:
        self.abundance = abundance


class FrequencySensitiveComponents(Components):
    """Dispatches frequency-sensitive diversity components based on
    specified measure"""

    def __init__(self, abundance: Abundance) -> None:
        super().__init__(abundance=abundance)
        self.numerators = {
            **dict.fromkeys(["alpha", "gamma", "normalized_alpha"], 1),
            **dict.fromkeys(
                [
                    "beta",
                    "rho",
                    "normalized_beta",
                    "normalized_rho",
                    "beta_hat",
                    "rho_hat",
                ],
                self.abundance.metacommunity_abundance,
            ),
        }
        self.denominators = {
            **dict.fromkeys(
                ["alpha", "beta", "rho", "beta_hat", "rho_hat"],
                self.abundance.subcommunity_abundance,
            ),
            **dict.fromkeys(
                ["normalized_alpha", "normalized_beta", "normalized_rho"],
                self.abundance.normalized_subcommunity_abundance,
            ),
            "gamma": self.abundance.metacommunity_abundance,
        }


class SimilaritySensitiveComponents(Components):
    """Dispatches similarity-sensitive diversity components based on
    specified measure"""

    def __init__(self, abundance: Abundance, similarity: Similarity) -> None:
        """Create the weighted similarity vectors by multipying the
        similarity matrix to each of the metacommunity abundance vector,
        the subcommunity abundance vectors, and the normalized
        subcommunity vectors.
        Note that all of these vectors are unified into one matrix so
        that the similarity matrix only has to be generated and used
        once (in the case where a pre-computed similarity matrix is not
        in RAM). That is, we make only one call to weighted_similarities().
        """
        super().__init__(abundance=abundance)
        self.similarity = similarity

        all_similarity = self.similarity.weighted_similarities(
            relative_abundance=self.abundance.unified_abundance_array
        )
        self.metacommunity_similarity = all_similarity[:, [0]]
        self.subcommunity_similarity = all_similarity[
            :, 1 : (1 + self.abundance.num_subcommunities)
        ]
        self.normalized_subcommunity_similarity = all_similarity[
            :, (1 + self.abundance.num_subcommunities) :
        ]

        self.numerators = {
            **dict.fromkeys(["alpha", "gamma", "normalized_alpha"], 1),
            **dict.fromkeys(
                [
                    "beta",
                    "rho",
                    "normalized_beta",
                    "normalized_rho",
                    "beta_hat",
                    "rho_hat",
                ],
                self.metacommunity_similarity,
            ),
        }
        self.denominators = {
            **dict.fromkeys(
                ["alpha", "beta", "rho", "beta_hat", "rho_hat"],
                self.subcommunity_similarity,
            ),
            **dict.fromkeys(
                ["normalized_alpha", "normalized_beta", "normalized_rho"],
                self.normalized_subcommunity_similarity,
            ),
            "gamma": self.metacommunity_similarity,
        }


def make_components(
    abundance: Abundance,
    similarity: Similarity,
) -> Components:
    if similarity is None:
        return FrequencySensitiveComponents(abundance=abundance)
    else:
        return SimilaritySensitiveComponents(abundance=abundance, similarity=similarity)
