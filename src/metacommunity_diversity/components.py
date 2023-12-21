from functools import cached_property
from numpy import ndarray

from metacommunity_diversity.abundance import Abundance
from metacommunity_diversity.similarity import Similarity


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
                ["beta", "rho", "normalized_beta", "normalized_rho", "beta_hat", "rho_hat"],
                self.abundance.metacommunity_abundance,
            ),
        }
        self.denominators = {
            **dict.fromkeys(
                ["alpha", "beta", "rho", "beta_hat", "rho_hat"], self.abundance.subcommunity_abundance
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
        super().__init__(abundance=abundance)
        self.similarity = similarity
        self.numerators = {
            **dict.fromkeys(["alpha", "gamma", "normalized_alpha"], 1),
            **dict.fromkeys(
                ["beta", "rho", "normalized_beta", "normalized_rho", "beta_hat", "rho_hat"],
                self.metacommunity_similarity,
            ),
        }
        self.denominators = {
            **dict.fromkeys(["alpha", "beta", "rho", "beta_hat", "rho_hat"], self.subcommunity_similarity),
            **dict.fromkeys(
                ["normalized_alpha", "normalized_beta", "normalized_rho"],
                self.normalized_subcommunity_similarity,
            ),
            "gamma": self.metacommunity_similarity,
        }

    @cached_property
    def metacommunity_similarity(self) -> ndarray:
        return self.similarity.weighted_similarities(
            relative_abundance=self.abundance.metacommunity_abundance
        )

    @cached_property
    def subcommunity_similarity(self) -> ndarray:
        return self.similarity.weighted_similarities(
            relative_abundance=self.abundance.subcommunity_abundance
        )

    @cached_property
    def normalized_subcommunity_similarity(self) -> ndarray:
        return self.similarity.weighted_similarities(
            relative_abundance=self.abundance.normalized_subcommunity_abundance
        )


def make_components(
    abundance: Abundance,
    similarity: Similarity,
) -> Components:
    if similarity is None:
        return FrequencySensitiveComponents(abundance=abundance)
    else:
        return SimilaritySensitiveComponents(abundance=abundance, similarity=similarity)
